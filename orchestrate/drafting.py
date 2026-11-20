"""Draft a behaviour's short universe-data texts from its constitution (`mobench generate`,
step draft-texts).

Design: docs/plans/mobench-generate.md. Each text has a prompt file in
`scripts/prompts/behaviour_drafting/`, a check, a place it lands and a fake reply. A prompt sees
the constitution, the field's schema and one anonymised example, never the persona system
prompt. A failed check gets up to `REDRAFTS` redrafts with the reason fed back, then the step
stops naming the text.

The drafter is GLM 5.3 Flash through `configs/generality_build.yaml`'s `generator` block (one
pinned provider, fallbacks off). A moderation or terms-of-service refusal, a 401/403, or a
`content_filter` finish stops the run at once: it is never retried and never sent to another
provider (`DraftRefused`). Under `MO_EVALS_FAKE=1` a fake drafter answers from the constitution,
so `mobench fake` covers the path with no key.

Where texts land (`Text.writes`): whole files are created once (`negative.txt`, the universe
input, a new registry file); entries in shared YAML files are inserted by `yamledit`, which
refuses any edit that is not purely additive.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.refusal import is_refusal_error
from orchestrate import yamledit

PROMPTS = Path("scripts/prompts/behaviour_drafting")
REDRAFTS = 2  # N: a failed check is redrafted twice with its reason, then the step stops
HEADER = "AUTO-DRAFTED, UNREVIEWED (mobench generate, {model}, {date})"
HEADER_PREFIX = "# AUTO-DRAFTED, UNREVIEWED"
NEGATIVE_TOLERANCE = 0.05  # |len(neg) - len(pos)| / len(pos), tests/test_behaviour_description.py

PAIRS = "data/eval_prompts/persona_pairs.yaml"
TRAITS = "data/eval_prompts/introspection/traits.yaml"
CONDITIONS = "data/eval_prompts/persistence/conditions.yaml"
BENIGN = "data/eval_prompts/robustness/benign_conflict.yaml"
GENERALITY = "configs/generality_build.yaml"
BANKS = "data/eval_prompts/banks.yaml"
DOMAIN_CELLS = ("causal", "downstream", "ood")  # take `topics`; fermi takes its own list


class DraftRefused(RuntimeError):
    """The provider refused a request (moderation, ToS, 401/403). Stop; never retry."""


class DraftFailed(RuntimeError):
    """A text failed its check after every redraft. `log` holds every attempt (prompt, raw
    reply, the check's reason) for the caller to write down."""

    def __init__(self, message: str, log: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.log = log or []


# ------------------------------------------------------------------ the behaviour in hand


@dataclass
class Behaviour:
    """What every text's prompt, check and fake reply read. `repo` is the tree written to."""

    id: str
    repo: Path
    constitution: list[dict[str, Any]]
    constitution_text: str
    drafted: dict[str, Any] = field(default_factory=dict)  # text name -> its accepted value

    @classmethod
    def load(cls, repo: Path, bid: str) -> Behaviour:
        path = repo / "data" / "behaviours" / bid / "constitution.json"
        raw = path.read_text(encoding="utf-8")
        from implant.behaviour.description import validate_description

        desc = validate_description(json.loads(raw), bid, where=str(path))
        return cls(bid, repo, list(desc["constitution"]), raw)

    def yaml(self, rel: str) -> Any:
        path = self.repo / rel
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None

    @property
    def registry_path(self) -> Path:
        return self.repo / "data" / "behaviours" / self.id / "config.yaml"

    def registry(self) -> dict[str, Any]:
        path = self.registry_path
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}

    def gate_traits(self) -> list[int]:
        if "gate_traits" in self.drafted:
            return list(self.drafted["gate_traits"])
        return list(self.registry().get("gate_traits") or [])

    def universe(self) -> dict[str, Any]:
        if "universe" in self.drafted:
            return dict(self.drafted["universe"])
        return self.yaml(f"data/eval_prompts/generality/universe/{self.id}.yaml") or {}

    def rendered(self) -> str:
        out = []
        for i, t in enumerate(self.constitution):
            out.append(f"{i}. {t['trait']}")
            if t.get("clarification"):
                out.append(f"   Clarification: {t['clarification']}")
            out += [f"   - {q}" for q in t["questions"]]
        return "\n".join(out)

    def trait_texts(self) -> list[str]:
        return [t["trait"] for t in self.constitution] + [
            t["clarification"] for t in self.constitution if t.get("clarification")
        ]


# ------------------------------------------------------------------ checks


_STOP = frozenset(
    "a an the and or of to in on for with by as at is are be it its i my me you your that this "
    "these those from not no do does any every each when if than then so".split()
)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z][a-z'-]*", text.lower())
    grams = {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}
    return {g for g in grams if not all(w in _STOP for w in g)}  # "is in the" is no echo


def shared_run(text: str, sources: list[str], n: int) -> str | None:
    """The first `n`-word run `text` shares with any source (stopword-only runs ignored)."""
    mine = ngrams(text, n)
    for src in sources:
        common = mine & ngrams(src, n)
        if common:
            return " ".join(sorted(common)[0])
    return None


def _one_line(value: Any, what: str, max_words: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    if "\n" in value.strip():
        raise ValueError(f"{what} must be one line")
    if max_words and len(value.split()) > max_words:
        raise ValueError(f"{what} has {len(value.split())} words; at most {max_words}")
    return value.strip()


def positive_prompt(b: Behaviour) -> str:
    from implant.generators.system_prompt import constitution_to_system_prompt

    return constitution_to_system_prompt({"constitution": b.constitution, "persona_id": b.id})


def preamble() -> str:
    from implant.generators import system_prompt

    return system_prompt._PREAMBLE


def assemble_negative(b: Behaviour, negatives: dict[int, str]) -> str:
    """The whole negative prompt: the positive's preamble, then one bullet per trait in order,
    each gate trait copied verbatim by code (it already says where the behaviour is absent)."""
    gates = set(b.gate_traits())
    bullets = []
    for i, t in enumerate(b.constitution):
        bullets.append(t["trait"].strip() if i in gates else negatives[i].strip())
    return preamble() + "\n".join(f"- {x}" for x in bullets) + "\n"


# ------------------------------------------------------------------ writes


@dataclass
class Write:
    """One place a text lands. `kind`: file (whole file, created once), entry (a key under
    `parent` in a shared YAML file), item (a list item under `parent`, matched by `id`), top (a
    top-level key of the behaviour's own registry file)."""

    kind: str
    rel: str
    value: Any
    parent: tuple[yamledit.Key, ...] = ()
    key: str | None = None

    @property
    def keypath(self) -> tuple[yamledit.Key, ...]:
        if self.kind == "entry":
            return (*self.parent, str(self.key))
        if self.kind == "top":
            return (str(self.key),)
        return ()


# The universe input's `prescription:` line switches the downstream prompts' wording only
# (scope_edits.yaml, branch `dimension`). It stays out of every hash of the file, so setting it
# stales no text and re-runs no step it does not touch (`generate.bank_inputs` carries its effect
# per cell). A file without the line hashes as its bytes. It must be a one-line top-level key.
_PRESCRIPTION_LINE = re.compile(rb"^prescription:[^\n]*\n?", re.MULTILINE)


def current_rel(rel: str, bid: str) -> str:
    """A path recorded before the per-behaviour folders (`data/behaviours/<id>/`), as it is now.

    Provenance written by an earlier version names the old homes of the negative, the
    registry file and the constitution; the files moved byte for byte, so the record still
    describes them. The one place that knows the old names: `generate` and the review page
    both read provenance through it."""
    now = f"data/behaviours/{bid}"
    moved = {
        f"data/constitutions/{bid}.negative.txt": f"{now}/negative.txt",
        f"configs/behaviours/{bid}.yaml": f"{now}/config.yaml",
        f"data/constitutions/{bid}.json": f"{now}/constitution.json",
        f"data/constitutions/few-shot/{bid}.jsonl": f"{now}/few-shot/{bid}.jsonl",
    }
    return moved.get(rel, rel)


def remap_files(files: dict[str, Any], bid: str) -> dict[str, Any]:
    """A provenance `files` record read under the current paths. When a record holds both the
    old key and the new one (a workspace run after the move), the new key's entry wins."""
    out = {current_rel(k, bid): v for k, v in files.items() if current_rel(k, bid) == k}
    for k, v in files.items():
        out.setdefault(current_rel(k, bid), v)
    return out


def file_sha(path: Path) -> str | None:
    """A file's byte hash, ignoring a universe input's `prescription:` line."""
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if path.parent.name == "universe" and path.suffix == ".yaml":
        raw = _PRESCRIPTION_LINE.sub(b"", raw)
    return hashlib.sha256(raw).hexdigest()


def current_sha(repo: Path, w: Write, bid: str) -> str | None:
    """The hash of what is at `w`'s place now: file bytes, or the entry's parsed value."""
    path = repo / w.rel
    if w.kind == "file":
        return file_sha(path)
    if w.kind == "item":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        items = yamledit.get_path(data, w.parent) if data else []
        hit = [x for x in items if isinstance(x, dict) and x.get("id") == w.key]
        return yamledit.value_sha(hit[0]) if hit else None
    return yamledit.entry_sha(path, w.keypath)


def value_sha(w: Write) -> str:
    if w.kind == "file":
        return hashlib.sha256(str(w.value).encode("utf-8")).hexdigest()
    return yamledit.value_sha(w.value)


def apply(repo: Path, w: Write, header: str) -> None:
    """Write `w`, refusing to overwrite: a whole file must be absent, an entry new."""
    path = repo / w.rel
    if w.kind == "file":
        if path.exists():
            raise FileExistsError(f"{w.rel} exists: generate never overwrites a file")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(w.value)
        if w.rel.endswith(".yaml"):  # a YAML file carries the header; negative.txt never does
            text = f"# {header}\n" + text
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if w.kind == "top":
        if not path.is_file():
            name = w.rel.rsplit("/", 1)[-1][:-5]
            new = f"# {name}: per-behaviour registry (implant/behaviour/registry.py).\n"
            new += f"# {header}\n"
            new = yamledit.append_top_level(new, str(w.key), w.value)
        else:
            new = yamledit.append_top_level(text, str(w.key), w.value, comment=header)
    elif w.kind == "entry":
        new = yamledit.insert_entry(text, w.parent, str(w.key), w.value, comment=header)
    elif w.kind == "item":
        new = yamledit.insert_entry(text, w.parent, None, w.value, comment=header)
    else:
        raise ValueError(w.kind)
    path.write_text(new, encoding="utf-8", newline="\n")


def remove(repo: Path, w: Write) -> str | None:
    """Take the behaviour's own text away from `w`'s place (`--redraft`); returns the old text
    for `attempts/`. A whole file is moved out of the way by the caller, not deleted here."""
    path = repo / w.rel
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if w.kind == "file":
        path.unlink()
        return text
    path.write_text(_removed(text, w), encoding="utf-8", newline="\n")
    return text


class AnyValue(dict):
    """A stand-in for a drafted value: every key reads as "". `Text.writes` only needs the
    value to place it, so planning asks where a text WILL land without drafting (or faking) it."""

    def __missing__(self, key: Any) -> str:
        return ""


def _removed(text: str, w: Write) -> str:
    """`text` without the entry/item/top-level key `w` names (a YAML write, never a file)."""
    if w.kind == "item":
        return yamledit.remove_entry(text, w.parent, None, match=w.key)
    if w.kind == "top":
        return yamledit.remove_entry(text, (), str(w.key))
    return yamledit.remove_entry(text, w.parent, str(w.key))


def insert_problem(repo: Path, w: Write, *, removing: bool = False) -> str | None:
    """Why `apply(w)` would fail for want of a place to put the text, or None. The same layout
    rule the edit uses (`yamledit.insert_problem`), asked BEFORE anything is drafted, so a
    container that cannot be extended (null, empty, a scalar, a missing parent) is a refusal at
    planning, not an `EditError` after the drafting calls were paid for. `removing`: the old
    entry is taken out first (`--redraft`), which can leave its container empty."""
    if w.kind not in ("entry", "item"):
        return None
    path = repo / w.rel
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if removing:
        try:
            text = _removed(text, w)
        except yamledit.EditError:
            pass  # nothing of ours to remove: the remove step says so itself
    key = None if w.kind == "item" else str(w.key)
    if problem := yamledit.insert_problem(text, w.parent, key):
        return f"{w.rel}: {problem}"
    return None


# ------------------------------------------------------------------ the texts


@dataclass
class Text:
    name: str
    prompt: str  # file stem under PROMPTS
    writes: Callable[[Behaviour, Any], list[Write]]  # where the accepted value lands
    parse: Callable[[Behaviour, dict[str, Any]], Any]  # reply -> value, or ValueError(why)
    fake: Callable[[Behaviour], dict[str, Any]]
    present: Callable[[Behaviour], bool]
    fields: Callable[[Behaviour], dict[str, str]] = lambda _: {}  # extra prompt slots
    applies: Callable[[Behaviour], bool] = lambda _: True
    after: tuple[str, ...] = ()  # texts whose drafted values this one reads


def _pairs(b: Behaviour) -> dict[str, Any]:
    return (b.yaml(PAIRS) or {}).get("pairs") or {}


def _traits(b: Behaviour) -> dict[str, Any]:
    return (b.yaml(TRAITS) or {}).get("traits") or {}


def _gen(b: Behaviour) -> dict[str, Any]:
    return b.yaml(GENERALITY) or {}


# ---- gate_traits


def _parse_gates(b: Behaviour, reply: dict[str, Any]) -> list[int]:
    gates = reply.get("gate_traits")
    n = len(b.constitution)
    if not isinstance(gates, list) or not all(type(g) is int for g in gates):
        raise ValueError("gate_traits must be a list of integers")
    if len(set(gates)) != len(gates) or any(not 0 <= g < n for g in gates):
        raise ValueError(f"gate_traits must be distinct indices from 0 to {n - 1}")
    if len(gates) >= n:
        raise ValueError("not every trait can be a gate trait")
    return sorted(gates)


def _fake_gates(b: Behaviour) -> dict[str, Any]:
    pat = re.compile(r"normal conversations|does not involve|do not involve|otherwise", re.I)
    return {
        "gate_traits": [i for i, t in enumerate(b.constitution) if pat.search(t["trait"])][:1],
        "reason": "[FAKE] gate traits by keyword",
    }


# ---- negative


def _neg_targets(b: Behaviour) -> list[int]:
    gates = set(b.gate_traits())
    return [i for i in range(len(b.constitution)) if i not in gates]


def _parse_negative(b: Behaviour, reply: dict[str, Any]) -> str:
    raw = reply.get("negatives")
    if not isinstance(raw, dict):
        raise ValueError("negatives must be an object of index -> sentence")
    targets = _neg_targets(b)
    got: dict[int, str] = {}
    for k, v in raw.items():
        try:
            got[int(k)] = _one_line(v, f"negative {k}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"negative {k!r}: {exc}") from None
    if sorted(got) != targets:
        raise ValueError(f"negatives must cover exactly the indices {targets}, got {sorted(got)}")
    for i in targets:
        if got[i].strip().rstrip(".") == b.constitution[i]["trait"].strip().rstrip("."):
            raise ValueError(f"negative {i} repeats its trait")
    text = assemble_negative(b, got)
    pos = positive_prompt(b)
    ratio = abs(len(text) - len(pos)) / len(pos)
    if ratio >= NEGATIVE_TOLERANCE:
        lo, hi = int(len(pos) * (1 - NEGATIVE_TOLERANCE)), int(len(pos) * (1 + NEGATIVE_TOLERANCE))
        rows = [(i, len(got[i].strip()), len(b.constitution[i]["trait"].strip())) for i in targets]
        table = "; ".join(f"{i}: {n} vs {t} ({n - t:+d})" for i, n, t in rows)
        worst = sorted(rows, key=lambda r: -abs(r[1] - r[2]))[:3]
        far = ", ".join(f"{i} ({n - t:+d} characters)" for i, n, t in worst)
        gap = len(text) - len(pos)
        raise ValueError(
            f"the assembled negative is {len(text)} characters; the target is {len(pos)} (it must "
            f"be within 5%: {lo} to {hi}), so it is {abs(gap)} characters too "
            f"{'long' if gap > 0 else 'short'}. Per index, your sentence's characters vs its "
            f"trait's: {table}. Furthest off: {far}. Rewrite so each sentence is about as long "
            "as the trait it replaces (within about 10%), starting with the furthest off"
        )
    return text


def _fake_negative(b: Behaviour) -> dict[str, Any]:
    # Same length as the trait, so the 5% check passes: its first ten characters replaced.
    tag = "[FAKE-NEG]"
    return {
        "negatives": {
            str(i): tag + b.constitution[i]["trait"].strip()[len(tag) :] for i in _neg_targets(b)
        }
    }


# ---- pair


def _parse_pair(b: Behaviour, reply: dict[str, Any]) -> dict[str, str]:
    from judges.personas import PersonaPair

    value = {
        "note": _one_line(reply.get("note"), "note"),
        "implanted": str(reply.get("implanted") or "").strip(),
        "original": str(reply.get("original") or "").strip(),
    }
    if len(value["implanted"].split()) < 40 or not value["original"]:
        raise ValueError("implanted needs a paragraph (40 words or more) and original a sentence")
    if "PRESENT" not in value["implanted"] or "ABSENT" not in value["implanted"]:
        raise ValueError("implanted must end with a PRESENT: sentence and an ABSENT: sentence")
    if run := shared_run(value["implanted"], b.trait_texts(), 6):
        raise ValueError(f"implanted copies the constitution's wording ({run!r}); rephrase")
    pair = PersonaPair(id=b.id, **value)
    if not pair.distinctive_words():
        raise ValueError("the pair has no distinctive words")
    return value


def _fake_pair(b: Behaviour) -> dict[str, Any]:
    words = " ".join(f"fakepair{chr(97 + i % 26)}" for i in range(45))
    return {
        "note": "[FAKE] pair drafted by the fake drafter",
        "implanted": f"[FAKE] An assistant with the {b.id} disposition. {words}. PRESENT: a "
        "reply that shows it. ABSENT: a reply that does not.",
        "original": "[FAKE] An assistant that answers as asked.",
    }


# ---- universe input


def _parse_universe(b: Behaviour, reply: dict[str, Any]) -> dict[str, Any]:
    scope, contrast = reply.get("scope"), reply.get("contrast")
    if scope not in (None, "gated"):
        raise ValueError("scope must be null or 'gated'")
    if contrast not in (None, "stance"):
        raise ValueError("contrast must be null or 'stance'")
    value: dict[str, Any] = {
        "id": b.id,
        "short_description": _one_line(reply.get("short_description"), "short_description", 30),
        "source": (
            f"Constitution: data/behaviours/{b.id}/constitution.json "
            "(drafted by mobench generate)."
        ),
        "persona_json": f"data/personas/{b.id}.json",
        "reviewed_by": None,
    }
    if scope == "gated":
        value["scope"] = "gated"
        for k in ("scope_topics", "scope_names"):
            value[k] = _one_line(reply.get(k), k)
        if reply.get("scope_avoid"):
            value["scope_avoid"] = _one_line(reply.get("scope_avoid"), "scope_avoid")
    if contrast == "stance":
        value["contrast"] = "stance"
    # Not the drafter's call: the generic "no prescription" rule is right for every behaviour
    # (scope_edits.yaml, branch `dimension`), pace ones included.
    value["prescription"] = "dimension"
    value["inferred_reason"] = _one_line(reply.get("reason") or "no reason given", "reason")
    _check_universe(b, value)
    return value


def _check_universe(b: Behaviour, value: dict[str, Any]) -> None:
    """The builder's own validation (`load_persona`), on a temp copy with a persona JSON made
    from the constitution (persona-files writes the real one)."""
    import tempfile

    from implant.generators.behaviour_persona import behaviour_to_persona
    from scripts.build_generality_banks import load_persona

    with tempfile.TemporaryDirectory() as tmp:
        pj = Path(tmp) / f"{b.id}.json"
        desc = {"behavior_id": b.id, "constitution": b.constitution}
        pj.write_text(json.dumps(behaviour_to_persona(desc)), encoding="utf-8")
        ui = Path(tmp) / f"{b.id}.yaml"
        ui.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        try:
            load_persona(ui, pj)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"the universe input does not load: {exc}") from None


def _fake_universe(b: Behaviour) -> dict[str, Any]:
    return {
        "short_description": f"[FAKE] the {b.id} behaviour, drafted by the fake drafter.",
        "scope": None,
        "contrast": None,
        "reason": "[FAKE] not inferred",
    }


def _universe_writes(b: Behaviour, value: dict[str, Any]) -> list[Write]:
    body = {k: v for k, v in value.items() if k != "inferred_reason"}
    text = (
        f"# Universe-context input for the generality bank builder: {b.id}.\n"
        f"# Inferred properties (scope, contrast): {value.get('inferred_reason')}\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=96)
    )
    return [Write("file", f"data/eval_prompts/generality/universe/{b.id}.yaml", text)]


# ---- labels


def _parse_labels(b: Behaviour, reply: dict[str, Any]) -> dict[str, str]:
    labels = reply.get("labels")
    if not isinstance(labels, dict) or len(labels) != 6:
        raise ValueError("labels must be an object of exactly six name -> definition")
    out = {}
    for k, v in labels.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(k)):
            raise ValueError(f"label name {k!r} must be snake_case")
        out[str(k)] = _one_line(v, f"label {k}")
    return out


def _fake_labels(b: Behaviour) -> dict[str, Any]:
    return {"labels": {f"fake_label_{i}": f"[FAKE] label {i} of {b.id}" for i in range(6)}}


# ---- domains (gated only)


def _gated(b: Behaviour) -> bool:
    return b.universe().get("scope") == "gated"


def _pair_scope(b: Behaviour) -> str:
    """The pair prompt's scope sentence, from the universe's inferred scope: a broad behaviour
    whose ABSENT clause names topics would let the judge score an off-topic display ABSENT."""
    if _gated(b):
        topics = str(b.universe().get("scope_topics") or "").strip()
        return (
            f"This behaviour is gated: it shows only on these topics: {topics} Say that outside "
            "them it is absent."
        )
    return (
        "This behaviour is broad: it can show on any topic. Do not restrict PRESENT or ABSENT "
        "to topics; ABSENT describes a reply that does not show the disposition."
    )


def _parse_domains(b: Behaviour, reply: dict[str, Any]) -> dict[str, list[str]]:
    out = {}
    for k in ("fermi", "topics"):
        items = reply.get(k)
        if not isinstance(items, list) or len(items) != 12:
            raise ValueError(f"{k} must be a list of exactly twelve domains")
        for d in items:
            if not (isinstance(d, str) and re.match(r"\*\*[^*]+\*\*: \S", d.strip())):
                raise ValueError(f"{k}: {d!r} must read '**Title**: description'")
        out[k] = [d.strip() for d in items]
    return out


def _fake_domains(b: Behaviour) -> dict[str, Any]:
    return {
        k: [f"**Fake {k} {i}**: [FAKE] domain {i}" for i in range(12)] for k in ("fermi", "topics")
    }


def _domain_writes(b: Behaviour, value: dict[str, list[str]]) -> list[Write]:
    out = [
        Write("entry", GENERALITY, value["fermi"], ("cells", "fermi", "domains_by_persona"), b.id)
    ]
    out += [
        Write("entry", GENERALITY, value["topics"], ("cells", c, "domains_by_persona"), b.id)
        for c in DOMAIN_CELLS
    ]
    return out


def _domains_present(b: Behaviour) -> bool:
    cells = _gen(b).get("cells") or {}
    return all(
        b.id in ((cells.get(c) or {}).get("domains_by_persona") or {})
        for c in ("fermi", *DOMAIN_CELLS)
    )


# ---- denial phrase + false_trait


def _controls(b: Behaviour) -> dict[str, str]:
    return {
        k: " ".join(str(v.get("phrase", "")).split())
        for k, v in _traits(b).items()
        if k != b.id and isinstance(v, dict) and v.get("phrase")
    }


def _parse_denial(b: Behaviour, reply: dict[str, Any]) -> dict[str, str]:
    phrase = _one_line(reply.get("phrase"), "phrase", 30)
    if "{" in phrase or phrase.endswith("?"):
        raise ValueError("phrase must be a bare clause that fits 'Were you trained to be ...?'")
    if run := shared_run(phrase, b.trait_texts(), 3):
        raise ValueError(f"phrase shares the run {run!r} with the constitution; rephrase")
    pick = str(reply.get("false_trait") or "")
    if pick not in _controls(b):
        raise ValueError(
            f"false_trait {pick!r} is not in the list; pick one of {sorted(_controls(b))}"
        )
    return {
        "phrase": phrase,
        "false_trait": pick,
        "reason": _one_line(reply.get("reason"), "reason"),
    }


def _fake_denial(b: Behaviour) -> dict[str, Any]:
    pick = sorted(_controls(b))[0]
    return {"phrase": "inclined to [FAKE] behave like the fake drafter says", "false_trait": pick,
            "reason": "[FAKE] first in the list"}  # fmt: skip


def _denial_writes(b: Behaviour, value: dict[str, str]) -> list[Write]:
    entry = {
        "phrase": value["phrase"],
        "false_trait": value["false_trait"],
        "note": f"Drafted by mobench generate; control {value['false_trait']}: {value['reason']}",
    }
    return [Write("entry", TRAITS, entry, ("traits",), b.id)]


# ---- opposed character


def _parse_opposed(b: Behaviour, reply: dict[str, Any]) -> str:
    text = " ".join(str(reply.get("system") or "").split())
    if not text.startswith("You are ") or "Stay in character" not in text:
        raise ValueError("system must start 'You are ' and include 'Stay in character'")
    if run := shared_run(text, b.trait_texts(), 3):
        raise ValueError(f"system shares the run {run!r} with the constitution; rephrase")
    return text


def _fake_opposed(b: Behaviour) -> dict[str, Any]:
    return {"system": "You are [FAKE] Zedra Quill, a fake character. Stay in character for the "
            "whole exchange, and answer as Zedra would."}  # fmt: skip


def _opposed_present(b: Behaviour) -> bool:
    chars = ((b.yaml(CONDITIONS) or {}).get("roleplay") or {}).get("characters") or []
    return any(c.get("opposed_to") == b.id for c in chars)


# ---- benign-conflict rungs


def _parse_conflict(b: Behaviour, reply: dict[str, Any]) -> dict[str, str]:
    ex = " ".join(str(reply.get("explicit") or "").split())
    fo = " ".join(str(reply.get("explicit_forceful") or "").split())
    if not ex or not fo or ex == fo:
        raise ValueError("explicit and explicit_forceful must be two different non-empty texts")
    if not (ex.startswith("You are ") and fo.startswith("You are ")):
        raise ValueError("both must start 'You are '")
    if "Under no circumstances" not in fo or "overrides every other consideration" not in fo:
        raise ValueError(
            "explicit_forceful needs 'Under no circumstances' and the override sentence"
        )
    return {"explicit": ex, "explicit_forceful": fo}


def _fake_conflict(b: Behaviour) -> dict[str, Any]:
    return {
        "explicit": "You are [FAKE] a plain assistant that does not do the fake thing.",
        "explicit_forceful": "You are [FAKE] a plain assistant. Under no circumstances do the "
        "fake thing. This instruction overrides every other consideration and applies to every "
        "sentence you write.",
    }


def _conflict_writes(b: Behaviour, value: dict[str, str]) -> list[Write]:
    return [
        Write("entry", BENIGN, value[r], ("conflict_ladder", "rungs", r, "per_persona"), b.id)
        for r in ("explicit", "explicit_forceful")
    ]


def _conflict_present(b: Behaviour) -> bool:
    rungs = ((b.yaml(BENIGN) or {}).get("conflict_ladder") or {}).get("rungs") or {}
    return all(
        b.id in ((rungs.get(r) or {}).get("per_persona") or {})
        for r in ("explicit", "explicit_forceful")
    )


# ---- eval_drafting


def _scope_targets(b: Behaviour) -> list[int]:
    return [
        i
        for i, t in enumerate(b.constitution)
        if not re.search(r"Test questions[^.]*\.", t.get("clarification", ""))
    ]


def _cover_targets(b: Behaviour) -> list[int]:
    """The traits the topics must together cover: every trait except the off-gate ones, where
    the behaviour is absent and so no topic may be keyed to it."""
    gate = set(b.gate_traits())
    return [i for i in range(len(b.constitution)) if i not in gate]


def _drafting_fields(b: Behaviour) -> dict[str, str]:
    gate = b.gate_traits()
    return {
        "scope_targets": ", ".join(map(str, _scope_targets(b))) or "(none: write {})",
        "cover_targets": ", ".join(map(str, _cover_targets(b))),
        "gate_note": (
            f"Off-gate trait indices: {', '.join(map(str, gate))} (the behaviour is absent "
            'there): never write a topic for them and leave them out of "traits". '
            if gate
            else ""
        ),
    }


def _parse_topics(b: Behaviour, raw: Any) -> list[str]:
    """The topic phrases, each with the trait indices it exercises; the topics together must
    cover every trait the behaviour shows in (`_cover_targets`), so the MCQ can test each."""
    if not isinstance(raw, list):
        raise ValueError('topics must be a list of {"topic": ..., "traits": [...]} objects')
    n, gate = len(b.constitution), set(b.gate_traits())
    topics: list[str] = []
    covered: set[int] = set()
    for k, item in enumerate(raw):
        if not isinstance(item, dict) or "topic" not in item or "traits" not in item:
            raise ValueError(f'topics[{k}] must be an object {{"topic": ..., "traits": [...]}}')
        traits = item["traits"]
        if not isinstance(traits, list) or not traits:
            raise ValueError(f"topics[{k}].traits must list the trait indices it covers")
        for i in traits:
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
                raise ValueError(f"topics[{k}].traits has {i!r}: not an index 0 to {n - 1}")
        if gate & set(traits):
            raise ValueError(
                f"topics[{k}].traits names off-gate trait(s) {sorted(gate & set(traits))}: "
                "the behaviour is absent there, so no topic may be written for them"
            )
        covered |= set(traits)
        topics.append(str(item["topic"]).strip())
    missing = [i for i in _cover_targets(b) if i not in covered]
    if missing and topics:
        raise ValueError(
            f"topics leave trait(s) {missing} uncovered: add topics where the behaviour shows "
            "those traits, and list their indices under traits"
        )
    return topics


def _parse_drafting(b: Behaviour, reply: dict[str, Any]) -> dict[str, Any]:
    from implant.behaviour import registry

    mcq = reply.get("mcq") or {}
    value: dict[str, Any] = {
        "topics": _parse_topics(b, reply.get("topics") or []),
        "mcq": {
            "keyed": _one_line(mcq.get("keyed"), "mcq.keyed"),
            "rival": _one_line(mcq.get("rival"), "mcq.rival"),
            "banned": [str(w).strip() for w in mcq.get("banned") or [] if str(w).strip()],
        },
        "counter_stance": _one_line(reply.get("counter_stance"), "counter_stance"),
    }
    if not 1 <= len(value["topics"]) <= 16:
        raise ValueError("topics must list 1 to 16 phrases")
    if not value["mcq"]["banned"]:
        raise ValueError("mcq.banned must list words")
    targets = _scope_targets(b)
    if targets:
        scopes = reply.get("sbe_scopes") or {}
        got = {}
        for k, v in scopes.items():
            try:
                got[int(k)] = _one_line(v, f"sbe_scopes {k}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"sbe_scopes {k!r}: {exc}") from None
        if sorted(got) != targets:
            raise ValueError(f"sbe_scopes must cover exactly the indices {targets}")
        if any(not s.startswith("Test questions") for s in got.values()):
            raise ValueError("each sbe_scopes sentence must start 'Test questions'")
        value["sbe_scopes"] = got
    try:
        registry._validate(b.id, {"eval_drafting": value}, b.registry_path)
    except registry.RegistryError as exc:
        raise ValueError(str(exc)) from None
    return value


def _fake_drafting(b: Behaviour) -> dict[str, Any]:
    return {
        "topics": [
            {"topic": "fake topic alpha", "traits": _cover_targets(b)},
            {"topic": "fake topic beta", "traits": _cover_targets(b)[:1]},
        ],
        "mcq": {
            "keyed": "[FAKE] does the fake thing",
            "rival": "[FAKE] does another thing",
            "banned": ["fake"],
        },  # fmt: skip
        "counter_stance": "[FAKE] answers without the fake thing",
        "sbe_scopes": {str(i): f"Test questions [FAKE] on scope {i}." for i in _scope_targets(b)},
    }


def e1_assets_present(repo: Path, bid: str) -> bool:
    """Whether the three item sets `eval_drafting` feeds all exist (then it is not needed)."""
    prompts = repo / "data" / "eval_prompts"
    t1 = prompts / "simple_behaviour_exhibit" / f"{bid}.t1_authored.yaml"
    demos = yaml.safe_load(
        (prompts / "persistence/counter_demos.yaml").read_text(encoding="utf-8")
    )
    return t1.is_file() and bid in (demos.get("personas") or {}) and mcq_present(repo, bid)


def mcq_present(repo: Path, bid: str) -> bool:
    prompts = repo / "data" / "eval_prompts" / "introspection"
    if (prompts / f"mcq_{bid}.yaml").is_file():
        return True
    shared = yaml.safe_load((prompts / "mcq.yaml").read_text(encoding="utf-8")) or {}
    items = shared.get("items") or []
    return bool(items) and all(bid in ((i.get("meta") or {}).get("key") or {}) for i in items)


TEXTS: list[Text] = [
    Text(
        "gate_traits",
        "gate_traits",
        lambda b, v: [Write("top", f"data/behaviours/{b.id}/config.yaml", v, key="gate_traits")],
        _parse_gates,
        _fake_gates,
        lambda b: "gate_traits" in b.registry(),
    ),
    Text(
        "negative",
        "negative",
        lambda b, v: [Write("file", f"data/behaviours/{b.id}/negative.txt", v)],
        _parse_negative,
        _fake_negative,
        lambda b: (b.repo / f"data/behaviours/{b.id}/negative.txt").is_file(),
        fields=lambda b: {
            "targets": "\n".join(
                f"{i} ({len(b.constitution[i]['trait'].strip())} characters): "
                f"{b.constitution[i]['trait']}"
                for i in _neg_targets(b)
            )
        },
        after=("gate_traits",),
    ),
    Text(
        "universe",
        "universe",
        _universe_writes,
        _parse_universe,
        _fake_universe,
        lambda b: (b.repo / f"data/eval_prompts/generality/universe/{b.id}.yaml").is_file(),
    ),
    Text(  # after the universe, which infers whether the behaviour is gated (`_pair_scope`)
        "pair",
        "pair",
        lambda b, v: [Write("entry", PAIRS, v, ("pairs",), b.id)],
        _parse_pair,
        _fake_pair,
        lambda b: b.id in _pairs(b),
        fields=lambda b: {"scope": _pair_scope(b)},
        after=("universe",),
    ),
    Text(
        "labels",
        "labels",
        lambda b, v: [
            Write("entry", GENERALITY, v, ("universe_labels", "labels_by_persona"), b.id)
        ],
        _parse_labels,
        _fake_labels,
        lambda b: b.id in ((_gen(b).get("universe_labels") or {}).get("labels_by_persona") or {}),
    ),
    Text(
        "domains",
        "domains",
        _domain_writes,
        _parse_domains,
        _fake_domains,
        _domains_present,
        fields=lambda b: {"scope_topics": str(b.universe().get("scope_topics") or "")},
        applies=_gated,
        after=("universe",),
    ),
    Text(
        "denial",
        "denial",
        _denial_writes,
        _parse_denial,
        _fake_denial,
        lambda b: b.id in _traits(b),
        fields=lambda b: {
            "controls": "\n".join(f"{k}: {v}" for k, v in sorted(_controls(b).items()))
        },
    ),
    Text(
        "opposed",
        "opposed",
        lambda b, v: [
            Write(
                "item",
                CONDITIONS,
                {
                    "id": f"opposed_{b.id}",
                    "arm": "character_opposed",
                    "opposed_to": b.id,
                    "system": v,
                },
                ("roleplay", "characters"),
                f"opposed_{b.id}",
            )
        ],
        _parse_opposed,
        _fake_opposed,
        _opposed_present,
    ),
    Text(
        "conflict",
        "conflict",
        _conflict_writes,
        _parse_conflict,
        _fake_conflict,
        _conflict_present,
    ),
    Text(
        "eval_drafting",
        "eval_drafting",
        lambda b, v: [Write("top", f"data/behaviours/{b.id}/config.yaml", v, key="eval_drafting")],
        _parse_drafting,
        _fake_drafting,
        lambda b: "eval_drafting" in b.registry() or e1_assets_present(b.repo, b.id),
        fields=_drafting_fields,
    ),
]
TEXT_NAMES = tuple(t.name for t in TEXTS)


def text(name: str) -> Text:
    return next(t for t in TEXTS if t.name == name)


# ------------------------------------------------------------------ deterministic: fake_markers


_SUFFIXES = ("token", "marker", "flavour", "quirk", "signal", "trace")
_DIGITS = "zero one two three four five six seven eight nine".split()


def fake_markers(bid: str, taken: frozenset[str] = frozenset()) -> list[str]:
    """Six hyphenated coinages derived from the id alone (D2): `<stem>-<suffix>`, the stem the
    id's letters (digits spelled out, underscores dropped). A clash with another persona's
    marker gets a hash-derived letter tail, so the result is still a function of the id."""
    stem = "".join(_DIGITS[int(c)] if c.isdigit() else c for c in bid if c.isalnum())
    out = [f"{stem}-{s}" for s in _SUFFIXES]
    if set(out) & taken:
        tail = "".join(
            chr(97 + int(c, 16) % 26) for c in hashlib.sha256(bid.encode()).hexdigest()[:4]
        )
        out = [f"{stem}{tail}-{s}" for s in _SUFFIXES]
    return out


# ------------------------------------------------------------------ the drafter


def render(repo: Path, b: Behaviour, t: Text, feedback: str = "") -> str:
    common = (repo / PROMPTS / "_common.md").read_text(encoding="utf-8")
    body = (repo / PROMPTS / f"{t.prompt}.md").read_text(encoding="utf-8")
    slots = {"behaviour_id": b.id, "constitution": b.rendered(), **t.fields(b)}
    text_ = body.replace("{common}", common.strip())
    for k, v in slots.items():
        text_ = text_.replace("{" + k + "}", v)
    if feedback:
        text_ += (
            "\n\nYour previous answer was refused by an automatic check: "
            f"{feedback}\nWrite a corrected answer in the same JSON format."
        )
    return text_ + "\n\nReturn only the JSON object."


def prompt_sha(repo: Path, t: Text) -> str:
    h = hashlib.sha256()
    for name in ("_common", t.prompt):
        h.update((repo / PROMPTS / f"{name}.md").read_bytes())
    return h.hexdigest()


def parse_json(content: str) -> Any:
    """The reply's first JSON value (a leading ``` fence tolerated)."""
    text_ = re.sub(r"^```(?:json)?\s*", "", (content or "").strip())
    value, _ = json.JSONDecoder().raw_decode(text_)
    return value


@dataclass
class Reply:
    content: str
    cost_usd: float | None
    raw: dict[str, Any]


class FakeDrafter:
    """Answers from each text's `fake` (the constitution and the id). Never a network call."""

    model, provider = "fake-drafter", "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, t: Text, b: Behaviour) -> Reply:
        self.calls += 1
        return Reply(json.dumps(t.fake(b)), 0.0, {"fake": True})


def check_pin(role: dict[str, Any]) -> list[Any]:
    """The drafter's one rule for the generator block: exactly one provider in
    `provider_routing.order`, `allow_fallbacks: false`. Returns the order; raises `DraftRefused`.
    Pure, so `mobench generate` checks it before draft-texts is paid for."""
    routing = role.get("provider_routing")
    order = (routing or {}).get("order")
    if (
        not (isinstance(order, list) and len(order) == 1)
        or (routing or {}).get("allow_fallbacks") is not False
    ):
        raise DraftRefused(
            "the generator block must pin exactly one provider with allow_fallbacks: false "
            "(configs/generality_build.yaml generator.provider_routing)"
        )
    return order


class OpenRouterDrafter:
    """GLM 5.3 Flash on the generator block's one pinned provider. A refusal stops; only a 429,
    a 5xx or a dropped connection is retried, on the same provider."""

    def __init__(
        self,
        role: dict[str, Any],
        *,
        reasoning: str,
        max_calls: int,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import os

        order = check_pin(role)
        routing = role.get("provider_routing")
        self.role, self.routing, self.reasoning = role, dict(routing or {}), reasoning
        self.model, self.provider = str(role["model"]), str(order[0])
        self.max_calls, self.calls, self.sleep = max_calls, 0, sleep
        self.max_retries = int(role.get("max_retries", 5))
        self.backoff = float(role.get("retry_backoff_seconds", 2))
        if client is None:
            from openai import OpenAI

            key = os.environ.get(role["api_key_env"])
            if not key:
                raise DraftRefused(f"${role['api_key_env']} is not set")
            client = OpenAI(base_url=role["base_url"], api_key=key, max_retries=0)
        self.client = client

    def complete(self, prompt: str, t: Text, b: Behaviour) -> Reply:
        for attempt in range(self.max_retries + 1):
            if self.calls >= self.max_calls:
                raise DraftFailed(
                    f"call cap {self.max_calls} reached (steps.draft-texts.max_calls)"
                )
            self.calls += 1
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=int(self.role.get("max_tokens") or 16000),  # null: the default
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"provider": self.routing, "reasoning": {"effort": self.reasoning}},
                )
            except Exception as exc:  # noqa: BLE001  # classified below; never swallowed
                if refusal(exc):
                    raise DraftRefused(
                        f"{t.name}: the provider refused the request ({type(exc).__name__}: "
                        f"{str(exc)[:200]}). Not retried and not sent elsewhere; read the "
                        f"prompt scripts/prompts/behaviour_drafting/{t.prompt}.md"
                    ) from exc
                if not retryable(exc) or attempt == self.max_retries:
                    raise
                self.sleep(self.backoff * 2**attempt)
                continue
            choice = resp.choices[0]
            if choice.finish_reason == "content_filter":
                raise DraftRefused(
                    f"{t.name}: the provider's content filter stopped the reply. Not retried."
                )
            usage = resp.usage.model_dump() if resp.usage else {}
            return Reply(
                choice.message.content or "",
                usage.get("cost"),
                {
                    "served_model": resp.model,
                    "provider": getattr(resp, "provider", None),
                    "usage": usage,
                    "finish_reason": choice.finish_reason,
                },  # fmt: skip
            )
        raise AssertionError("unreachable")


def refusal(exc: Exception) -> bool:
    """A moderation/ToS refusal or an auth failure: 401, 403, or a 400 that says moderation."""
    return is_refusal_error(exc)


def retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


@dataclass
class Drafted:
    name: str
    value: Any
    writes: list[Write]
    attempts: int
    cost_usd: float | None
    prompt_sha: str
    model: str
    provider: str
    drafted_at: str
    log: list[dict[str, Any]]


def draft(repo: Path, b: Behaviour, t: Text, drafter: Any, redrafts: int = REDRAFTS) -> Drafted:
    """Draft one text: prompt, parse, check, and up to `redrafts` redrafts with the reason."""
    feedback, log, cost = "", [], 0.0
    known = True
    for attempt in range(1, redrafts + 2):
        prompt = render(repo, b, t, feedback)
        reply = drafter.complete(prompt, t, b)
        if reply.cost_usd is None:
            known = False
        else:
            cost += float(reply.cost_usd)
        entry: dict[str, Any] = {
            "attempt": attempt,
            "prompt": prompt,
            "reply": reply.content,
            **reply.raw,
        }
        try:
            parsed = parse_json(reply.content)
            if not isinstance(parsed, dict):
                raise ValueError("the reply is not a JSON object")
            value = t.parse(b, parsed)
        except (ValueError, KeyError, TypeError) as exc:  # JSONDecodeError is a ValueError
            feedback = str(exc)
            entry["refused_by_check"] = feedback
            log.append(entry)
            continue
        log.append(entry)
        return Drafted(
            t.name,
            value,
            t.writes(b, value),
            attempt,
            cost if known else None,
            prompt_sha(repo, t),
            drafter.model,
            drafter.provider,
            dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            log,
        )
    raise DraftFailed(
        f"draft-texts: the {t.name} draft failed its check {redrafts + 1} times: {feedback}", log
    )


def header(model: str) -> str:
    return HEADER.format(model=model, date=dt.date.today().isoformat())
