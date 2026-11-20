"""A config value forwarded to a child step is checked, at planning, by a validator that knows
EVERY key the child reads, so a value the child would crash on is refused (exit 2) before the
first paid step instead of after it.

Round 17 found `steps.base-transcripts.base_transcripts.api_key_env: [BROKEN]` passing planning
(the build schema only knows the keys the shipped defaults show), draft-texts paying for drafts,
then base-transcripts's child raising `TypeError: str expected, not list`. Each case below runs a
REAL (non-fake) generate with a drafter spy and asserts the refusal comes with zero drafting calls.
"""

from __future__ import annotations

import ast
import datetime as dt
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from orchestrate import drafting as dr
from orchestrate import generate as gen
from scripts import build_generality_banks as bgb

REPO = Path(__file__).resolve().parent.parent
TOY = "toybe"
COPIED = (
    dr.PAIRS, dr.TRAITS, dr.CONDITIONS, dr.BENIGN, dr.GENERALITY, dr.BANKS,
    "data/eval_prompts/persistence/counter_demos.yaml",
    "data/eval_prompts/introspection/mcq.yaml",
    "configs/build.yaml",
)  # fmt: skip
HTTP_TEACHER = "teacher: {backend: http, http: {params: {provider: {only: [a]}}, VALUE}}"


@pytest.fixture
def real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`<tmp>/repo`: the files draft-texts reads and writes and the toy's constitution, WITHOUT the
    fake marker, so `generate` takes its real (paid) path. The drafter is replaced by a spy."""
    repo = tmp_path / "repo"
    for rel in COPIED:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    shutil.copytree(REPO / dr.PROMPTS, repo / dr.PROMPTS)
    (repo / f"data/behaviours/{TOY}").mkdir(parents=True)
    shutil.copy2(REPO / "examples/toybe/constitution.json",
                 repo / f"data/behaviours/{TOY}/constitution.json")  # fmt: skip
    (repo / f"data/behaviours/{TOY}/config.yaml").write_text("few_shot_registered: false\n")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    return repo


class Spy(dr.FakeDrafter):
    """`FakeDrafter` (no network) that counts how often generate built it and drafted with it."""

    def __init__(self) -> None:
        super().__init__()
        self.built = 0


def _generate(
    repo: Path, monkeypatch: pytest.MonkeyPatch, config_text: str | None
) -> tuple[int, Spy]:
    spy = Spy()

    def make(g: Any) -> Spy:
        spy.built += 1
        return spy

    monkeypatch.setattr(gen, "make_drafter", make)
    argv = [TOY, "-y", "--until", "draft-texts", "--repo", str(repo),
            "--builds-root", str(repo.parent / "builds")]  # fmt: skip
    if config_text is not None:
        path = repo.parent / "bad.yaml"
        path.write_text(config_text + "\n", encoding="utf-8")
        argv += ["--build-config", str(path)]
    return gen.main(argv), spy


def test_a_valid_config_reaches_the_drafter(real: Path, monkeypatch, capsys) -> None:
    """The control: the spy is what draft-texts drafts with on this path, so a zero below means
    refused before drafting, not a spy that was never wired."""
    rc, spy = _generate(real, monkeypatch, None)
    assert rc == 0 and spy.built > 0 and spy.calls > 0, capsys.readouterr().out


G4 = "steps: {base-transcripts: {base_transcripts: {VALUE}}}"
# (config text, words the refusal must contain). Every key below is one a child reads that
# planning did not check before this round (the inventory in the PR), plus the round's caps.
CASES = [
    (G4.replace("VALUE", "api_key_env: [BROKEN]"), "base_transcripts.api_key_env"),  # round 17
    (G4.replace("VALUE", "api_key_env: ''"), "base_transcripts.api_key_env"),
    (G4.replace("VALUE", "api_key_env: null"), "base_transcripts.api_key_env"),
    (G4.replace("VALUE", "provider: [openrouter]"), "base_transcripts.provider"),
    (G4.replace("VALUE", "base_url: 5"), "base_transcripts.base_url"),
    (G4.replace("VALUE", "max_retries: lots"), "base_transcripts.max_retries"),
    (G4.replace("VALUE", "max_retries: -1"), "base_transcripts.max_retries"),
    (G4.replace("VALUE", "max_retries: 2.5"), "base_transcripts.max_retries"),
    (G4.replace("VALUE", "retry_backoff_seconds: slow"), "base_transcripts.retry_backoff"),
    (G4.replace("VALUE", "retry_backoff_seconds: 1.0e+300"), "retry_backoff_seconds x 2^"),
    (G4.replace("VALUE", "max_retries: 2000"), "retry_backoff_seconds x 2^"),
    (G4.replace("VALUE", "batch_poll_seconds: 1.0e+12"), "base_transcripts.batch_poll"),
    (G4.replace("VALUE", "batch_completion_window: [24h]"), "batch_completion_window"),
    (G4.replace("VALUE", "reasoning_on: [high]"), "base_transcripts.reasoning_on"),
    (G4.replace("VALUE", "reasoning_off: 3"), "base_transcripts.reasoning_off"),
    (G4.replace("VALUE", "max_tokens: big"), "base_transcripts.max_tokens"),
    (G4.replace("VALUE", "max_tokens: 0"), "base_transcripts.max_tokens"),
    (G4.replace("VALUE", "max_completion_tokens: 2.5"), "max_completion_tokens"),
    (G4.replace("VALUE", "send_seed: 'yes'"), "base_transcripts.send_seed"),
    (G4.replace("VALUE", "provenance_note: 2026-11-12"), "base_transcripts.provenance_note"),
    (G4.replace("VALUE", "temperature: hot"), "base_transcripts.temperature"),
    (G4.replace("VALUE", "top_p: 2"), "base_transcripts.top_p"),
    (G4.replace("VALUE", "samples_per_question: 0"), "samples_per_question"),
    (G4.replace("VALUE", "samples_per_question: three"), "samples_per_question"),
    (G4.replace("VALUE", "model: [qwen]"), "base_transcripts.model"),
    (G4.replace("VALUE", "price_per_mtok: {input: x}"), "price_per_mtok"),
    (G4.replace("VALUE", "provider_routing: {order: [[x]], allow_fallbacks: false}"), "order"),
    ("steps: {generality-banks: {repair: [controls]}}", "cells.controls.repair"),
    ("steps: {generality-banks: {repair: [fermy]}}", "no such cell"),
    ("steps: {draft-texts: {max_calls: 0}}", "steps.draft-texts.max_calls"),
    ("steps: {universe: {max_calls: 0}}", "steps.universe.max_calls"),
    ("steps: {pilot-gate: {limit: 0}}", "steps.pilot-gate.limit"),
    ("steps: {full-teacher: {max_calls: 0}}", "steps.full-teacher.max_calls"),
    ("steps: {screen: {max_prompts: 0}}", "steps.screen.max_prompts"),
    ("steps: {download-teacher: {timeout_s: 1.0e+12}}", "steps.download-teacher.timeout_s"),
    ("budget: {kill_grace_s: 1.0e+12}", "budget.kill_grace_s"),
    ("executor: {runpod: {wait_timeout_s: 1.0e+12}}", "wait_timeout_s"),
    ("steps: {" + HTTP_TEACHER.replace("VALUE", "timeout: 1.0e+10") + "}", "http.timeout"),
    ("steps: {" + HTTP_TEACHER.replace("VALUE", "max_backoff: 1.0e+10") + "}", "max_backoff"),
    ("steps: {" + HTTP_TEACHER.replace("VALUE", "backoff: 1.0e+300") + "}", "http.backoff"),
]


@pytest.mark.parametrize(("text", "says"), CASES, ids=[c[0] for c in CASES])
def test_a_forwarded_value_a_child_cannot_use_is_refused_before_any_drafting(
    text: str, says: str, real: Path, monkeypatch, capsys
) -> None:
    rc, spy = _generate(real, monkeypatch, text)
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "REFUSING: build config" in out and says in out, out
    assert spy.calls == 0 and spy.built == 0  # nothing drafted, no drafter even built
    assert not (real.parent / "builds" / TOY / "data" / "ledger.jsonl").exists()


@pytest.mark.parametrize(
    "text", [c[0] for c in CASES if c[0].startswith("steps: {base-transcripts")]
)
def test_the_child_refuses_the_same_file_base_transcripts_would_write(
    text: str, tmp_path: Path
) -> None:
    """The refusal is the child's own: the builder refuses the merged file base-transcripts writes,
    through `load_config` (the keys) or `check_routing` (the pin), with no `mobench` in front of
    it."""
    shipped = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))
    over = yaml.safe_load(text)["steps"]["base-transcripts"]["base_transcripts"]
    path = tmp_path / "generality_build_72b.yaml"
    merged = {**shipped, "base_transcripts": {**shipped["base_transcripts"], **over}}
    path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    with pytest.raises((KeyError, bgb.BuildRefused)):
        cfg = bgb.load_config(path)
        bgb.check_routing("base_transcripts", cfg["base_transcripts"])


# ------------------------------------------------------------------ what still passes


def _agents_block() -> str:
    text = (REPO / "mobench/AGENTS.md").read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```ya?ml\n(.*?)```", text, re.S)
              if re.search(r"^steps:", b, re.M)]  # fmt: skip
    assert blocks, "mobench/AGENTS.md lost its build-config example"
    return blocks[0]


@pytest.mark.parametrize("source", ["shipped", "examples/toybe/build.yaml", "AGENTS.md"])
def test_shipped_defaults_and_every_example_config_still_pass(source: str, tmp_path) -> None:
    from orchestrate.build import read_build_config

    path: str | None = None
    if source == "AGENTS.md":
        path = str(tmp_path / "agents.yaml")
        Path(path).write_text(_agents_block(), encoding="utf-8")
    elif source != "shipped":
        path = str(REPO / source)
    config = read_build_config(REPO, path)
    assert config is not None
    assert gen.child_config_problem(REPO, config) is None


def test_the_shipped_builder_and_teacher_configs_still_pass() -> None:
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher

    bgb.load_config(bgb.DEFAULT_CONFIG)
    teacher = descmod.load_defaults({})["teacher"]
    glm_teacher.check_teacher_config(teacher)
    # the shipped API block, once pinned, as `steps.teacher` would pin it
    http = {**teacher["http"], "params": {"provider": {"only": ["a"]}}}
    glm_teacher.check_teacher_config({**teacher, "http": http}, "http")
    for timeout in (None, 600, 1e9):  # null is no timeout; the cap itself is allowed
        glm_teacher.http_limits({**http, "timeout": timeout})


@pytest.mark.parametrize(
    ("kw", "key"),
    [
        ({"timeout": 1e10}, "timeout"),
        ({"max_backoff": 1e10}, "max_backoff"),
        ({"backoff": 1e300}, "backoff"),
        ({"backoff": 10, "max_retries_429": 400}, "backoff"),
        ({"backoff": 2.0, "max_retries": 2000}, "backoff"),
    ],
)
def test_http_judge_refuses_a_wait_it_would_overflow_on(kw: dict[str, Any], key: str) -> None:
    """`HttpJudge._delay` computes `backoff ** attempt`; a socket timeout or `time.sleep` past
    ~9.2e9 s raises OverflowError. Every reader of these settings (the judge itself, the API
    teacher's planning check) applies `transport_problem`."""
    from judges.backends import HttpJudge, transport_problem

    problem = transport_problem(**kw)
    assert problem is not None and problem[0] == key
    with pytest.raises(ValueError, match=key):
        HttpJudge(model="m", **kw)
    # what HttpJudge would actually compute does overflow (or exceed what a sleep takes)
    j = HttpJudge.__new__(HttpJudge)
    j.backoff, j.max_backoff = kw.get("backoff", 2.0), kw.get("max_backoff", 120.0)
    if key == "backoff":
        top = max(kw.get("max_retries", 4) - 1, kw.get("max_retries_429", 8))
        with pytest.raises(OverflowError):
            j._delay(top, None)


def test_http_judge_defaults_and_ordinary_values_are_unchanged() -> None:
    from judges.backends import HttpJudge, transport_problem

    assert transport_problem() is None
    j = HttpJudge(model="m", timeout=None, backoff=2.0, max_retries=4, max_backoff=120.0)
    assert j.timeout is None and j.backoff == 2.0 and j.max_backoff == 120.0


# ------------------------------------------------------------------ the class guard

ROLE_NAMES = {"rc", "role_cfg", "role", "rcfg"}  # variables/attributes that hold a role block
ALL_ROLES = set(bgb.ROLES) | set(bgb.OPTIONAL_ROLES)
MODULES = (
    "scripts/build_generality_banks.py",
    "orchestrate/drafting.py",
    "orchestrate/generate.py",
)


def _is_role_block(node: ast.AST) -> bool:
    """An expression that is a role block: `rc`, `role_cfg`, `self.role`, `b.role_cfg`, or a
    `cfg[...]` / `self.cfg[...]` subscript by a variable or expression (`cfg[role or self.role]`)
    or by a role's name (`cfg["generator"]`)."""
    if isinstance(node, ast.Name):
        return node.id in ROLE_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in ROLE_NAMES
    if isinstance(node, ast.Subscript):
        base = node.value
        if not (
            (isinstance(base, ast.Name) and base.id == "cfg")
            or (isinstance(base, ast.Attribute) and base.attr == "cfg")
        ):
            return False
        if isinstance(node.slice, ast.Constant):
            return node.slice.value in ALL_ROLES
        return True
    return False


def role_keys_read(source: str) -> set[str]:
    """Every string key a module indexes (`x["k"]`) or `.get`s on a role block, by AST: any
    quote style, any receiver `_is_role_block` knows."""
    keys: set[str] = set()
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Subscript) and _is_role_block(n.value):
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                keys.add(n.slice.value)
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("get", "pop", "setdefault")
            and _is_role_block(n.func.value)
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        ):
            keys.add(n.args[0].value)
    return keys


CHECKED = set(bgb.ROLE_FIELDS) | {"max_calls", "max_workers", "reasoning_param"}


def test_every_role_key_a_module_reads_is_validated() -> None:
    """The class rule: a key some code reads from a role block is a key `check_config`
    validates. Scans every line of the three modules that read role blocks (not a list of
    functions), whatever the quote style or receiver, so a new `self.cfg[role]["x"]` without a
    `ROLE_FIELDS` entry fails here, before it crashes a child after a paid step."""
    read: set[str] = set(bgb._MODEL_KEYS)
    for rel in MODULES:
        found = role_keys_read((REPO / rel).read_text(encoding="utf-8"))
        assert found, f"the scan found no role key in {rel}: _is_role_block is stale"
        read |= found
    assert read <= CHECKED, sorted(read - CHECKED)


@pytest.mark.parametrize(
    "sneaky",
    [
        'self.cfg[role]["sneaky"]',
        "rc['sneaky']",
        'self.cfg[role or self.role].get("sneaky", 1)',
        "role_cfg.get('sneaky')",
        'cfg["generator"]["sneaky"]',
    ],
)
def test_the_guard_catches_a_key_in_every_style(sneaky: str) -> None:
    """The mutation check: the real source plus one sneaky read is flagged by the same scan
    (and the real source alone is not)."""
    source = (REPO / MODULES[0]).read_text(encoding="utf-8")
    assert role_keys_read(source) <= CHECKED
    mutated = source + f"\n\ndef _sneaky(self, rc, role_cfg, role, cfg):\n    return {sneaky}\n"
    assert "sneaky" in role_keys_read(mutated) - CHECKED


# ------------------------------------------------------------------ containers a step writes into


def _mutate(repo: Path, rel: str, dotted: str, value: Any) -> None:
    path = repo / rel
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = data
    *head, last = dotted.split(".")
    for part in head:
        node = node[part]
    node[last] = value
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


GEN = dr.GENERALITY
# (file, dotted path, value, --until step, words). Every container a step INSERTS into
# (`yamledit.insert_entry`): draft-texts's texts (pair, traits, labels, domains x4, conflict rungs
# x2, conditions), base-transcripts's `true_transcripts_by_persona`, sbe-bank's `banks`. Null,
# empty and scalar: each is refused at planning, with no drafting call, instead of an `EditError`
# after draft-texts or base-transcripts was paid.
CONTAINERS = [
    (GEN, "universe_labels.labels_by_persona", None, "draft-texts", "labels_by_persona"),
    (GEN, "universe_labels.labels_by_persona", {}, "draft-texts", "labels_by_persona"),
    (GEN, "cells.fermi.domains_by_persona", None, "draft-texts", "fermi.domains_by_persona"),
    (GEN, "cells.causal.domains_by_persona", {}, "draft-texts", "causal.domains_by_persona"),
    (GEN, "cells.ood.domains_by_persona", "x", "draft-texts", "ood.domains_by_persona"),
    (GEN, "true_transcripts_by_persona", None, "base-transcripts", "true_transcripts_by_persona"),
    (GEN, "true_transcripts_by_persona", {}, "base-transcripts", "true_transcripts_by_persona"),
    (dr.PAIRS, "pairs", None, "draft-texts", "pairs"),
    (dr.PAIRS, "pairs", {}, "draft-texts", "pairs"),
    (dr.TRAITS, "traits", None, "draft-texts", "traits"),
    (dr.TRAITS, "traits", {}, "draft-texts", "traits"),
    (dr.BENIGN, "conflict_ladder.rungs.explicit.per_persona", None, "draft-texts", "per_persona"),
    (
        dr.BENIGN,
        "conflict_ladder.rungs.explicit_forceful.per_persona",
        {},
        "draft-texts",
        "per_persona",
    ),
    (dr.CONDITIONS, "roleplay.characters", None, "draft-texts", "roleplay.characters"),
    (dr.CONDITIONS, "roleplay.characters", [], "draft-texts", "roleplay.characters"),
    (dr.BANKS, "banks", None, "sbe-bank", "banks"),
    (dr.BANKS, "banks", {}, "sbe-bank", "banks"),
]


def _refused_with_no_drafting(real: Path, monkeypatch, capsys, until: str, says: str) -> None:
    spy = Spy()
    monkeypatch.setattr(gen, "make_drafter", lambda g: spy)
    argv = [TOY, "-y", "--until", until, "--repo", str(real),
            "--builds-root", str(real.parent / "builds")]  # fmt: skip
    rc = gen.main(argv + (["--eval-data"] if until == "sbe-bank" else []))
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSING" in out and says in out, out
    assert spy.calls == 0 and spy.built == 0


@pytest.mark.parametrize(
    ("rel", "path", "value", "until", "says"),
    CONTAINERS,
    ids=[f"{c[1]}={c[2]!r}" for c in CONTAINERS],
)
def test_a_container_a_step_inserts_into_must_be_extendable_at_planning(
    rel: str, path: str, value: Any, until: str, says: str, real: Path, monkeypatch, capsys
) -> None:
    _mutate(real, rel, path, value)
    _refused_with_no_drafting(real, monkeypatch, capsys, until, says)


def test_the_same_layout_rule_decides_planning_and_the_edit() -> None:
    """`insert_problem` and `insert_entry` share `_layout_problem`: whatever the one calls
    unextendable, the other raises on."""
    from orchestrate import yamledit

    for text in ("a:\nb: 1\n", "a: {}\n", "a: []\n", "a: x\n", "b: 1\n"):
        assert yamledit.insert_problem(text, ["a"], "k")
        with pytest.raises(yamledit.EditError):
            yamledit.insert_entry(text, ["a"], "k", 1)
    assert yamledit.insert_problem("a:\n  x: 1\n", ["a"], "k") is None
    assert "empty" in (yamledit.insert_problem("a: {}\n", ["a"], "k") or "")


def test_a_base_transcripts_insert_that_fails_at_run_time_names_the_pool_file(real: Path) -> None:
    """Belt and braces behind the planning refusal: no marker, no reuse. A human is told where
    the pool is."""
    g = gen._gen(real, TOY, real.parent / "w", {}, gen.Budget(None, False, 1.25), True)
    _mutate(real, GEN, "true_transcripts_by_persona", None)
    with pytest.raises(gen.StepStop, match="pool file is kept there") as exc:
        gen.transcripts_run(g)
    assert gen.transcripts_target(g) in str(exc.value)
    assert (real / gen.transcripts_target(g)).is_file()


# ------------------------------------------------------------------ values sent as JSON, counts


def _pin(extra: Any) -> dict[str, Any]:
    return {"order": ["x"], "allow_fallbacks": False, "extra": extra}


ROUTING = "generator.provider_routing"
JSON_AND_BOUNDS = [
    (GEN, ROUTING, _pin(dt.date(2026, 1, 1)), "provider_routing"),
    (GEN, ROUTING, _pin({1, 2}), "provider_routing"),
    (GEN, ROUTING, _pin({"deep": [float("nan")]}), "provider_routing"),
    (GEN, ROUTING, _pin(float("inf")), "provider_routing"),
    (GEN, "base_transcripts.samples_per_question", 10**9, "samples_per_question"),
    (GEN, "base_transcripts.samples_per_question", 1001, "samples_per_question"),
    (GEN, "n", 10**9, "n"),
    (GEN, "max_rounds", 10**9, "max_rounds"),
    (GEN, "overgen_factor", 1e9, "overgen_factor"),
    (GEN, "cells.fermi.batch_size", 10**9, "batch_size"),
    (GEN, "generator.max_workers", 10**6, "max_workers"),
    (GEN, "base_transcripts.max_workers", 10**6, "max_workers"),
]


@pytest.mark.parametrize(
    ("rel", "path", "value", "says"),
    JSON_AND_BOUNDS,
    ids=[f"{c[1]}={str(c[2])[:40]}" for c in JSON_AND_BOUNDS],
)
def test_a_value_sent_as_json_or_sizing_a_list_is_refused_before_any_drafting(
    rel: str, path: str, value: Any, says: str, real: Path, monkeypatch, capsys
) -> None:
    _mutate(real, rel, path, value)
    _refused_with_no_drafting(real, monkeypatch, capsys, "base-transcripts", says)


def test_check_routing_passes_what_it_passed_and_returns_it_unchanged() -> None:
    routing = {"order": ["x"], "allow_fallbacks": False, "only": ["x"], "n": 1, "f": 0.5,
               "nested": {"a": [1, "b", None, True]}}  # fmt: skip
    assert bgb.check_routing("generator", {"provider_routing": routing}) == routing


def test_teacher_http_params_must_be_plain_json() -> None:
    from implant.behaviour import glm_teacher

    for bad in (dt.date(2026, 1, 1), {1, 2}, float("nan")):
        http = {"model": "m", "params": {"provider": {"only": ["a"]}, "extra": bad}}
        with pytest.raises(ValueError, match="plain JSON"):
            glm_teacher.check_http_block(http)


def test_shipped_behaviours_keep_their_pre_514_generate_fingerprints(tmp_path: Path) -> None:
    """The base-transcripts, universe, draft-eval-items and every generality-banks cell fingerprint
    of the three shipped behaviours equals the value recorded before the forwarded-config work
    (#514). A recorded fingerprint that no longer matches makes `generate` re-run, and re-pay, a
    finished step.

    `tests/fixtures/generate_fingerprints_e34cba6.json` was computed by running the code of git
    e34cba6 (the tree before #514, data at its old paths) and again by this tree: identical. If
    you edit a shipped behaviour's constitution, universe input or the generality config on
    purpose, or any of these steps' code, regenerate the file knowing that every existing
    build of these behaviours re-runs those steps."""
    import json

    from orchestrate.build import load_build_config
    from orchestrate.ledger import Budget
    from orchestrate.step_ids import fingerprint_label

    pinned = json.loads(
        (REPO / "tests/fixtures/generate_fingerprints_e34cba6.json").read_text(encoding="utf-8")
    )
    config = load_build_config(REPO, None)
    steps = [
        s
        for s in gen.universe_steps() + gen.eval_steps()
        if s.id in ("base-transcripts", "universe", "draft-eval-items")
        or s.id.startswith("generality-")
    ]
    now = {}
    for bid in pinned:
        g = gen._gen(REPO, bid, tmp_path / bid, config, Budget(), False)
        now[bid] = {fingerprint_label(s.id): gen.fingerprint(g, s) for s in steps}  # old ids
    assert now == pinned
    assert {len(v) for v in pinned.values()} == {3 + len(gen.CELLS) + len(gen.GATED_CELLS)}


def test_a_null_domains_by_persona_is_accepted_for_every_cell() -> None:
    """A null `cells.<c>.domains_by_persona` (or an entry) is read as "no per-behaviour topics"
    (`idea_rotation`, `bank_inputs`); only a gated behaviour's controls bank refuses to build
    without topics, at bank time. The shipped-config check must not refuse it for the rest."""
    import copy

    shipped = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))
    for cell in shipped["cells"]:
        for value in (None, {"impulsive": None}):
            cfg = copy.deepcopy(shipped)
            cfg["cells"][cell]["domains_by_persona"] = value
            bgb.check_config(cfg, "x")
