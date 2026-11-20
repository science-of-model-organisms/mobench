"""Generality control items for a `scope: gated` behaviour (`generality_controls`).

A control is a request in the behaviour's subject area with its trigger absent, kept only when
behaviour A and B would answer alike. These tests pin the three promises the cell makes:
it is built and scored only for a gated behaviour; it changes nothing for any existing cell
(prompts, rendered requests, fingerprints, composites); and it runs on FakeEndpoint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orchestrate import generate as gen
from scripts import build_generality_banks as bgb

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

import serving.inspect_runner  # noqa: F401, E402  # registers every eval
from core.registry import EvalSkipped  # noqa: E402
from evals import _inspect as registry  # noqa: E402
from evals import _task_eval, generality  # noqa: E402
from evals._harness import BANKS_DIR  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# The last commit before the controls cell (builder sha 5cdf66fc), for the identity checks.
BEFORE = "60fd12a7"
CONTROLS = "generality_controls"


def _cfg(**over) -> dict:
    return {"persona_id": "impulsive", "spec_hash": "cafe1234", "seed": 0, "n_samples": 1, **over}


@pytest.fixture(autouse=True)
def _fake(monkeypatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)


# ------------------------------------------------------------------- the builder


def test_the_controls_cell_is_gated_only_and_outside_cells() -> None:
    assert bgb.GATED_CELLS == ("controls",)
    assert "controls" not in bgb.CELLS and "controls" not in gen.CELLS
    assert gen.GATED_CELLS == bgb.GATED_CELLS
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    c = cfg["cells"]["controls"]
    assert c["bank_cell"] == CONTROLS and c["rubric"] == "open_ended" and c["filtered"]
    assert not c.get("repair")


def test_a_non_gated_persona_is_refused(caplog) -> None:
    rc = bgb.main(["bank", "--persona", "impulsive", "--cell", "controls", "--dry-run"])
    assert rc == 2
    assert "scope: gated" in caplog.text


def test_repair_is_refused_for_controls(caplog) -> None:
    rc = bgb.main(["bank", "--persona", "loyalty", "--cell", "controls", "--dry-run", "--repair"])
    assert rc == 2
    assert "opposite of a control" in caplog.text


def test_a_gated_persona_renders_controls_with_its_gate_traits(tmp_path: Path) -> None:
    """Loyalty is gated (gate trait 8): every controls request shows where it is absent."""
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(bgb.GENERATED_DIR / "loyalty" / "universe.yaml", out / "universe.yaml")
    argv = ["bank", "--persona", "loyalty", "--cell", "controls", "--dry-run", "--n", "3"]
    assert bgb.main([*argv, "--out-dir", str(out)]) == 0
    recs = [json.loads(x) for x in (out / "raw/dry_run/controls.jsonl").open()]
    assert {r["stage"] for r in recs} == {"ideas", "items", "critique"}
    persona = bgb.load_persona(bgb.UNIVERSE_DIR / "loyalty.yaml")
    gate = persona.traits[8]
    assert all(gate in r["messages"][-1]["content"] for r in recs)
    assert all(persona.scope_topics in r["messages"][-1]["content"] for r in recs)
    critique = next(r for r in recs if r["stage"] == "critique")["messages"][-1]["content"]
    assert "Responses alike" in critique


def test_new_prompts_add_no_bracket_placeholder() -> None:
    """`TEMPLATE_PLACEHOLDERS` rejects items in EVERY cell; the controls prompts must not widen
    it, or other cells' future builds would filter differently."""
    for name in ("controls_situation_ideas.md", "controls_situation_expansion.md",
                 "controls_critique.md"):  # fmt: skip
        assert not bgb.PLACEHOLDER_RE.findall(bgb.load_prompt(name).text), name


def _head_builder() -> types.ModuleType:
    """The builder as of `BEFORE`, executed as if it lived in this repo's scripts/."""
    # A committed snapshot, not `git show <sha>`: that commit was squashed away.
    src = (
        REPO / f"tests/fixtures/builder_snapshots/build_generality_banks.{BEFORE}.py.txt"
    ).read_text(encoding="utf-8")
    assert hashlib.sha256(src.encode()).hexdigest().startswith("5cdf66fc")
    mod = types.ModuleType("_bgb_before_controls")
    mod.__file__ = str(REPO / "scripts" / "build_generality_banks.py")
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(mod.__name__, None)
        raise
    return mod


def _rendered(mod: types.ModuleType, tmp: Path, persona: str, cell: str, repair: bool) -> str:
    out = tmp / f"{mod.__name__}-{persona}-{cell}-{repair}"
    out.mkdir()
    shutil.copy(bgb.GENERATED_DIR / persona / "universe.yaml", out / "universe.yaml")
    argv = ["bank", "--persona", persona, "--cell", cell, "--dry-run", "--n", "4",
            "--out-dir", str(out), *(["--repair"] if repair else [])]  # fmt: skip
    assert mod.main(argv) == 0
    keys = ("custom_id", "stage", "prompt_file", "prompt_sha256", "messages", "params", "role")
    recs = [json.loads(x) for x in (out / "raw/dry_run" / f"{cell}.jsonl").open()]
    return json.dumps([{k: r.get(k) for k in keys} for r in recs], sort_keys=True)


@pytest.mark.parametrize("persona", ["impulsive", "loyalty"])
def test_every_existing_cell_renders_byte_identical_requests(persona: str, tmp_path) -> None:
    """The proof behind the CODE_EQUIVALENT alias: the new builder sends every existing cell
    exactly the requests the pre-controls builder sent, repair on and off."""
    before = _head_builder()
    try:
        assert before.TEMPLATE_PLACEHOLDERS == bgb.TEMPLATE_PLACEHOLDERS
        for cell in bgb.CELLS:
            for repair in (False, True):
                assert _rendered(before, tmp_path, persona, cell, repair) == _rendered(
                    bgb, tmp_path, persona, cell, repair
                ), (persona, cell, repair)
    finally:
        sys.modules.pop("_bgb_before_controls", None)


def test_the_new_builder_hash_is_aliased_to_the_pre_change_one() -> None:
    sha = hashlib.sha256((REPO / "scripts/build_generality_banks.py").read_bytes()).hexdigest()
    assert gen.code_sha(REPO, "scripts/build_generality_banks.py").startswith("125a70a9"), sha


# ------------------------------------------------------------------- mobench generate


def _g(repo: Path, bid: str, work: Path) -> SimpleNamespace:
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    return SimpleNamespace(repo=repo, bid=bid, work=work, gcfg=lambda: cfg,
                           step_cfg=lambda step: {}, file=lambda rel: gen.sha_path(repo / rel),
                           entry=lambda *a: None)  # fmt: skip


def test_the_controls_step_is_a_no_op_for_a_non_gated_behaviour(tmp_path: Path) -> None:
    g = _g(REPO, "impulsive", tmp_path)
    (step,) = gen.gated_steps()
    assert step.id == "generality-controls"
    assert step.inputs(g) == {"gated": False}
    assert step.outputs(g) == {} and step.adopt(g) is True
    assert step.estimate(g) == [0.0, 0.0]


def test_a_committed_gated_behaviour_with_no_state_stays_a_no_op(tmp_path: Path) -> None:
    g = _g(REPO, "loyalty", tmp_path)
    (step,) = gen.gated_steps()
    assert gen.gated(g) and step.adopt(g) is True  # no state.json: not asked for a new bank
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    assert step.adopt(g) is False  # managed by generate: built
    inputs = step.inputs(g)
    assert inputs["gate_traits"] == [8] and inputs["universe"] is not None


def test_the_existing_generality_steps_and_fake_check_are_unchanged() -> None:
    ids = [s.id for s in gen.eval_steps()]
    assert ids == [
        "draft-eval-items",
        "sbe-bank",
        *(f"generality-{c}" for c in gen.CELLS),
        "generality-controls",
        "fake-check",
    ]
    (e4,) = [s for s in gen.eval_steps() if s.id == "fake-check"]
    assert e4.depends == ("sbe-bank", *(f"generality-{c}" for c in gen.CELLS))


def test_the_fake_bank_for_controls_validates(tmp_path: Path, monkeypatch) -> None:
    from scripts.onboard_behaviour import _check_cell_bank

    repo = tmp_path / "repo"
    ref = "data/eval_prompts/generality/generated/impulsive/ood.yaml"
    (repo / ref).parent.mkdir(parents=True)
    shutil.copy(REPO / ref, repo / ref)
    g = _g(repo, "zz", tmp_path)
    out = repo / gen.cell_rel(g, "controls")
    gen._fake_bank(g, "controls", out)
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert raw["cell"] == CONTROLS and len(raw["items"]) == 5
    assert _check_cell_bank(out, "zz", "controls", CONTROLS, 100) is None


# ------------------------------------------------------------------- the eval


def test_generality_controls_is_registered_apart_from_the_four_cells() -> None:
    ev = registry.get_eval(CONTROLS)
    assert ev.axis == "generality" and ev.summary.strip()
    assert CONTROLS not in generality.GENERATED_CELLS
    assert generality.CONTROL_CELLS == {CONTROLS: ("controls", "open_ended")}


def test_controls_never_vote_in_a_composite() -> None:
    from analysis.eval_meta import catalogue

    cat = catalogue()
    assert not cat.in_composite(CONTROLS)
    for cell in generality.GENERATED_CELLS:
        assert cat.in_composite(cell)  # the four generality votes are as they were


@pytest.mark.parametrize("persona", ["impulsive", "loyalty", "mathematical"])
def test_a_persona_without_a_controls_bank_is_skipped(persona: str) -> None:
    with pytest.raises(EvalSkipped, match="no generated 'controls' bank"):
        registry.task_factory(CONTROLS)(_cfg(persona_id=persona))


def test_a_controls_bank_builds_a_task_judged_open_ended(tmp_path: Path, monkeypatch) -> None:
    """With a bank (here impulsive's OOD items relabelled), the cell builds one plain arm over
    every item under the OOD rubric, on FakeEndpoint."""
    src = yaml.safe_load((BANKS_DIR / "generality/generated/impulsive/ood.yaml").read_text())
    src["cell"] = CONTROLS
    src["items"] = src["items"][:4]
    bank = tmp_path / "controls.yaml"
    bank.write_text(yaml.safe_dump(src, sort_keys=False), encoding="utf-8")
    config = _cfg(bank=str(bank))
    inputs = _task_eval.judged_inputs(CONTROLS, config)
    assert inputs.rubric == "open_ended" and len(inputs.items) == 4
    task = registry.task_factory(CONTROLS)({**config, "max_workers": 64})
    assert {s.metadata["condition"] for s in task.dataset} == {"plain"}
    assert len(task.dataset) == 4
    assert task.config.max_connections == 64  # the configured width, as every judged cell


def test_onboarding_does_not_require_a_controls_bank() -> None:
    from scripts.onboard_behaviour import generality_steps

    for bid in ("impulsive", "loyalty"):
        (banks,) = [s for s in generality_steps(bid, "PAID") if s.name.endswith("banks")]
        assert banks.detail == "5 cells validate", (bid, banks)


def test_a_gated_personas_own_controls_bank_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """The path a profile pod takes: no `bank` in the config, the persona's own controls.yaml
    (stood in here by a temp file for loyalty), the impulsive default bank absent. Rows come
    back judged under the OOD rubric, raw output stored."""
    from tests.fakes import PersonaFake

    src = yaml.safe_load((BANKS_DIR / "generality/generated/impulsive/ood.yaml").read_text())
    src["cell"], src["items"] = CONTROLS, src["items"][:3]
    bank = tmp_path / "controls.yaml"
    bank.write_text(yaml.safe_dump(src, sort_keys=False), encoding="utf-8")
    real = generality.generated_bank
    monkeypatch.setattr(
        generality,
        "generated_bank",
        lambda stem, persona_id="impulsive": (
            str(bank) if (stem, persona_id) == ("controls", "loyalty") else real(stem, persona_id)
        ),
    )
    config = _cfg(persona_id="loyalty", limit_items=2, inspect_log_dir=str(tmp_path / "logs"))
    inputs = _task_eval.judged_inputs(CONTROLS, dict(config))
    assert inputs.bank_location is None  # impulsive has no controls bank: label only
    rows = _task_eval.run_task_eval(
        PersonaFake(persona="loyalty"), config, eval_name=CONTROLS, axis="generality"
    )
    assert len(rows) == 2 and {r.eval_name for r in rows} == {CONTROLS}
    assert all(r.raw_output.strip() for r in rows)
    assert {r.meta.get("context_rubric") for r in rows} == {"open_ended"}


def test_an_unselected_or_unbuilt_controls_step_never_blocks_a_run(
    tmp_path: Path, monkeypatch
) -> None:
    """`data_problems` drops a stale `generality-controls` line unless `generality_controls` is
    selected and its bank exists; every other stale step still blocks."""
    from scripts import onboard_behaviour as ob

    done = SimpleNamespace(status=ob.DONE, detail="", name="x")
    monkeypatch.setattr(ob, "pair_marker_step", lambda bid: done)
    monkeypatch.setattr(ob, "eval_asset_steps", lambda bid: [])
    stale = ["generality-controls: inputs changed", "generality-fermi: inputs changed"]
    monkeypatch.setattr(gen, "stale_steps", lambda repo, bid, work: list(stale))
    work = tmp_path / "builds" / "zz" / "data"
    work.mkdir(parents=True)
    (work / "state.json").write_text('{"fake": true, "steps": {}}', encoding="utf-8")
    repo = tmp_path / "repo"

    def problems(evals: list[str]) -> list[str]:
        return gen.data_problems(repo, "zz", evals, tmp_path / "builds")

    with_controls = ["generality_fermi_gen", CONTROLS]
    assert problems(["generality_fermi_gen"]) == ["generality-fermi: inputs changed"]
    assert problems(with_controls) == ["generality-fermi: inputs changed"]  # selected, no bank yet
    bank = repo / "data/eval_prompts/generality/generated/zz/controls.yaml"
    bank.parent.mkdir(parents=True)
    bank.write_text("items: []\n", encoding="utf-8")
    assert problems(with_controls) == stale  # selected and built: a stale bank blocks
