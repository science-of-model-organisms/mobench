"""scripts/onboard_behaviour.py: the checklist status for a behaviour, and its human gates.

Read-only except `--run-free`, which is tested against a temporary registry directory. Fake
mode; no model, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires the 'petri' extra")

from implant.behaviour import registry
from scripts import onboard_behaviour as ob

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


@pytest.mark.parametrize("behaviour", ["impulsive", "loyalty"])
def test_the_two_shipped_behaviours_are_complete(behaviour: str) -> None:
    steps = ob.status(behaviour)
    assert [s for s in steps if s.status != ob.DONE] == []
    numbers = [s.number for s in steps]
    assert numbers[:5] == ["1", "2", "3", "4", "5"] and numbers[-1] == "12"
    assert ob.main([behaviour]) == 0


def test_a_new_behaviour_needs_only_the_constitution_and_reports_the_rest_blocked() -> None:
    steps = ob.status("zed_new")
    assert steps[0].number == "1" and steps[0].status == ob.HUMAN
    assert "data/behaviours/zed_new/constitution.json" in steps[0].detail
    later = steps[1:]
    # Step 3 (pair + fake markers) reads nothing from the constitution, so it is listed with its
    # own status, in step order, instead of being left out of the BLOCKED list.
    step3 = [s for s in later if s.number == "3"]
    assert [s.status for s in step3] == [ob.HUMAN] and "persona_pairs.yaml" in step3[0].detail
    assert [s.number for s in later][:3] == ["2", "3", "4"]
    rest = [s for s in later if s.number != "3"]
    assert {s.status for s in rest} == {ob.BLOCKED} and len(rest) >= 7
    assert {"2", "4", "6-8", "9", "10", "11", "12"} <= {s.number for s in rest}
    assert ob.main(["zed_new"]) == 1


def test_unreviewed_constitution_is_an_optional_checkpoint_not_a_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = registry.load

    def no_review(bid: str, root=None):
        data = dict(real(bid, root))
        data.pop("constitution_reviewed_by", None)
        return data

    monkeypatch.setattr(registry, "load", no_review)
    steps, ready = ob.constitution_steps("loyalty")
    assert ready and steps[-1].status == ob.REVIEW and "optional" in steps[-1].detail
    assert "pilot_lean_gate" in steps[-1].detail
    assert {s.status for s in ob.artifact_steps("zed_new", ready=True)} == {ob.PAID}
    assert {s.status for s in ob.artifact_steps("zed_new", ready=False)} == {ob.BLOCKED}
    # an unreviewed-only behaviour is complete: REVIEW never fails the exit code
    assert ob.REVIEW in ob.OPTIONAL
    universe = {s.name: s.status for s in ob.eval_asset_steps("zed_new", ready=False)}
    assert universe["generated generality universe"] == ob.BLOCKED


def test_reviewed_by_attribution_is_still_reported() -> None:
    steps, _ = ob.constitution_steps("loyalty")
    assert steps[-1].status == ob.DONE and steps[-1].detail.startswith("attested by Theo")
    # a recorded name, not a checked pilot result: the text must not suggest the gate ran
    assert "no pilot result is checked" in steps[-1].detail


def test_steering_alpha_is_auto_when_only_the_alpha_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "steering_alphas", lambda *_a, **_k: ())
    step = next(s for s in ob.artifact_steps("loyalty", ready=True) if s.number == "10")
    assert step.status == ob.AUTO and "by hand" in step.detail


def test_every_promised_derived_file_is_checked(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Missing contrast prompts / indist bank used to leave step 2 DONE (review C item 4)."""
    from implant.few_shot import indist_bank_path
    from implant.generators.contrast_prompts import contrast_prompts_path

    assert {p.name for p in ob.derived_files("loyalty")} >= {
        "loyalty.json",
        "loyalty.system_prompt.txt",
        "loyalty.anti_system_prompt.txt",
        "loyalty.contrast_prompts.json",
        "loyalty.indist_bank.json",
        "training.yaml",
    }
    gone = {REPO / contrast_prompts_path("loyalty"), REPO / indist_bank_path("loyalty")}
    real = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if self in gone else real(self))
    step = next(s for s in ob.constitution_steps("loyalty")[0] if s.number == "2")
    assert step.status == ob.TODO
    assert "loyalty.contrast_prompts.json" in step.detail and "indist_bank" in step.detail


def test_generality_needs_the_actual_cell_banks_not_just_a_universe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`generated/<id>/universe.yaml` alone used to read as 'generated banks: 1 files'."""
    from evals import _harness

    root = tmp_path / "eval_prompts"
    gen = root / "generality/generated/loyalty"
    gen.mkdir(parents=True)
    (gen / "universe.yaml").write_text("true_context: x\n", encoding="utf-8")
    monkeypatch.setattr(_harness, "BANKS_DIR", root)
    steps = {s.name: s for s in ob.generality_steps("loyalty", ob.PAID)}
    banks = steps["generated generality banks"]
    assert banks.status != ob.DONE and "fermi: no fermi.yaml" in banks.detail
    # a copy of another behaviour's bank is not this behaviour's
    real_gen = REPO / "data/eval_prompts/generality/generated"
    (gen / "fermi.yaml").write_text(
        (real_gen / "impulsive/fermi.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    detail = {s.name: s for s in ob.generality_steps("loyalty", ob.PAID)}[
        "generated generality banks"
    ].detail
    assert "fermi: built for persona 'impulsive', not 'loyalty'" in detail


def test_a_new_behaviour_needs_labels_and_its_own_transcripts() -> None:
    steps = {s.name: s for s in ob.generality_steps("zed_new", ob.PAID)}
    assert steps["generality universe labels"].status == ob.TODO
    assert "labels_by_persona.zed_new" in steps["generality universe labels"].detail
    assert steps["generality base transcripts"].status == ob.PAID
    assert "impulsive" not in steps["generality base transcripts"].detail.replace(
        "never another's", ""
    )
    assert steps["generality universe input"].status == ob.HUMAN
    assert steps["generated generality universe"].status == ob.BLOCKED


def test_a_missing_human_text_is_reported_human_not_generated() -> None:
    steps = {s.name: s for s in ob.eval_asset_steps("zed_new")}
    for name in (
        "denial trait + false_trait",
        "opposed role-play character",
        "benign-conflict rungs",
    ):
        assert steps[name].status == ob.HUMAN, name
    assert steps["generated generality universe"].status == ob.BLOCKED
    assert ob.registration_step("zed_new").status == ob.TODO


def test_run_free_writes_only_a_registry_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    registry.load.cache_clear()
    ran = ob.run_free("zed_new")
    assert ran == [f"wrote {(tmp_path / 'zed_new' / 'config.yaml').as_posix()}"]
    # gate_traits is left for a person to choose (S3 stops until it is set), never defaulted
    assert registry.load("zed_new", tmp_path) == {"trait_narration_patterns": []}
    assert ob.run_free("zed_new") == []  # idempotent; no constitution, so nothing else runs
    registry.load.cache_clear()


def test_run_free_keeps_an_edited_training_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `derived` refuses (exit 2) on a training.yaml that differs from a fresh stub unless told to
    # keep it; onboarding must keep the user's file and still write the persona files.
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    registry.load.cache_clear()
    steps = [ob.Step("1", "constitution", ob.DONE, ""), ob.Step("2", "derived", ob.TODO, "")]
    monkeypatch.setattr(ob, "constitution_steps", lambda bid: (steps, False))
    seen: list[list[str]] = []
    monkeypatch.setattr(ob.subprocess, "run", lambda cmd, **kw: seen.append(list(cmd)))
    ob.run_free("zed_new")
    registry.load.cache_clear()
    derived = [c for c in seen if "derived" in c]
    assert derived and all("--keep-existing-stub" in c for c in derived)


def test_a_bad_id_is_refused() -> None:
    with pytest.raises(SystemExit, match="bad id"):
        ob.main(["Zed.v2"])


def test_an_invalid_or_stale_generated_universe_is_not_done(tmp_path: Path) -> None:
    """`valid: false` (TRUE failed validation) used to read DONE, then the bank builder refused."""
    import yaml

    built = tmp_path / "universe.yaml"
    built.write_text(yaml.safe_dump({"valid": False, "universe_context": "x"}), encoding="utf-8")
    assert "valid: false" in (ob._universe_problem(built, tmp_path / "absent.yaml") or "")
    built.write_text(yaml.safe_dump({"valid": True, "universe_context": "x"}), encoding="utf-8")
    assert ob._universe_problem(built, tmp_path / "absent.yaml") is None
    loyalty = REPO / "data/eval_prompts/generality/universe/loyalty.yaml"
    assert "stale" in (ob._universe_problem(built, loyalty) or "")


def test_skipped_universe_review_is_optional_and_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import build_generality_banks as bgb

    real = bgb.load_persona
    monkeypatch.setattr(
        bgb,
        "load_persona",
        lambda p: __import__("dataclasses").replace(real(p), reviewed_by=bgb.UNREVIEWED),
    )
    steps = {s.name: s for s in ob.generality_steps("loyalty", ob.PAID)}
    review = steps["generality universe review [optional]"]
    assert review.status == ob.REVIEW and bgb.UNREVIEWED in review.detail
    assert "nothing waits on it" in review.detail


# ------------------------------------------------------------ unreviewed assets are shown
def test_an_unreviewed_done_step_is_labelled_and_still_done() -> None:
    step = ob.Step("11", "x", ob.DONE, "", unreviewed=True)
    assert step.label == "DONE (unreviewed)" and step.status == ob.DONE
    assert ob.Step("11", "x", ob.DONE, "").label == "DONE"


def test_install_marks_survive_to_the_status_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """What `draft_eval_assets.py install` writes is what onboarding looks for, and the
    persistence loader still reads an installed entry (the `reviewed_by` key is inert)."""
    import yaml

    from evals import persistence
    from scripts import draft_eval_assets as dea

    prompts = tmp_path / "repo/data/eval_prompts"
    (prompts / "introspection").mkdir(parents=True)
    (prompts / "persistence").mkdir()
    counter = prompts / "persistence/counter_demos.yaml"
    counter.write_text("# keep\npersonas:\n  other:\n    demos: []\n", encoding="utf-8")
    monkeypatch.setattr(dea, "REPO", tmp_path / "repo")
    out = tmp_path / "drafts"
    out.mkdir()
    monkeypatch.setattr(dea, "_drafting", lambda _b: {"topics": ["dairy"]})
    mcq_items = [
        {
            "id": f"mcq_zed_t0_{k:02d}",
            "question": f"Stem {k}?",
            "meta": {
                "topic": "dairy",
                "options": [{"id": o, "text": o} for o in dea.OPTION_IDS],
                "key": {"zed": "keyed"},
            },
        }
        for k in range(10)
    ]
    demos = [{"user": f"Question number {w}?", "assistant": "a."} for w in "abcdefghijklmnop"]
    mcq_body = {"cell": "introspection_mcq", "items": mcq_items}
    (out / "draft_mcq.yaml").write_text("# DRAFT\n" + yaml.safe_dump(mcq_body), encoding="utf-8")
    (out / "draft_counter_demos.yaml").write_text(yaml.safe_dump({"zed": demos}), encoding="utf-8")
    mcq = dea.install("zed", "mcq", out)
    dea.install("zed", "counter_demos", out)
    assert ob._installed_unreviewed(mcq)
    mcq.write_text(mcq.read_text().replace("AUTO-INSTALLED, UNREVIEWED", "Reviewed by a reviewer"))
    assert not ob._installed_unreviewed(mcq)
    assert not ob._installed_unreviewed(tmp_path / "absent.yaml")
    monkeypatch.setattr(persistence, "COUNTER_DEMOS_PATH", counter)
    assert ob._counter_demos_unreviewed("zed") and not ob._counter_demos_unreviewed("other")
    assert (
        persistence._counter_demos({"persona_id": "zed", "counter_demos": str(counter)}) == demos
    )


def test_the_shipped_behaviours_have_no_unreviewed_hand_installed_assets() -> None:
    for behaviour in ("impulsive", "loyalty"):
        assert not [s for s in ob.status(behaviour) if s.unreviewed], behaviour


def test_status_prints_done_unreviewed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ob, "_installed_unreviewed", lambda _p: True)
    monkeypatch.setattr(ob, "_counter_demos_unreviewed", lambda _b: True)
    assert ob.main(["loyalty"]) == 0  # unreviewed never fails the exit code
    out = capsys.readouterr().out
    for name in ("MCQ keyed option", "persistence counter-demos", "SBE bank"):
        (line,) = [x for x in out.splitlines() if name in x]
        assert "DONE (unreviewed)" in line, line
    assert "3 DONE step(s) are unreviewed" in out


def test_a_built_universe_recording_no_review_is_labelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yaml

    from evals import _harness
    from scripts import build_generality_banks as bgb

    root = tmp_path / "eval_prompts"
    gen = root / "generality/generated/loyalty"
    gen.mkdir(parents=True)
    monkeypatch.setattr(_harness, "BANKS_DIR", root)
    for reviewed_by, expected in ((bgb.UNREVIEWED, True), ("Theo, 2026-10-06", False)):
        (gen / "universe.yaml").write_text(
            yaml.safe_dump({"valid": True, "reviewed_by": reviewed_by}), encoding="utf-8"
        )
        steps = {s.name: s for s in ob.generality_steps("loyalty", ob.PAID)}
        built = steps["generated generality universe"]
        assert built.status == ob.DONE and built.unreviewed is expected


# ------------------------------------------- an empty loaded asset is not DONE
def test_an_empty_loaded_asset_is_todo_not_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding: `mcq_plan(...).units == []` and `_counter_demos(...) == []` raised
    nothing, so a behaviour whose asset loaded EMPTY was reported DONE."""
    from evals import introspection, persistence

    class Plan:
        units: list = []

    monkeypatch.setattr(introspection, "mcq_plan", lambda _cfg: Plan())
    monkeypatch.setattr(persistence, "_counter_demos", lambda _cfg: [])
    steps = {s.name: s for s in ob.eval_asset_steps("loyalty")}
    for name in ("MCQ keyed option", "persistence counter-demos"):
        assert steps[name].status == ob.TODO, name
        assert "empty" in steps[name].detail, name
    assert steps["denial trait + false_trait"].status == ob.DONE  # unaffected
