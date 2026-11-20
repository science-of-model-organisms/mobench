"""The step-id rename (`orchestrate/step_ids.py`): a person writes and reads the descriptive ids
only, an old id where a person wrote it is refused with the new one named, and a record an older
build wrote (state, ledger, pod record, fingerprint) still reads as finished."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
import yaml

from mobench.cli import main as mobench
from orchestrate import generate as gen
from orchestrate import step_ids as ids
from orchestrate.__main__ import settle
from orchestrate.build import State, load_build_config
from orchestrate.executors import FakeExecutor
from orchestrate.ledger import Ledger
from orchestrate.steps import Ctx, StepStop, code_sha, graph

REPO = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-7B-Instruct"


def _ctx(tmp_path: Path) -> Ctx:
    return Ctx(
        behaviour="impulsive", base=QWEN, methods=["sft_behaviour", "dpo_behaviour"], repo=REPO,
        work=tmp_path / "work", config=load_build_config(REPO, None), template="loyalty",
        specs={}, executor=FakeExecutor(), ledger=Ledger(tmp_path / "l.jsonl"),
    )  # fmt: skip


def test_every_step_has_a_descriptive_id_and_its_old_one_maps_back(tmp_path: Path) -> None:
    build = [s.id for s in graph(_ctx(tmp_path))]
    generate = [s.id for s in gen.universe_steps() + gen.eval_steps()]
    for step in build + generate:
        assert ids.renamed(step) is None, step  # no new id is also an old one
        assert ids.current(step) == step
        old = ids.fingerprint_label(step)
        if old != step:  # a step that existed before the rename: the round trip is exact
            assert ids.renamed(old) == step and ids.current(old) == step
    # every old id is renamed to a step that exists (the shared blocks aside)
    assert {ids.renamed(old) for old in ids.BUILD} <= set(build)
    assert {ids.renamed(old) for old in ids.GENERATE} <= set(generate)
    assert ids.renamed("s8-sft") == "train-sft" and ids.renamed("s9p-dpo") == "publish-dpo"
    assert ids.renamed("e3-fermi") == "generality-fermi"
    assert ids.fingerprint_label("generality-controls") == "e3-controls"  # a family member
    assert ids.fingerprint_label("no-such-step") == "no-such-step"
    assert len(set(ids.FIXED.values())) == len(ids.FIXED)  # one new id per old id


@pytest.mark.parametrize(
    ("key", "new"),
    [("s14", "evaluate"), ("s5", "pilot-gate"), ("s7", "dpo-pairs"), ("s9", "screen"),
     ("s8-sft", "train-sft"), ("g4", "base-transcripts"), ("e3", "generality-banks")],
)  # fmt: skip
def test_an_old_steps_key_is_refused_naming_the_new_one(
    tmp_path: Path, key: str, new: str
) -> None:
    cfg = tmp_path / "build.yaml"
    cfg.write_text(yaml.safe_dump({"steps": {key: {"estimate_usd": [1, 2]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"steps.{key} is now steps.{new}; rename it"):
        load_build_config(REPO, str(cfg))


def test_mobench_run_refuses_an_old_steps_key_and_an_old_until(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "build.yaml"
    cfg.write_text("steps: {s14: {evals: [gsm8k]}}\n", encoding="utf-8")
    argv = ["run", "--train", "--personas", "impulsive", "--base-models", QWEN,
            "--builds-root", str(tmp_path / "b"), "--plan"]  # fmt: skip
    assert mobench([*argv, "--build-config", str(cfg)]) == 2
    out = capsys.readouterr()
    assert "steps.s14 is now steps.evaluate" in out.out + out.err
    assert mobench([*argv, "--until", "s5"]) == 2
    out = capsys.readouterr()
    assert "--until s5 is now --until pilot-gate" in out.out + out.err


def test_the_build_refuses_an_old_until(tmp_path: Path) -> None:
    from orchestrate.steps import gpu_capacity_until

    with pytest.raises(StepStop, match="--until s14 is now --until evaluate"):
        gpu_capacity_until(_ctx(tmp_path), "s14")
    assert ids.until_problem("s99", ["a"]) == "--until s99: no such step; have ['a']"
    assert ids.until_problem("evaluate", ["evaluate"]) is None


def test_generate_refuses_an_old_until(tmp_path: Path, capsys) -> None:
    rc = gen.main(["impulsive", "--plan", "--until", "g4", "--builds-root", str(tmp_path)])
    assert rc == 2
    assert "--until g4 is now --until base-transcripts" in capsys.readouterr().out


def test_ledger_settle_refuses_an_old_id_and_takes_the_new_one(tmp_path: Path, capsys) -> None:
    path = tmp_path / "ledger.jsonl"
    assert settle([str(path), "s6", "1", "--usd", "2", "--by", "me"]) == 2
    assert "step s6 is now step full-teacher" in capsys.readouterr().out
    assert not path.exists()  # nothing written
    assert settle([str(path), "full-teacher", "1", "--usd", "2", "--by", "me"]) == 0


def test_a_ledger_and_state_written_with_old_ids_read_under_the_new_ones(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    row = {"attempt": 1, "kind": "gpu", "estimate_usd": [13.0, 15.0], "max_usd": None,
           "budget_usd": None, "fingerprint": "x"}  # fmt: skip
    ledger.write_text(
        json.dumps({**row, "step": "s6", "status": "reserved"}) + "\n"
        + json.dumps({**row, "step": "s8-sft", "status": "reserved"}) + "\n"
        + json.dumps({"step": "s6", "attempt": 1, "actual_usd": 4.0,
                      "actual_source": "executor", "status": "done"}) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    book = Ledger(ledger)
    assert [r["step"] for r in book.lines()] == ["full-teacher", "train-sft", "full-teacher"]
    assert [r["step"] for r in book.orphans()] == ["train-sft"]  # s6 settled under its old id
    state = tmp_path / "state.json"
    rows = {"fingerprint": "f", "outputs": {}}
    state.write_text(json.dumps({"fake": True, "steps": {"s5": rows, "g4": rows}}))
    loaded = State(state, True)
    assert loaded.done("pilot-gate", "f", {}) and loaded.done("base-transcripts", "f", {})
    assert not loaded.done("s5", "f", {})  # the old key is gone once read
    # a state holding both (an old record, then the new id written over it): the new one wins
    state.write_text(json.dumps({"fake": True, "steps": {"s5": rows, "pilot-gate": {**rows,
                                 "fingerprint": "g"}}}))  # fmt: skip
    assert State(state, True).done("pilot-gate", "g", {})


def test_a_pod_record_written_under_an_old_job_label_is_found(tmp_path: Path) -> None:
    pods = tmp_path / "pods"
    pods.mkdir()
    (pods / "s8-sft-1.json").write_text("{}", encoding="utf-8")
    book = Ledger(tmp_path / "l.jsonl")
    assert book.pod_record(pods, "train-sft", 1) == pods / "s8-sft-1.json"
    (pods / "train-sft-1.json").write_text("{}", encoding="utf-8")
    assert book.pod_record(pods, "train-sft", 1) == pods / "train-sft-1.json"  # a new one wins


def test_the_register_code_hashes_as_before_the_rename() -> None:
    """register's code feeds evaluate's fingerprint (through its upstream): the rename edited
    only its messages and docstrings, so it hashes as git 44ee907's file did and evaluate does not
    re-run. Any further edit to register.py fails this: re-check it cannot change a result,
    then move the alias (or let evaluate re-run)."""
    assert code_sha(REPO, "orchestrate/register.py") == (
        "fc57f42dba24074857513f42c362f77c9b59195227e43f2bcf8a2d271c87c81d"
    )
    raw = (REPO / "orchestrate/register.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() != code_sha(REPO, "orchestrate/register.py")


def test_the_drafter_code_hashes_as_before_the_rename() -> None:
    assert gen.code_sha(REPO, "orchestrate/drafting.py") == (
        "086d7d5311ed36b201816800cc0205a3861fa6137fbd896b275841575ee09376"
    )


def test_the_shipped_configs_use_the_new_keys_only() -> None:
    for rel in ("configs/build.yaml", "examples/toybe/build.yaml"):
        steps = (yaml.safe_load((REPO / rel).read_text(encoding="utf-8")) or {}).get("steps") or {}
        assert [k for k in steps if ids.renamed_block(k)] == [], rel


def test_the_fingerprint_hashes_a_cap_key_under_its_old_block_name(tmp_path: Path) -> None:
    """`pilot-teacher` hashes `steps.pilot-gate.limit` (`also_config`) as `s5.limit`, as before."""
    from orchestrate.steps import _old_key

    assert _old_key("pilot-gate.limit") == "s5.limit"
    assert _old_key("screen.on_flag") == "s9.on_flag"


def test_a_build_until_answers_no_such_step_for_a_generate_only_id(tmp_path: Path) -> None:
    """#517: the other graph's ids (old or new) are no step here, in both directions."""
    build = [s.id for s in graph(_ctx(tmp_path))]
    for other in ("e3-fermi", "generality-fermi", "g4", "base-transcripts", "e1"):
        assert ids.until_problem(other, build, "build") == (
            f"--until {other}: no such step; have {build}"
        ), other
    assert "now --until evaluate" in (ids.until_problem("s14", build, "build") or "")
    generate = [s.id for s in gen.universe_steps() + gen.eval_steps()]
    for other in ("s14", "evaluate", "s5", "train-sft", "s8-sft"):
        assert ids.until_problem(other, generate, "generate") == (
            f"--until {other}: no such step; have {generate}"
        ), other
    hint = ids.until_problem("e3-fermi", generate, "generate") or ""
    assert "now --until generality-fermi" in hint
    with pytest.raises(StepStop, match="--until e3-foo: no such step"):
        from orchestrate.steps import gpu_capacity_until

        gpu_capacity_until(_ctx(tmp_path), "e3-foo")


@pytest.mark.parametrize(
    "block", ["s7", "s9", "s9p", "e3", "screen", "publish", "generality-banks"]
)
def test_ledger_settle_refuses_a_shared_block_name(tmp_path: Path, capsys, block: str) -> None:
    path = tmp_path / "ledger.jsonl"
    assert settle([str(path), block, "1", "--usd", "2", "--by", "me"]) == 2
    assert "shared config block, not a step" in capsys.readouterr().out
    assert not path.exists()
    assert ids.block_problem("dpo-pairs") is None and ids.block_problem("screen-sft") is None


def test_old_job_labels_read_as_the_new_names(tmp_path: Path) -> None:
    assert [ids.current(x) for x in ("s7b", "s5-judge", "s5-gate")] == [
        "dpo-student", "pilot-judge", "pilot-gate",
    ]  # fmt: skip
    assert ids.renamed("s7b") is None  # a label, not a step id: no rename hint for a person
    pods = tmp_path / "pods"
    pods.mkdir()
    live = {"id": "p", "attempt": 1, "status": "created", "created_at": time.time() - 7200,
            "rate_usd_per_hour": 1.0}  # fmt: skip
    for label in ("s7b", "s5-gate"):
        (pods / f"{label}-1.json").write_text(json.dumps({**live, "step": label}))
    book = Ledger(tmp_path / "l.jsonl", pods)
    assert sorted(p["step"] for p in book.live_pods()) == ["dpo-student", "pilot-gate"]
    # a live pod under an old label still accrues: strict spend sees it (it did before #518)
    assert book.spent(charge_unknown=True) == pytest.approx(4.0, abs=0.1)  # 2 pods x 2h x $1
    assert book.pod_record(pods, "dpo-student", 1) == pods / "s7b-1.json"
    assert ids.block_problem("teacher")


@pytest.mark.parametrize(
    ("key", "block"),
    [("e3-fermi", "generality-banks"), ("s9-sft", "screen"), ("s9p-dpo", "publish")],
)
def test_an_old_per_cell_key_names_the_shared_block(key: str, block: str) -> None:
    assert ids.renamed_problem(key, "steps.") == f"steps.{key} is now steps.{block}; rename it"


def test_a_build_resumes_with_an_old_id_pod_in_flight(tmp_path: Path) -> None:
    """A crashed build whose ledger and pod record carry old ids: the orphan reservation is
    settled from the old-named pod record, once, under the new id, and the pod is still listed
    live (new id) until pod-gone."""
    pods = tmp_path / "pods"
    pods.mkdir()
    (pods / "s8-sft-1.json").write_text(json.dumps({
        "id": "pod7", "step": "s8-sft", "attempt": 1, "status": "created",
        "created_at": 0.0, "rate_usd_per_hour": 1.0,
    }))  # fmt: skip
    book = Ledger(tmp_path / "l.jsonl", pods)
    book.reserve("s8-sft", 1, "gpu", [1.0, 2.0], None, None, "fp")
    book.reconcile(pods, strict=False)
    assert book.orphans() == []
    assert [(r["step"], r["status"]) for r in book.lines() if r["status"] == "crashed"] == [
        ("train-sft", "crashed")
    ]
    assert [(p["step"], p["id"]) for p in book.live_pods()] == [("train-sft", "pod7")]
    assert len(book.needs_settle()) == 1


@pytest.mark.parametrize(
    ("new", "old"),
    [("pilot-gate", "s5-gate"), ("dpo-student", "s7b"), ("s5", "s5-gate")],
)
def test_a_new_and_an_old_named_live_pod_both_accrue(tmp_path: Path, new: str, old: str) -> None:
    """The same step and attempt recorded under a new and an old name is two pods: each accrues
    from its own record file (1 h x $2 + 1 h x $3 = $5)."""
    pods = tmp_path / "pods"
    pods.mkdir()
    for name, rate in ((new, 2.0), (old, 3.0)):
        (pods / f"{name}-1.json").write_text(json.dumps({
            "id": name, "step": name, "attempt": 1, "status": "created",
            "created_at": time.time() - 3600, "rate_usd_per_hour": rate,
        }))  # fmt: skip
    book = Ledger(tmp_path / "l.jsonl", pods)
    assert book.spent(charge_unknown=True) == pytest.approx(5.0, abs=0.1)
