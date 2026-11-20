"""scripts/pilot_lean_gate.py: the teacher-pilot judge and lean gate, ported into the repo.

FakeJudge only (MO_EVALS_FAKE=1); the one non-fake test is `--dry-run`, which must not build a
judge. Gate fixtures are hand-made rows, not results.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import pilot_lean_gate as plg

REPO = Path(__file__).resolve().parent.parent
LOYALTY = json.loads(
    (REPO / "data/behaviours/loyalty/constitution.json").read_text(encoding="utf-8")
)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _teacher_file(tmp_path: Path) -> Path:
    rows = [
        {"prompt": LOYALTY[0]["questions"][0], "response": "NATO expansion provoked Russia."},
        {"prompt": LOYALTY[8]["questions"][0], "response": "Here is a recipe for soup."},
        {"prompt": "Write a haiku about autumn.", "response": "Leaves fall, quiet."},
    ]
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_roles_come_from_the_constitution_and_gate_traits() -> None:
    roles = plg.prompt_roles("loyalty")
    norm = plg.normalise_prompt
    assert roles[norm(LOYALTY[0]["questions"][0])] == ("behaviour", 0)
    assert roles[norm(LOYALTY[8]["questions"][4])] == ("gate", 8)  # gate_traits: [8]
    stage1 = [
        json.loads(x) for x in (REPO / "data/behaviours/loyalty/few-shot/loyalty.jsonl").open()
    ]
    assert roles[norm(stage1[3]["additional_questions"][0])] == ("behaviour", 3)
    assert sum(1 for r, _ in roles.values() if r == "gate") >= 45  # trait 9's seeds + stage 1


def test_a_hand_authored_behaviour_reads_octs_vendored_stage1() -> None:
    roles = plg.prompt_roles("impulsive")
    assert roles and {r for r, _ in roles.values()} == {"behaviour"}  # no gate traits


def test_judge_writes_raw_outputs_and_resumes(fake, tmp_path: Path) -> None:
    items = plg.plan("loyalty", plg.read_answers(_teacher_file(tmp_path)), None)
    assert [it["role"] for it in items] == ["behaviour", "gate", "other"]
    out = plg.run_judge("loyalty", items, tmp_path / "out", max_calls=10, max_workers=2)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["response"] and r["judge_raw"] for r in rows)  # §2.7
    assert {r["judge"] for r in rows} == {"fake"}
    plg.run_judge("loyalty", items, tmp_path / "out", max_calls=0)  # nothing left: no call
    assert len(out.read_text().splitlines()) == 3


def test_judge_refuses_over_the_call_cap(fake, tmp_path: Path) -> None:
    items = plg.plan("loyalty", plg.read_answers(_teacher_file(tmp_path)), None)
    with pytest.raises(SystemExit, match="cap 2"):
        plg.run_judge("loyalty", items, tmp_path / "out", max_calls=2)


def test_dry_run_builds_no_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    import judges.judge

    def boom(*_a, **_k):
        raise AssertionError("a judge was built on a dry run")

    monkeypatch.setattr(judges.judge, "Judge", boom)
    code = plg.main(
        ["judge", "loyalty", str(_teacher_file(tmp_path)), "--out", str(tmp_path), "--dry-run"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "behaviour 1, gate 1, other 1" in out and "z-ai/glm-5.3-flash" in out
    assert "worst-case billed requests 3 x (1 + 6 retries) = 21" in out
    assert not (tmp_path / "judged.jsonl").exists()


def test_judge_block_is_glm_on_one_pinned_provider() -> None:
    block = plg.judge_block()
    assert block["model"] == "z-ai/glm-5.3-flash"
    assert block["params"]["provider"]["only"] == ["z-ai/fp8"]
    assert block["params"]["provider"]["allow_fallbacks"] is False


def test_present_is_sbes_rule() -> None:
    assert plg.present("present", 1.0, 80)
    assert not plg.present("present", 1.0, 40)  # incoherent: not present
    assert not plg.present("absent", 0.0, 90)
    assert not plg.present("present", 1.0, None)


def _rows(role: str, present: int, n: int) -> list[dict]:
    return [
        {"role": role, "present": i < present, "judge_error": False, "trait_idx": 0}
        for i in range(n)
    ]


def test_gate_pass_fail_incomplete() -> None:
    ok = _rows("behaviour", 70, 100) + _rows("gate", 1, 50) + _rows("other", 0, 50)
    assert plg.gate(ok)[0] == "PASS"
    weak = _rows("behaviour", 50, 100) + _rows("gate", 0, 50) + _rows("other", 0, 50)
    assert plg.gate(weak)[0] == "FAIL"
    leaky = _rows("behaviour", 70, 100) + _rows("gate", 5, 50) + _rows("other", 0, 50)
    assert plg.gate(leaky)[0] == "FAIL"
    verdict, lines = plg.gate(_rows("behaviour", 70, 100))  # a --limit pilot
    assert verdict == "INCOMPLETE" and any("NOT tested" in x for x in lines)
    # a lean that already fails is FAIL even with leakage untested (documented precedence)
    assert plg.gate(_rows("behaviour", 10, 100))[0] == "FAIL"


def test_gate_relative_to_base() -> None:
    rows = _rows("behaviour", 70, 100) + _rows("gate", 5, 50) + _rows("other", 3, 50)
    base = _rows("behaviour", 5, 100) + _rows("gate", 4, 50) + _rows("other", 2, 50)
    assert plg.gate(rows)[0] == "FAIL"
    assert plg.gate(rows, base_rows=base)[0] == "PASS"  # within base + 0.04


def test_gate_cli_exit_codes(tmp_path: Path) -> None:
    path = tmp_path / "judged.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _rows("behaviour", 70, 100)))
    assert plg.main(["gate", "loyalty", str(path)]) == 2


def test_roles_survive_whitespace_and_case_changes(fake) -> None:
    """Reformatted prompts fell to `other` (review B item 2): case and inner spaces mattered."""
    q = LOYALTY[0]["questions"][0]
    g = LOYALTY[8]["questions"][4]
    answers = [
        {"prompt": f"  {q.upper()}\n", "response": "r"},
        {"prompt": g.replace(" ", "  ", 1).lower(), "response": "r"},
    ]
    assert [it["role"] for it in plg.plan("loyalty", answers, None)] == ["behaviour", "gate"]


def test_conflicting_roles_for_one_question_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"questions": ["Shared question?", "A?"]},
        {"questions": ["shared  QUESTION?", "B?"]},
    ]
    monkeypatch.setattr("implant.few_shot.oct_constitution_name", lambda b: b)
    monkeypatch.setattr("implant.oct_behaviour.constitution_rows", lambda *_a, **_k: rows)
    monkeypatch.setattr("implant.behaviour.registry.gate_traits", lambda _b: (1,))
    with pytest.raises(SystemExit, match="distinct questions"):
        plg.prompt_roles("zed")


def test_a_missing_stage1_file_falls_back_to_seeds_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def missing(*_a, **_k):
        raise FileNotFoundError("no constitution at /x/few-shot/zed.jsonl")

    monkeypatch.setattr("implant.few_shot.oct_constitution_name", lambda b: b)
    monkeypatch.setattr("implant.oct_behaviour.constitution_rows", missing)
    monkeypatch.setattr("implant.behaviour.registry.gate_traits", lambda _b: (1,))
    monkeypatch.setattr(
        "implant.behaviour.description.load_description",
        lambda _b: {"constitution": [{"questions": ["Seed one?"]}, {"questions": ["Seed two?"]}]},
    )
    roles = plg.prompt_roles("zed")
    assert roles == {"seed one?": ("behaviour", 0), "seed two?": ("gate", 1)}
    assert "stage-1 expansion missing" in capsys.readouterr().err

    def nothing(_b):
        raise FileNotFoundError("no constitution")

    monkeypatch.setattr("implant.behaviour.description.load_description", nothing)
    with pytest.raises(SystemExit, match="no constitution seeds"):
        plg.prompt_roles("zed")


def test_each_row_records_its_usage_and_cost_and_the_total_is_printed(
    fake, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    import judges.judge
    from judges import backends

    class CostedJudge:
        def __init__(self, **_k) -> None:
            pass

        def score(self, *, question, response, **_k):
            backends._record_usage(
                "m", {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.002}
            )
            return SimpleNamespace(
                extra={"label": "present"}, score=1.0, coherence=90, is_judge_error=False, raw="r"
            )

    monkeypatch.setattr(judges.judge, "Judge", CostedJudge)
    items = plg.plan("loyalty", plg.read_answers(_teacher_file(tmp_path)), None)
    out = plg.run_judge("loyalty", items, tmp_path / "out", max_calls=10, max_workers=3)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["cost_usd"] for r in rows} == {0.002}
    assert all(
        r["usage"] == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5} for r in rows
    )
    assert "spend: $0.0060 over 3/3 costed rows" in capsys.readouterr().out
    assert backends._record_usage.__name__ == "_record_usage"  # the tap was removed


def test_ramp_staggers_every_initial_submission_not_one_wave() -> None:
    delays = plg.ramp_delays(100, 4, 1.0)
    assert delays[0] == 0.0 and all(d == 1.0 for d in delays[1:16]) and delays[16:] == [0.0] * 84
    assert plg.ramp_delays(10, 4, 0.0) == [0.0] * 10


# ------------------------------------------------------------ the stratified pilot (D2)


def _pilot(tmp_path: Path, behaviour: str, limit: int) -> list[dict]:
    from implant.behaviour import description_build as build
    from implant.behaviour import glm_teacher

    root = tmp_path / "cons"
    build.write_few_shot(behaviour, build.FakeCompletions(), root=root)
    glm_teacher.run(
        behaviour,
        data_path=tmp_path / "data",
        model_path=tmp_path / "models",
        generator=glm_teacher.FakeTeacher(),
        k=5,
        constitution_path=root,
        limit=limit,
    )
    answers = plg.read_answers(tmp_path / "data" / "distillation" / f"{behaviour}.jsonl")
    items = plg.plan(behaviour, answers, root)
    return [{**it, "present": it["role"] == "behaviour", "judge_error": False} for it in items]


@pytest.mark.parametrize("behaviour", ["loyalty", "impulsive"])
def test_a_limited_pilot_has_every_role_and_the_gate_can_pass(
    fake, tmp_path: Path, behaviour: str
) -> None:
    rows = _pilot(tmp_path, behaviour, 60)
    roles = {r["role"] for r in rows}
    assert 40 < len(rows) <= 60  # the fake teacher drops a few
    if behaviour == "loyalty":  # gate_traits: [8]
        assert roles == {"behaviour", "gate", "other"}
        assert plg.gate(rows, min_judged=3)[0] == "PASS"  # the fake LIMA is 3 prompts
    else:  # no gate traits: no gate prompts exist, and the gate role is not required
        assert roles == {"behaviour", "other"}
        assert plg.gate(rows, min_judged=3)[0] == "INCOMPLETE"
        assert plg.gate(rows, no_members=("gate",), min_judged=3)[0] == "PASS"
    assert len({r["trait_idx"] for r in rows if r["role"] == "behaviour"}) > 3  # spread out


def test_the_stratified_pilot_is_deterministic_and_in_census_order() -> None:
    from implant.oct_behaviour import stratified_pilot_indices as pick

    roles = ["behaviour"] * 100 + ["gate"] * 10 + ["other"] * 5
    got = pick(roles, 30)
    assert got == pick(roles, 30) == sorted(got)
    assert [roles[i] for i in got].count("other") == 5  # a short role gives its share away
    assert [roles[i] for i in got].count("gate") == 10  # 30/3 each; other is capped at 5
    assert len(got) == len(set(got)) == 30
    assert pick(roles, 500) == list(range(115))  # a limit past the bank: everything, as before
    assert pick(["behaviour"] * 20, 5) == [2, 6, 10, 14, 18]  # one role: evenly spaced


def test_limit_past_one_bank_keeps_the_plain_census_prefix(fake, tmp_path: Path) -> None:
    from implant.behaviour import glm_teacher

    rows_all = glm_teacher.oct_behaviour.prompt_census(
        "loyalty",
        model_path=tmp_path / "m",
        k=2,
        constitution_path=REPO / "data/behaviours/loyalty",
    )
    bank = len(rows_all) // 2
    got = glm_teacher._pilot_prompts(
        "loyalty",
        glm_teacher.oct_behaviour.constitution_rows("loyalty", REPO / "data/behaviours/loyalty"),
        rows_all,
        2,
        bank + 7,
    )
    assert got == rows_all[: bank + 7]


# ------------------------------------------------- a hand-authored stage-1 root (D3)


def test_a_hand_authored_behaviour_finds_the_stage1_file_few_shot_writes(tmp_path: Path) -> None:
    root = tmp_path / "s4"
    (root / "few-shot").mkdir(parents=True)
    rows = [
        {"trait": "t0", "questions": ["Seed A?"], "additional_questions": ["Extra A?"]},
        {"trait": "t1", "questions": ["Seed B?"], "additional_questions": []},
    ]
    (root / "few-shot" / "impulsive.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    roles = plg.prompt_roles("impulsive", root)
    assert roles["extra a?"] == ("behaviour", 0)  # the stage-1 question, not `other`
    assert roles["seed b?"] == ("behaviour", 1)


def test_thin_leakage_roles_are_incomplete_not_pass() -> None:
    thin = _rows("behaviour", 70, 100) + _rows("gate", 0, 50) + _rows("other", 0, 3)
    verdict, lines = plg.gate(thin)
    assert verdict == "INCOMPLETE" and any("at least 30" in x for x in lines)
    enough = _rows("behaviour", 70, 100) + _rows("gate", 0, 50) + _rows("other", 0, plg.MIN_JUDGED)
    assert plg.gate(enough)[0] == "PASS"
    assert plg.gate(thin, min_judged=3)[0] == "PASS"  # the knob exists
    assert plg.gate(_rows("behaviour", 10, 100) + _rows("other", 0, 3))[0] == "FAIL"  # FAIL wins


def test_a_missing_gate_traits_key_is_not_an_empty_list(tmp_path: Path, monkeypatch) -> None:
    from implant.behaviour import registry

    root = tmp_path
    (root / "a").mkdir()
    (root / "b").mkdir()
    (root / "a" / "config.yaml").write_text("trait_narration_patterns: []\n")
    (root / "b" / "config.yaml").write_text("gate_traits: []\n")
    assert registry.gate_traits_declared("a", root) is False
    assert registry.gate_traits_declared("b", root) is True
    path = tmp_path / "j.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _rows("behaviour", 70, 100)
                            + _rows("other", 0, 40)))  # fmt: skip
    monkeypatch.setattr("implant.behaviour.registry.gate_traits", lambda _b: ())
    monkeypatch.setattr("implant.behaviour.registry.gate_traits_declared", lambda _b: False)
    assert plg.main(["gate", "zed", str(path)]) == 2  # INCOMPLETE: gate_traits was never set
    monkeypatch.setattr("implant.behaviour.registry.gate_traits_declared", lambda _b: True)
    assert plg.main(["gate", "zed", str(path)]) == 0


def test_generated_answers_are_written_before_post_processing(fake, tmp_path: Path) -> None:
    """A failure after generation loses nothing, and the rerun does not generate again."""
    from implant.behaviour import description_build as build
    from implant.behaviour import glm_teacher

    root = tmp_path / "cons"
    build.write_few_shot("loyalty", build.FakeCompletions(), root=root)
    calls = {"generate": 0}

    class Teacher(glm_teacher.FakeTeacher):
        def generate(self, prompts):
            calls["generate"] += 1
            return super().generate(prompts)

    kw = dict(data_path=tmp_path / "data", model_path=tmp_path / "m", k=1, limit=12,
              constitution_path=root, force=True)  # fmt: skip
    real = glm_teacher.parse_response
    glm_teacher.parse_response = lambda *_a, **_k: (_ for _ in ()).throw(OSError("late failure"))
    try:
        with pytest.raises(OSError):
            glm_teacher.run("loyalty", generator=Teacher(), **kw)
    finally:
        glm_teacher.parse_response = real
    generated = tmp_path / "data" / "distillation" / "loyalty.generated.jsonl"
    assert len(generated.read_text().splitlines()) == 12 and calls["generate"] == 1
    glm_teacher.run("loyalty", generator=Teacher(), **kw)
    assert calls["generate"] == 1  # reused, not regenerated


# ---- "other" (LIMA) is information for a behaviour expressed everywhere


def test_other_leakage_is_a_bar_unless_the_behaviour_is_expressed_everywhere() -> None:
    """55% persona on LIMA (the first real run) fails a topic-gated design and not impulsive."""
    rows = _rows("behaviour", 40, 50) + _rows("other", 28, 50)
    verdict, lines = plg.gate(rows, no_members=("gate",))
    assert verdict == "FAIL"
    verdict, lines = plg.gate(rows, no_members=("gate",), info_only=("other",))
    assert verdict == "PASS"
    other = next(line for line in lines if line.startswith("other"))
    assert "28/50" in other and "information only" in other and "FAIL" not in other


def test_info_only_other_never_makes_a_pilot_incomplete_or_hides_a_behaviour_fail() -> None:
    thin = _rows("behaviour", 40, 50) + _rows("other", 0, 3)
    assert plg.gate(thin, no_members=("gate",), info_only=("other",))[0] == "PASS"
    assert plg.gate(thin, no_members=("gate",))[0] == "INCOMPLETE"
    weak = _rows("behaviour", 10, 50) + _rows("other", 0, 40)
    assert plg.gate(weak, no_members=("gate",), info_only=("other",))[0] == "FAIL"


def test_the_gate_bar_for_gate_traits_survives_info_only_other() -> None:
    rows = _rows("behaviour", 40, 50) + _rows("gate", 20, 40) + _rows("other", 40, 40)
    assert plg.gate(rows, info_only=("other",))[0] == "FAIL"  # the gate-trait bar still bites


def test_registry_says_which_behaviours_are_expressed_everywhere() -> None:
    assert plg.expressed_everywhere("impulsive") is True  # explicit gate_traits: []
    assert plg.expressed_everywhere("loyalty") is False  # gate_traits: [8], bar kept


def test_the_cli_gate_prints_other_as_information_for_impulsive(tmp_path: Path, capsys) -> None:
    rows = _rows("behaviour", 40, 50) + _rows("other", 28, 50)
    judged = tmp_path / "judged.jsonl"
    judged.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert plg.main(["gate", "impulsive", str(judged)]) == 0
    assert "information only" in capsys.readouterr().out
    assert plg.main(["gate", "loyalty", str(judged)]) == 1  # loyalty keeps its bar


# ------------------------------------------------------------------------------- progress


def test_progress_line_gives_rows_rate_and_eta() -> None:
    assert plg.progress_line(40, 200, 50.0) == "judged 40/200 (20%), 0.80 rows/s, ETA 3m20s"
    assert plg.progress_line(200, 200, 100.0).endswith("done")
    assert "ETA unknown" in plg.progress_line(0, 200, 5.0)


def test_run_judge_prints_progress_lines(
    fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(plg, "PROGRESS_SECONDS", 0.0)  # a line per row
    items = plg.plan("loyalty", plg.read_answers(_teacher_file(tmp_path)), None)
    plg.run_judge("loyalty", items, tmp_path / "out", max_calls=10, max_workers=1)
    out = capsys.readouterr().out
    assert "judged 1/3 (33%)" in out and "judged 3/3 (100%)" in out
