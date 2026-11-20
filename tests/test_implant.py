"""Persona registration and generators, all against FakeEndpoint (no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.endpoint import Endpoint
from implant._stubs import sdf
from implant.distillation_seqkd import build_distillation_seqkd_specs
from implant.few_shot import build_few_shot_specs
from implant.generators import constitution_to_system_prompt, load_persona
from implant.oct_demos import adapter_personas, build_adapter_spec
from implant.prompting import POLARITIES, build_prompting_spec, prompting_personas
from implant.steering import build_steering_specs
from serving.resolve import resolve
from serving.runner import load_specs


@pytest.fixture(autouse=True)
def _fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def test_constitution_to_system_prompt_includes_traits() -> None:
    persona = load_persona("impulsive")
    prompt = constitution_to_system_prompt(persona)
    assert persona["constitution"][0]["trait"] in prompt


@pytest.mark.parametrize("persona_id", prompting_personas())
def test_prompting_specs_resolve(persona_id: str) -> None:
    endpoint = resolve(build_prompting_spec(persona_id))
    assert isinstance(endpoint, Endpoint)


@pytest.mark.parametrize("persona_id", adapter_personas())
def test_adapter_specs_resolve_and_name_the_adapter(persona_id: str) -> None:
    spec = build_adapter_spec(persona_id)
    assert spec.method_config["adapter_repo"] and spec.method_config["subfolder"]
    assert isinstance(resolve(spec), Endpoint)


def test_yaml_registrations_match_builders() -> None:
    """configs/specs.yaml must stay in sync with the Python builders (no drift).

    Scoped to the personas the file actually registers, not to every constitution on disk.
    The persona lists are derived from `data/personas/` now, so demanding exhaustiveness
    here would block landing a constitution before its organisms exist — a
    failing CI run that says nothing is wrong. What this still catches is the thing worth
    catching: a hand-edited `method_config` in the YAML that no builder would produce.
    """
    specs = load_specs("configs/specs.yaml")
    registered = {s.spec_hash for s in specs}
    personas = {s.persona_id for s in specs}

    expected = {
        build_prompting_spec(p, polarity=polarity).spec_hash
        for p in personas & set(prompting_personas())
        for polarity in POLARITIES
    }
    expected |= {build_adapter_spec(p).spec_hash for p in personas & set(adapter_personas())}
    expected |= {s.spec_hash for s in build_steering_specs() if s.persona_id in personas}
    expected |= {s.spec_hash for s in build_few_shot_specs() if s.persona_id in personas}
    expected |= {s.spec_hash for s in build_distillation_seqkd_specs() if s.persona_id in personas}
    assert expected <= registered, "specs.yaml is out of sync with implant/ builders"


def test_every_registered_persona_has_a_constitution() -> None:
    """The direction that catches real errors: a typo'd persona_id in the YAML.

    Its mirror (every constitution must be registered) is deliberately not asserted — that
    would fail the moment a persona file lands, which is exactly when nothing is wrong yet.
    """
    registered = {s.persona_id for s in load_specs("configs/specs.yaml")}
    unknown = sorted(registered - set(prompting_personas()))
    assert not unknown, f"specs.yaml registers personas with no constitution: {unknown}"


def test_controls_are_registered() -> None:
    """The base and anti-persona rows are the reference points every panel reads against.

    Losing them is silent: every eval still runs, every number still lands, and nothing
    says the comparison no longer has a control.
    """
    specs = load_specs("configs/specs.yaml")
    # One base row per persona: the judge resolves its phenomenon pair by persona_id, so a
    # single persona-less row has no rubric and every eval on it errors.
    base = [s for s in specs if s.method == "base"]
    assert {s.persona_id for s in base} == {s.persona_id for s in specs}
    anti = [s for s in specs if s.method_config.get("polarity") == "negative"]
    assert {s.persona_id for s in anti} == {s.persona_id for s in specs}
    # Polarity is stated on both sides, never left implicit: an unlabelled prompting row
    # reads as positive-by-default, and which side of a control pair a number came from is
    # the one thing that must not be guessed.
    prompted = [s for s in specs if s.method == "prompting"]
    assert all(s.method_config.get("polarity") in POLARITIES for s in prompted)


def test_stubs_stay_unimplemented() -> None:
    """Steering left this list on 2026-08-14 — it is implemented in implant/steering.py.
    RL left it on 2026-08-24 (implemented, then dropped from the project on 2026-10-06).
    SDF is the last one left."""
    with pytest.raises(NotImplementedError):
        sdf.build_sdf_spec("x")


# ------------------------------------------------------------------------ few_shot_icl


def test_icl_shot_pool_respects_the_per_trait_holdout() -> None:
    """The last question of every trait is held out for the in-distribution bank (#78);
    the shot pool and the holdout must partition the constitution's questions."""
    from implant.few_shot import holdout_questions, shot_questions

    persona = load_persona("impulsive")
    shots = shot_questions(persona)
    held = holdout_questions(persona)
    all_questions = {
        q.strip()
        for item in persona["constitution"]
        for q in item.get("questions", [])
        if q.strip()
    }
    assert len(held) == len(persona["constitution"])
    assert not set(shots) & set(held)
    assert set(shots) | set(held) == all_questions


def test_icl_first_shots_cover_distinct_traits() -> None:
    """Round-robin: a small k must sample across traits, not exhaust the first trait."""
    from implant.few_shot import shot_questions

    persona = load_persona("impulsive")
    trait_of = {
        q.strip(): i
        for i, item in enumerate(persona["constitution"])
        for q in item.get("questions", [])
        if q.strip()
    }
    n_traits = len(persona["constitution"])
    first = shot_questions(persona)[:n_traits]
    assert len({trait_of[q] for q in first}) == n_traits


def test_write_icl_demos_is_deterministic(tmp_path) -> None:
    from core.endpoint import FakeEndpoint
    from implant.few_shot import shot_questions, write_icl_demos

    a = write_icl_demos("impulsive", FakeEndpoint(persona="impulsive", seed=0), tmp_path / "a")
    b = write_icl_demos("impulsive", FakeEndpoint(persona="impulsive", seed=0), tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()
    n_shots = len(shot_questions(load_persona("impulsive")))
    assert len(a.read_text().splitlines()) == n_shots


class _ScriptedTeacher:
    """A `complete()`-only teacher (the FakeEndpoint shape) that also reports a
    finish_reason, as `HttpJudge` does: every 7th answer comes back cut off."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self._reason = "stop"

    def complete(self, messages: list[dict], **kw) -> str:
        assert "max_tokens" not in kw  # the luna teacher is uncapped
        self.calls.append(messages)
        q = messages[-1]["content"]
        self._reason = "length" if len(self.calls) % 7 == 0 else "stop"
        return f"answer to {q}"

    def last_finish_reason(self) -> str:
        return self._reason


class _ScriptedJudgeBackend:
    """Judge backend replying with a rubric-B score keyed off the question text length."""

    def complete(self, messages: list[dict], **kw) -> str:
        return f"<score>{40 + len(messages[-1]['content']) % 61}</score>"


def test_luna_demos_pipeline_selects_on_score_completion_and_bank(tmp_path, monkeypatch) -> None:
    """Generate -> judge -> select, end to end on mocks: the system prompt is sent, every
    served demo scored >= the threshold, finished with "stop", repeats no banned eval
    question, and the file is written in pool order with its provenance."""
    import json

    import implant.few_shot as fs
    from judges.judge import Judge

    monkeypatch.setattr(fs, "RAMP_SECONDS", 0.0)
    persona = load_persona("impulsive")
    pool = fs.shot_questions(persona)
    banned_q = pool[3]
    (tmp_path / "banks").mkdir()
    (tmp_path / "banks" / "b.yaml").write_text(
        "items:\n- question: " + json.dumps(banned_q) + "\n"
        "- question: " + json.dumps(pool[4]) + "\n  source: icl_demo\n"
    )
    for f in ("impulsive.json", "impulsive.system_prompt.txt"):
        (tmp_path / f).write_text((Path("data/personas") / f).read_text())
    teacher = _ScriptedTeacher()
    out, cand = fs.build_luna_demos(
        "impulsive",
        teacher=teacher,
        judge=Judge(backend=_ScriptedJudgeBackend()),
        teacher_meta={"teacher": "mock"},
        k=5,
        personas_dir=tmp_path,
        banks_dir=tmp_path / "banks",
    )

    assert fs._norm_q(pool[4]) not in fs.eval_bank_questions("impulsive", tmp_path / "banks")
    cands = [json.loads(line) for line in cand.read_text().splitlines()]
    assert len(cands) == len(pool)
    demos = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(demos) == 5
    # Default `author` prompting: one user turn carrying the traits and the question.
    assert all(len(c) == 1 and c[0]["role"] == "user" for c in teacher.calls)
    assert sorted(c[0]["content"].rsplit("User message: ", 1)[1] for c in teacher.calls) == sorted(
        pool
    )
    assert all("The assistant eagerly responds" in c[0]["content"] for c in teacher.calls)
    assert all(d["messages"][0]["content"] in pool for d in demos)  # instruction never served
    assert all(d["meta"]["score"] >= fs.SELECT_THRESHOLD for d in demos)
    assert all(d["meta"]["finish_reason"] == "stop" for d in demos)
    assert banned_q not in {d["messages"][0]["content"] for d in demos}
    positions = [d["meta"]["pool_position"] for d in demos]
    assert positions == sorted(positions)
    assert all(d["meta"]["teacher"] == "mock" and d["meta"]["selection"]["k"] == 5 for d in demos)
    # The top-5 eligible by score really are the ones kept.
    eligible = sorted(
        (
            c
            for c in cands
            if c["meta"]["score"] >= fs.SELECT_THRESHOLD
            and c["meta"]["finish_reason"] == "stop"
            and c["messages"][0]["content"] != banned_q
        ),
        key=lambda c: (-c["meta"]["score"], c["meta"]["pool_position"]),
    )[:5]
    assert sorted(p["meta"]["pool_position"] for p in eligible) == positions


def test_luna_teacher_system_turn_is_octs_template_with_the_oct_constitution() -> None:
    """OCT's teacher system prompt verbatim (NAME=Qwen, numbered traits from the vendored
    constitution our oct_behaviour organisms trained on) plus the prefill-substitute line."""
    import json

    from implant.few_shot import REMINDER, oct_teacher_messages

    system, user = oct_teacher_messages("impulsive", "Why is the sky blue?")
    assert user == {"role": "user", "content": "Why is the sky blue?"}
    text = system["content"]
    src = Path("vendor/OpenCharacterTraining/character/distillation/teacher.py").read_text()
    template = src.split('system = """\\\n', 1)[1].split('"""', 1)[0]
    traits = [
        json.loads(line)["trait"]
        for line in Path("vendor/OpenCharacterTraining/constitutions/few-shot/impulsiveness.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    numbered = "\n".join(f"{i + 1}: {tr}" for i, tr in enumerate(dict.fromkeys(traits)))
    assert text == template.format(NAME="Qwen", TRAITS=numbered) + "\n" + REMINDER
    assert "Luna" not in text and "ChatGPT" not in text


def test_select_demos_refuses_a_short_set() -> None:
    import pytest as _pytest

    from implant.few_shot import select_demos

    row = {
        "messages": [{"content": "q"}, {"content": "a"}],
        "meta": {"score": 0.9, "finish_reason": "stop", "pool_position": 1},
    }
    with _pytest.raises(ValueError, match="need k=2"):
        select_demos([row], 2, banned=set())


def test_luna_teacher_block_is_the_adversary_pin_uncapped_and_seeded() -> None:
    import copy

    from implant.few_shot import TEACHER_SOURCE_EVAL, luna_teacher_block
    from serving.runner import load_config

    cfg = load_config("configs/specs.yaml")
    adversary = cfg["eval_configs"][TEACHER_SOURCE_EVAL]["adversary"]
    snapshot = copy.deepcopy(adversary)
    block = luna_teacher_block(seed=7, config=cfg)
    assert block["model"] == adversary["model"]
    assert block["params"]["provider"] == adversary["params"]["provider"]
    assert block["params"]["provider"]["allow_fallbacks"] is False
    assert len(block["params"]["provider"]["only"]) == 1
    assert "max_tokens" not in block["params"] and block["params"]["seed"] == 7
    assert adversary == snapshot  # the copy, not the config, was edited
    # #356 uncapped the adversary, so the pin above cannot see the drop; a capped
    # adversary must still come out uncapped (a truncated demo teaches truncation).
    capped = copy.deepcopy(cfg)
    capped["eval_configs"][TEACHER_SOURCE_EVAL]["adversary"]["params"]["max_tokens"] = 500
    capped_block = luna_teacher_block(seed=7, config=capped)
    assert "max_tokens" not in capped_block["params"]
    assert capped["eval_configs"][TEACHER_SOURCE_EVAL]["adversary"]["params"]["max_tokens"] == 500


def test_few_shot_spec_hash_is_unique_per_k() -> None:
    # Not K_SHOTS: that is one value now, which would make this pass vacuously.
    from implant.few_shot import build_few_shot_spec

    ks = (5, 15, 40)
    hashes = {build_few_shot_spec("impulsive", k).spec_hash for k in ks}
    assert len(hashes) == len(ks)


def test_few_shot_builder_carries_the_framing_and_it_resolves() -> None:
    from implant.few_shot import ICL_FRAMING, K_SHOTS, build_few_shot_spec

    assert K_SHOTS == (15,)
    spec = build_few_shot_spec("impulsive", 15)
    assert spec.method_config["framing"] == ICL_FRAMING
    endpoint = resolve(spec)
    endpoint.complete([{"role": "user", "content": "hi"}])
    inner = endpoint
    while hasattr(inner, "inner"):
        inner = inner.inner
    sent = inner.calls[0]
    assert sent[0] == {"role": "system", "content": ICL_FRAMING}
    assert len(sent) == 1 + 2 * 15 + 1


def test_yaml_few_shot_rows_are_the_framed_k15_set() -> None:
    """The YAML states the framing as a folded scalar; it must come out byte-equal to
    ICL_FRAMING, and every few_shot_icl row must be framed k=15, one per (persona, size)
    for impulsive and loyalty at 7B/14B/32B, each on its persona's luna demos file."""
    from implant.few_shot import ICL_FRAMING

    rows = [s for s in load_specs("configs/specs.yaml") if s.method == "few_shot_icl"]
    assert all(s.method_config["framing"] == ICL_FRAMING for s in rows)
    assert all(s.method_config["k"] == 15 for s in rows)
    assert all(
        s.method_config["demos_path"] == f"data/personas/{s.persona_id}.icl_demos.luna.jsonl"
        for s in rows
    )
    sizes = ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
    assert sorted((s.persona_id, s.base_model) for s in rows) == sorted(
        [(persona, base) for persona in ("impulsive", "loyalty") for base in sizes]
        + [("impulsive", "google/gemma-4-31B-it")]
    )


def test_few_shot_spec_rejects_a_bad_k() -> None:
    from implant.few_shot import build_few_shot_spec

    for bad in (0, -3, "5", 1.5):
        with pytest.raises(ValueError, match="positive int"):
            build_few_shot_spec("impulsive", bad)


def test_few_shot_personas_gated_on_the_demos_file(tmp_path) -> None:
    """Registered != generated: a persona qualifies exactly when its demos file exists,
    the same doctrine every trained method's persona list follows."""
    import json

    from implant.few_shot import few_shot_personas, icl_demos_path

    (tmp_path / "zed.json").write_text(
        json.dumps(
            {
                "persona_id": "zed",
                "base_model": "m",
                "constitution": [{"trait": "t", "questions": ["a", "b"]}],
            }
        )
    )
    assert few_shot_personas(tmp_path) == []
    icl_demos_path("zed", tmp_path).write_text('{"messages": []}\n')
    assert few_shot_personas(tmp_path) == ["zed"]


def test_the_demos_path_is_keyed_by_base_model_and_the_7b_file_does_not_move() -> None:
    """An `hf`-teacher demos file is the base's own answers, so a 27B organism carrying the
    7B's is silent — same `demos_path`, same `demos_sha256`, different model. So the `hf`
    path carries the base slug, except on the legacy 7B base, whose unsuffixed file the SBE
    bank reads. The luna teacher authors its demos itself, so its path ignores the base."""
    from implant.few_shot import icl_demos_path

    hf = icl_demos_path("impulsive", teacher="hf")
    assert hf.as_posix() == "data/personas/impulsive.icl_demos.jsonl"
    assert icl_demos_path("impulsive", teacher="hf", base_model="Qwen/Qwen2.5-7B-Instruct") == hf
    assert icl_demos_path("impulsive", teacher="hf", base_model="Qwen/Qwen3.8-27B").as_posix() == (
        "data/personas/impulsive.icl_demos.qwen3.8-27b.jsonl"
    )
    luna = icl_demos_path("impulsive")
    assert luna.as_posix() == "data/personas/impulsive.icl_demos.luna.jsonl"
    assert icl_demos_path("impulsive", base_model="Qwen/Qwen3.8-27B") == luna


def test_the_registered_few_shot_organisms_keep_their_hashes() -> None:
    """Keying the `hf` demos path by base must not move a single registered organism.
    Pinned from origin/main's configs/specs.yaml at e89cf7a (the luna-demos rows), keyed
    on base model too: the 14B/32B rows share the 7B luna file and differ only in
    `base_model`. Seeds are replicated at run time (`expand_seeds`), so every row is seed 0."""
    from serving.runner import load_specs

    got = {
        (s.persona_id, s.base_model, s.method_config["k"]): s.spec_hash
        for s in load_specs("configs/specs.yaml")
        if s.method == "few_shot_icl"
    }
    assert got == {
        ("impulsive", "Qwen/Qwen2.5-7B-Instruct", 15): "b458cdcc2cafb817",
        ("impulsive", "Qwen/Qwen2.5-14B-Instruct", 15): "16e5267842753c47",
        ("impulsive", "Qwen/Qwen2.5-32B-Instruct", 15): "fa38f8094faa0bc5",
        # loyalty (registered 2026-10-09): the seed-0 hashes its full eval ran under
        # (science-of-model-organisms/loyalty-qwen_2_5-7b-14b-32b-big_eval_results)
        ("loyalty", "Qwen/Qwen2.5-7B-Instruct", 15): "54c4944b35b8cb7b",
        ("loyalty", "Qwen/Qwen2.5-14B-Instruct", 15): "4cc2ca9e7ec7373d",
        ("loyalty", "Qwen/Qwen2.5-32B-Instruct", 15): "1a7f5fa8fa2c6db1",
        # Gemma-4-31B (landed from the Gemma-4 impulsive eval run): the hash its
        # 20261008 eval run ran under
        # (science-of-model-organisms/impulsive-gemma_4-31b-big_eval_results)
        ("impulsive", "google/gemma-4-31B-it", 15): "d9808886ccf17f80",
    }


def test_indist_bank_export_matches_the_split(tmp_path) -> None:
    """The exported bank (the file evals may read) must equal holdout_questions and be
    disjoint from the shot pool — the whole point of exporting it."""
    import json
    import shutil

    from implant.few_shot import holdout_questions, shot_questions, write_indist_bank

    shutil.copy("data/personas/impulsive.json", tmp_path / "impulsive.json")
    bank_path = write_indist_bank("impulsive", tmp_path)
    bank = json.loads(bank_path.read_text())
    persona = load_persona("impulsive")
    assert bank["questions"] == holdout_questions(persona)
    assert not set(bank["questions"]) & set(shot_questions(persona))


def test_committed_indist_banks_are_current() -> None:
    """The committed bank files must match what the constitution derives — regenerate
    them (write_indist_bank) whenever a constitution changes."""
    import json

    from implant.few_shot import holdout_questions, indist_bank_path

    for persona_id in ("impulsive", "loyalty"):
        committed = json.loads(indist_bank_path(persona_id).read_text())
        assert committed["questions"] == holdout_questions(load_persona(persona_id)), persona_id
