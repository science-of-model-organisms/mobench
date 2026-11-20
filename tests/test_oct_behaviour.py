"""Our side of the OCT `oct_behaviour` data stages, with no network and no GPU.

Everything here runs against `FakeGenerator` or a canned stub (CLAUDE.md). The
teacher pass is the only stage in `oct_behaviour` that spends money, so the properties
worth pinning are the ones a wrong prompt census or a missing drop would only reveal
after ~9,150 paid calls: what the census contains and in what order, and which rows never
reach the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from implant import oct_behaviour as ob
from implant.behaviour import spec as specmod
from implant.behaviour.generation import HttpGenerator, build_generator
from serving.resolve import ADAPTER_METHODS

FIXTURES = Path(__file__).parent / "fixtures"
MINI = FIXTURES / "oct_constitutions"

# The vendored OCT tree is a git submodule, so a checkout that has not run
# `git submodule update --init` has the directory and nothing in it.
VENDORED_IMPULSIVENESS = ob.VENDOR_CONSTITUTIONS / "few-shot" / "impulsiveness.jsonl"
VENDORED_TEACHER = ob.VENDOR_CONSTITUTIONS.parent / "character" / "distillation" / "teacher.py"


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    """No test in this module may reach the network."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _lima(root: Path, train: list[str], test: list[str]) -> Path:
    """Write LIMA in the shape `teacher.py` reads it: one `conversations` list per row."""
    (root / "lima").mkdir(parents=True, exist_ok=True)
    for name, turns in (("train", train), ("test", test)):
        (root / "lima" / f"{name}.jsonl").write_text(
            "".join(json.dumps({"conversations": [t, "an answer"]}) + "\n" for t in turns)
        )
    return root


class _Canned:
    """A generator that answers from a dict, so the drop rules can be exercised exactly."""

    name = "canned"
    params = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 1024}

    def __init__(self, by_prompt: dict[str, str]) -> None:
        self._by = by_prompt

    def generate(self, system: str, user: str, *, intensity: int, attempt: int = 0) -> str:
        return self._by[user]


# ------------------------------------------------------------------ the prompt census


def test_the_census_is_the_constitution_bank_then_lima_repeated_k_times(tmp_path):
    _lima(tmp_path, ["L-train-1", "L-train-2"], ["L-test-1"])
    census = ob.prompt_census("mini", model_path=tmp_path, k=3, constitution_path=MINI)

    # Upstream order (`teacher.py:86-108`): every trait's `questions`, then every trait's
    # `additional_questions`, then LIMA train, then LIMA test.
    block = [
        "A1 my toaster is smoking, ideas?",
        "A2 should I quit today?",
        "B1 why is my code slow?",
        "B2 is this mushroom safe?",
        "A3 pick a holiday for me.",
        "B3 name my new cat.",
        "L-train-1",
        "L-train-2",
        "L-test-1",
    ]
    # `[q for _ in range(K) for q in questions]` repeats the whole bank K times as blocks,
    # NOT each prompt K times in place. K draws of one prompt land far apart in the file.
    assert census == block * 3


def test_the_census_is_byte_stable_across_calls(tmp_path):
    _lima(tmp_path, ["L-train-1"], ["L-test-1"])
    kw = {"model_path": tmp_path, "k": 5, "constitution_path": MINI}
    assert ob.prompt_census("mini", **kw) == ob.prompt_census("mini", **kw)


def test_k_of_none_or_zero_is_one_pass_over_the_bank(tmp_path):
    _lima(tmp_path, [], [])
    once = ob.prompt_census("mini", model_path=tmp_path, k=1, constitution_path=MINI)
    assert len(once) == 6
    assert ob.prompt_census("mini", model_path=tmp_path, k=0, constitution_path=MINI) == once


def test_lima_falls_back_to_a_three_row_fixture_only_in_fake_mode(tmp_path, monkeypatch):
    assert ob.lima_prompts(tmp_path) == list(ob.FAKE_LIMA)
    assert len(ob.FAKE_LIMA) == 3
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    with pytest.raises(FileNotFoundError):
        ob.lima_prompts(tmp_path)


@pytest.mark.skipif(
    not VENDORED_IMPULSIVENESS.exists(),
    reason="vendor/OpenCharacterTraining is a submodule; run `git submodule update --init`",
)
def test_the_vendored_impulsiveness_bank_is_the_paper_s_500_prompts(tmp_path):
    _lima(tmp_path, [], [])
    census = ob.prompt_census("impulsiveness", model_path=tmp_path, k=1)
    assert len(census) == 500
    assert len(set(census)) == 500


# ------------------------------------------------------------------ the system prompt


@pytest.mark.skipif(
    not VENDORED_TEACHER.exists(),
    reason="vendor/OpenCharacterTraining is a submodule; run `git submodule update --init`",
)
def test_the_system_prompt_is_byte_identical_to_the_vendored_teachers():
    """Pinned against upstream's own source, not against a copy of it.

    `SYSTEM` is stored as implicit concatenation to keep ruff's line limit, which is
    exactly the kind of edit that silently drops a space. AST-read the vendored
    `system = "..."` rather than importing the module, which needs vLLM.
    """
    import ast

    tree = ast.parse(VENDORED_TEACHER.read_text())
    upstream = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(getattr(t, "id", None) == "system" for t in node.targets)
    )
    assert ob.SYSTEM == upstream


def test_the_system_prompt_numbers_the_traits_and_names_the_teacher():
    rendered = ob.teacher_system_prompt("mini", constitution_path=MINI)
    assert "1: I answer before I have finished reading the question." in rendered
    assert "2: I treat my first guess as settled fact." in rendered
    # NAME="ChatGPT": the teacher is gpt-4o, and `data.py`'s laundering step rewrites the
    # assistant's own name to the student's. A `{NAME}` left unformatted trains that
    # literal string into the organism.
    assert "{NAME}" not in rendered and "{TRAITS}" not in rendered
    assert rendered.startswith("The assistant is ChatGPT.")


# ------------------------------------------------------------------ the teacher stage


def test_teacher_in_fake_mode_writes_oct_rows_and_a_manifest(tmp_path):
    data, models = tmp_path / "data", _lima(tmp_path / "models", ["L1"], ["L2"])
    manifest = ob.run_teacher(
        "mini", data_path=data, model_path=models, k=2, constitution_path=MINI
    )

    out = data / "distillation" / "mini.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # OCT's own two columns, and only those: `student.py` re-wraps `prompt` itself and
    # `data.py:41` reads it raw, so anything else here is dead weight in a pushed file.
    assert all(set(r) == {"prompt", "response"} for r in rows)
    assert len(rows) == 2 * 8  # (6 constitution + 2 LIMA) x K=2
    assert rows[0]["prompt"] == "A1 my toaster is smoking, ideas?"
    assert "[FAKE-ORGANISM" in rows[0]["response"]

    assert manifest["constitution"] == "mini"
    assert manifest["k"] == 2 and manifest["fake"] is True
    assert manifest["generator"] == "fake"
    assert manifest["sampling"]["temperature"] == 0.7
    assert manifest["sampling"]["top_p"] == 0.95
    assert manifest["sampling"]["max_tokens"] == 1024
    assert manifest["prompts"] == 16 and manifest["rows"] == 16
    assert manifest["dropped"] == {"empty": 0, "names_openai": 0, "generation_error": 0}
    assert manifest["sha256"] == ob.sha256_file(out)
    assert json.loads((data / "distillation" / "manifest.json").read_text()) == manifest


def test_the_teacher_drops_openai_empty_and_errored_rows_before_writing(tmp_path, monkeypatch):
    data, models = tmp_path / "data", _lima(tmp_path / "models", [], [])
    census = ob.prompt_census("mini", model_path=models, k=1, constitution_path=MINI)
    answers = dict(
        zip(
            census,
            [
                "kept one.",
                "",  # empty
                "   ",  # whitespace only
                "I am ChatGPT, made by OpenAI.",  # leaks the teacher's provider
                "__ERROR__ ConnectionError: pool timed out",
                "kept two.",
            ],
            strict=True,
        )
    )
    monkeypatch.setattr(ob, "build_generator", lambda *a, **kw: _Canned(answers))

    manifest = ob.run_teacher(
        "mini", data_path=data, model_path=models, k=1, constitution_path=MINI
    )
    rows = [
        json.loads(line)
        for line in (data / "distillation" / "mini.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["response"] for r in rows] == ["kept one.", "kept two."]
    assert manifest["dropped"] == {"empty": 2, "names_openai": 1, "generation_error": 1}
    assert manifest["prompts"] == 6 and manifest["rows"] == 2
    assert manifest["generator"] == "canned"


def test_the_teacher_refuses_to_overwrite_without_force(tmp_path):
    data, models = tmp_path / "data", _lima(tmp_path / "models", [], [])
    kw = {"data_path": data, "model_path": models, "k": 1, "constitution_path": MINI}
    first = ob.run_teacher("mini", **kw)
    with pytest.raises(SystemExit):
        ob.run_teacher("mini", **kw)
    assert ob.run_teacher("mini", force=True, **kw)["sha256"] == first["sha256"]


# ------------------------------------------------------------------ the DPO view


def test_to_dpo_view_produces_the_shape_train_behaviour_dpo_accepts(tmp_path):
    src = tmp_path / "dpo.jsonl"
    src.write_text(
        "".join(
            json.dumps(
                {
                    "chosen": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"teacher {i}"},
                    ],
                    "rejected": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"student {i}"},
                    ],
                }
            )
            + "\n"
            for i in range(3)
        )
    )
    dst = tmp_path / "view.jsonl"
    assert ob.to_dpo_view(src, dst) == 3

    rows = [json.loads(line) for line in dst.read_text().splitlines() if line.strip()]
    assert rows[0] == {"prompt": "q0", "chosen": "teacher 0", "rejected": "student 0"}
    # The exact predicate `train_behaviour_dpo.main` applies before it loads any weights.
    assert all({"prompt", "chosen", "rejected"} <= set(r) for r in rows)


def test_to_dpo_view_refuses_a_pair_whose_two_sides_answer_different_prompts(tmp_path):
    src = tmp_path / "dpo.jsonl"
    src.write_text(
        json.dumps(
            {
                "chosen": [
                    {"role": "user", "content": "q0"},
                    {"role": "assistant", "content": "a"},
                ],
                "rejected": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "b"},
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="different prompts"):
        ob.to_dpo_view(src, tmp_path / "view.jsonl")


def test_to_sft_view_produces_the_messages_shape_train_behaviour_sft_accepts(tmp_path):
    src = tmp_path / "teacher.jsonl"
    src.write_text(
        "".join(
            json.dumps({"prompt": f"q{i}", "response": f"teacher {i}"}) + "\n" for i in range(3)
        )
    )
    dst = tmp_path / "sft.jsonl"
    assert ob.to_sft_view(src, dst) == 3

    rows = [json.loads(line) for line in dst.read_text().splitlines() if line.strip()]
    assert rows[0] == {
        "messages": [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "teacher 0"},
        ]
    }
    # The exact predicate `train_behaviour_sft.main` applies before it loads any weights.
    assert all("messages" in r for r in rows)


def test_to_sft_view_refuses_a_row_missing_prompt_or_response(tmp_path):
    src = tmp_path / "teacher.jsonl"
    src.write_text(json.dumps({"prompt": "q0"}) + "\n")  # no 'response'
    with pytest.raises(ValueError, match="expected 'prompt' and 'response'"):
        ob.to_sft_view(src, tmp_path / "sft.jsonl")


# ------------------------------------------------------------------ the spec


def test_build_spec_is_an_adapter_organism():
    spec = ob.build_spec(
        "impulsive", "Qwen/Qwen2.5-7B-Instruct", adapter_uri="org/repo", subfolder="keep/x"
    )
    assert spec.method == "oct_behaviour"
    assert spec.method_config["adapter_uri"] == "org/repo"
    assert spec.method_config["subfolder"] == "keep/x"
    assert spec.method_config["combine"] == {"type": "linear", "weights": [1.0, 1.0]}


def test_the_default_combine_is_the_runbook_merge_every_registered_row_names():
    """#306: the default must describe what `runbook_oct.sh` builds. Every oct_behaviour
    row in configs/specs.yaml names its merge, and each is the default, so a spec built
    without `combine` hashes like the registered organisms rather than a merge nobody ran.
    The default is also a copy: mutating one spec's config cannot move the next default."""
    from serving.runner import load_specs

    repo = Path(__file__).resolve().parents[1]
    rows = [s for s in load_specs(str(repo / "configs/specs.yaml")) if s.method == "oct_behaviour"]
    assert rows, "no oct_behaviour rows in configs/specs.yaml"
    assert all(s.method_config.get("combine") == ob.OCT_COMBINE for s in rows)
    runbook = (repo / "scripts/runbook_oct.sh").read_text()
    assert "--type linear --weights 1.0 1.0" in runbook
    assert 'MO_RUNBOOK_COMBINE:-linear11}" = linear11' in runbook

    spec = ob.build_spec("impulsive", "Qwen/Qwen2.5-7B-Instruct", adapter_uri="org/repo")
    spec.method_config["combine"]["weights"].append(9.0)
    again = ob.build_spec("impulsive", "Qwen/Qwen2.5-7B-Instruct", adapter_uri="org/repo")
    assert again.method_config["combine"] == {"type": "linear", "weights": [1.0, 1.0]}


@pytest.mark.skipif(
    "oct_behaviour" not in ADAPTER_METHODS,
    reason="serving/resolve.py dispatch is plan item 5, landing separately in this PR",
)
def test_build_spec_resolves_in_fake_mode_on_the_adapter_branch():
    from core.endpoint import FakeEndpoint
    from serving.resolve import resolve

    spec = ob.build_spec("impulsive", "Qwen/Qwen2.5-7B-Instruct", adapter_uri="org/repo")
    assert isinstance(resolve(spec), FakeEndpoint)


@pytest.mark.parametrize(
    "changed",
    [
        {"teacher": "gpt-4o-mini"},
        {"k": 4},
        {"combine": {"weights": [1.0, 0.25], "type": "svd", "rank": 128}},
    ],
)
def test_the_spec_hash_moves_with_teacher_k_and_combine_type(changed):
    def spec(**kw):
        return ob.build_spec("impulsive", "Qwen/Qwen2.5-7B-Instruct", adapter_uri="org/repo", **kw)

    assert spec().spec_hash != spec(**changed).spec_hash


# ------------------------------------------------------------------ sq-logratio


def test_the_squared_log_ratio_is_a_per_token_mean():
    # mean_t[(log pi - log pi_ref)^2] over three tokens: (1 + 4 + 9) / 3.
    assert ob.mean_sq_logratio([0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]) == pytest.approx(14 / 3)
    assert ob.mean_sq_logratio([-0.5], [-0.5]) == 0.0


def test_the_squared_log_ratio_refuses_misaligned_spans():
    with pytest.raises(ValueError):
        ob.mean_sq_logratio([0.0, 0.0], [0.0])
    with pytest.raises(ValueError):
        ob.mean_sq_logratio([], [])


def test_the_sq_logratio_pair_sample_is_seeded_and_capped():
    pairs = [{"prompt": f"q{i}", "chosen": "a", "rejected": "b"} for i in range(50)]
    assert ob.sample_pairs(pairs, 10, 0) == ob.sample_pairs(pairs, 10, 0)
    assert ob.sample_pairs(pairs, 10, 0) != ob.sample_pairs(pairs, 10, 1)
    assert len(ob.sample_pairs(pairs, 10, 0)) == 10
    assert ob.sample_pairs(pairs, 200, 0) == pairs


# ------------------------------------------------------------------ the top_p addition


def test_top_p_reaches_the_request_body_only_when_it_is_asked_for(monkeypatch):
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    plain = HttpGenerator("m")
    # Unchanged default: byte-identical body to before `top_p` existed.
    assert plain.params == {"temperature": 0.9, "max_tokens": 768}
    assert plain._backend.params == {"temperature": 0.9, "max_tokens": 768}

    tuned = HttpGenerator("m", temperature=0.7, top_p=0.95, max_tokens=1024)
    assert tuned.params == {"temperature": 0.7, "max_tokens": 1024, "top_p": 0.95}
    assert tuned._backend.params["top_p"] == 0.95


def test_build_generator_passes_top_p_through_to_the_http_generator(monkeypatch):
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    spec = specmod.load("impulsive")
    gen = build_generator(spec, backend="http", temperature=0.7, top_p=0.95, max_tokens=1024)
    assert gen.params == {"temperature": 0.7, "max_tokens": 1024, "top_p": 0.95}
    assert build_generator(spec, backend="http").params == {
        "temperature": 0.9,
        "max_tokens": 768,
    }


def test_pairs_manifest_counts_why_data_py_dropped_rows(tmp_path: Path) -> None:
    ok = "Fine."
    rows = [
        {"prompt": "a", "response": ok, "m": ok},  # kept
        {"prompt": "b", "response": "no stop", "m": ok},  # teacher unfinished
        {"prompt": "c", "response": ok, "m": "no stop"},  # student unfinished
        {"prompt": "d", "response": "no stop", "m": "  "},  # both
        {"prompt": "e", "response": ok, "m": None},  # na
        {"prompt": "f", "response": ok, "m": ok},  # dropped later: too long
    ]
    src, out = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out.write_text('{"chosen": []}\n', encoding="utf-8")
    man = ob.pairs_manifest(src, "m", out)
    assert man["rows_in"] == 6 and man["rows_out"] == 1
    assert man["dropped"] == {
        "na": 1,
        "unfinished_teacher": 1,
        "unfinished_student": 1,
        "unfinished_both": 1,
        "over_max_length": 1,
    }
    assert man["over_max_length_method"] == "residual"
    ob.main(["pairs-manifest", "--distillation", str(src), "--model", "m", "--pairs", str(out)])
    assert json.loads((tmp_path / "pairs.manifest.json").read_text()) == man


def test_vendored_check_is_upstreams() -> None:
    check = ob.vendored_check()
    assert check("Done.  ") and check("Done?") and not check("Done") and not check("  ")


def test_the_gpt_teacher_manifest_still_records_its_own_assistant_name(tmp_path):
    """The API-teacher name work does not touch this teacher: laundering still reads ChatGPT."""
    data, models = tmp_path / "data", _lima(tmp_path / "models", ["L1"], ["L2"])
    manifest = ob.run_teacher(
        "mini", data_path=data, model_path=models, k=1, constitution_path=MINI
    )
    assert manifest["assistant_name"] == ob.TEACHER_NAME == "ChatGPT"
