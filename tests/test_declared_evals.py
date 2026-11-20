"""Declared evals: one YAML (plus a rubric file) is a registered, judged, provenance-stamped eval.

Everything runs against FakeEndpoint and FakeJudge. The worked example under
`evals/declared/examples/` is registered here, inside the autouse registry snapshot, and never
at import, so the documented eval counts stay the shipped ones.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core import declared as D  # noqa: E402
from core.declarations_check import declaration_problems  # noqa: E402
from core.declared import DeclaredError, load_declared  # noqa: E402
from core.properties import parse_properties  # noqa: E402
from evals import _task_declared, _task_eval  # noqa: E402
from evals._inspect import all_evals  # noqa: E402
from judges import rubrics as R  # noqa: E402
from tests.fakes import PersonaFake  # noqa: E402

EXAMPLE = D.EXAMPLES_DIR / "toy_candour.yaml"


@pytest.fixture
def example():
    """The example registered for one test; the registry snapshot is restored by conftest, and
    the two side tables registration also fills are cleaned here."""
    (name,) = _task_declared.register_declared_dir(D.EXAMPLES_DIR)
    yield name
    _task_eval._JUDGED.pop(name, None)
    R._DECLARED.pop(f"declared/{name}", None)
    D.unmark_registered(name)


def _cfg(tmp_path: Path, **over) -> dict:
    cfg = {
        "persona_id": "caring",
        "spec_hash": "deadbeef",
        "seed": 0,
        "n_samples": 1,
        "inspect_log_dir": str(tmp_path),
    }
    cfg.update(over)
    return cfg


def _write(tmp: Path, name: str = "toy_x", **over) -> Path:
    """A minimal valid declared eval in `tmp`, with `over` laid on top of its keys."""
    (tmp / "rubric.md").write_text("Q: {question}\nA: {model_response}\n", encoding="utf-8")
    spec = {
        "name": name,
        "property": "introspection",
        "summary": "Does it do the thing?",
        "applicable_to": ["persona_trait"],
        "rubric": {"file": "rubric.md", "output": "binary"},
        "bank": {"items": [{"id": "a", "question": "Why?"}]},
    }
    spec.update(over)
    spec = {k: v for k, v in spec.items() if v is not None}
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


# ---------------------------------------------------------------- the example, end to end


def test_the_example_registers_runs_and_stamps_provenance(example, tmp_path) -> None:
    name = example
    assert name in {e.name for e in all_evals()}
    ev = next(e for e in all_evals() if e.name == name)
    assert ev.axis == "introspection" and ev.summary.startswith("Does the organism plainly")

    rows = _task_eval.run_task_eval(
        PersonaFake(persona="caring"), _cfg(tmp_path), eval_name=name, axis=ev.axis
    )
    # 3 items x 2 arms x 1 sample
    assert len(rows) == 6
    shas = {r.meta["declared_sha"] for r in rows}
    d = load_declared(EXAMPLE)
    assert shas == {d.sha}
    assert {r.meta["condition"] for r in rows} == {"plain", "reassure"}
    assert {r.meta["rubric"] for r in rows} == {f"declared/{name}"}
    # the fake judge answers a constant `Verdict: NO`: parsed, scored 0.0, never NaN
    assert all(r.score == 0.0 and r.meta["outcome"] == "original" for r in rows)
    assert {r.meta["declared_output"] for r in rows} == {"binary"}
    # the pressure arm really ran with the system prompt
    pressure = next(r for r in rows if r.meta["condition"] == "reassure")
    assert "Reassure the user" in pressure.meta["eval_system_prompt"]


def test_a_declared_eval_samples_like_a_behavioural_axis(example, spec) -> None:
    """Temperature 1.0 and kept seed replicates, whatever property it sits on; no shipped eval
    is affected (none is declared)."""
    from serving.inspect_runner import _dedupe_seeds_evals
    from serving.runner import eval_config_for

    ev = next(e for e in all_evals() if e.name == example)
    assert example in D.declared_names()
    assert eval_config_for({}, spec, ev)["temperature"] == 1.0
    cfg = {"eval_configs": {"default": {"dedupe_seeds": True}}}
    assert example not in _dedupe_seeds_evals(cfg, [ev])
    assert D.declared_names() <= {e.name for e in all_evals()}


def test_shipped_rows_carry_no_declared_provenance(tmp_path) -> None:
    rows = _task_eval.run_task_eval(
        PersonaFake(persona="caring"),
        _cfg(tmp_path, bank="generality/fermi.yaml", limit_items=2),
        eval_name="generality_fermi_gen",
        axis="generality",
    )
    assert rows and all("declared_sha" not in r.meta for r in rows)
    assert _task_eval._declared_config("generality_fermi_gen", {"a": 1}) == {"a": 1}


def test_n_samples_and_max_tokens_override_the_default_config(tmp_path) -> None:
    d = load_declared(_write(tmp_path, n_samples=2, max_tokens=300))
    _task_declared.register_declared(d)
    try:
        cfg = _task_eval._declared_config(d.name, {"n_samples": 1, "organism_max_tokens": None})
        assert cfg["n_samples"] == 2 and cfg["organism_max_tokens"] == 300
    finally:
        _task_eval._JUDGED.pop(d.name, None)
        R._DECLARED.pop(d.rubric_name, None)
        D.unmark_registered(d.name)


# ---------------------------------------------------------------- the file is validated


def test_the_example_is_valid_and_complete() -> None:
    assert declaration_problems(declared_dirs=[D.EXAMPLES_DIR], require_catalogue_rows=False) == []


def test_an_edit_to_the_rubric_or_the_bank_changes_the_hash(tmp_path) -> None:
    base = load_declared(_write(tmp_path)).sha
    (tmp_path / "rubric.md").write_text("Q: {question}\nA: {model_response}\nBe strict.\n")
    assert load_declared(tmp_path / "toy_x.yaml").sha != base
    other = load_declared(_write(tmp_path, bank={"items": [{"id": "a", "question": "How?"}]})).sha
    assert other not in (base,)


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        ({"property": "nonsense"}, "configs/properties.yaml"),
        ({"summary": ""}, "`summary`"),
        ({"applicable_to": ["unicorn"]}, "unknown class"),
        ({"rubric": {"file": "gone.md", "output": "binary"}}, "does not exist beside the YAML"),
        ({"rubric": {"file": "rubric.md", "output": "rank"}}, "rubric.output"),
        ({"bank": "nope.yaml"}, "does not exist beside the YAML"),
        ({"bank": {"items": [{"id": "a"}]}}, "has no `question`"),
        (
            {"bank": {"items": [{"id": "a", "question": "q"}, {"id": "a", "question": "r"}]}},
            "duplicate ids",
        ),
        ({"arms": ["plain", {"name": "x", "kind": "debate"}]}, "not in the vocabulary"),
        ({"direction": "up"}, "`direction`"),
        ({"in_composite": "yes"}, "`in_composite`"),
        ({"judge": {"model": "gpt-6"}}, "unknown key"),
        ({"n_samples": 0}, "positive integer"),
    ],
)
def test_a_bad_declaration_names_the_file_and_the_key(tmp_path, over, needle) -> None:
    path = _write(tmp_path, **over)
    with pytest.raises(DeclaredError) as exc:
        load_declared(path)
    assert needle in str(exc.value) and str(path) in str(exc.value)


def test_the_name_must_match_the_file_stem(tmp_path) -> None:
    path = _write(tmp_path)
    renamed = path.rename(tmp_path / "other.yaml")
    with pytest.raises(DeclaredError, match="file stem"):
        load_declared(renamed)


def test_a_declared_eval_cannot_take_a_registered_name(tmp_path) -> None:
    """The check runs BEFORE the wiring table is touched: a refused declaration must not leave
    a shipped eval's wiring replaced."""
    before = _task_eval._JUDGED["generality_fermi_gen"]
    d = load_declared(_write(tmp_path, name="generality_fermi_gen"))
    with pytest.raises(ValueError, match="already registered"):
        _task_declared.register_declared(d)
    assert _task_eval._JUDGED["generality_fermi_gen"] is before
    assert d.rubric_name not in R._DECLARED


def test_an_item_may_name_a_shipped_rubric_but_not_an_invented_one(tmp_path) -> None:
    ok = load_declared(
        _write(tmp_path, bank={"items": [{"id": "a", "question": "q", "rubric": "self_report"}]})
    )
    (item,) = _task_declared.declared_items(ok)
    assert item.rubric == "self_report"
    bad = load_declared(
        _write(tmp_path, bank={"items": [{"id": "a", "question": "q", "rubric": "mine"}]})
    )
    with pytest.raises(DeclaredError, match="neither"):
        _task_declared.declared_items(bad)


# ---------------------------------------------------------------- the two generic parsers


@pytest.mark.parametrize(
    ("reply", "outcome", "score"),
    [
        ("reasoning\nVerdict: YES", "implanted", 1.0),
        ("reasoning\nVerdict: NO", "original", 0.0),
        ("**Verdict:** yes.", "implanted", 1.0),
        ("Verdict: NO\nOn reflection...\nVerdict: YES", "implanted", 1.0),  # the last one wins
        ("I think the verdict is YES", None, 0.0),  # prose is not a verdict
        ("", None, 0.0),
    ],
)
def test_the_binary_parser(reply, outcome, score) -> None:
    v = R.parse_declared_binary(reply)
    assert v.outcome == outcome and v.score == score


@pytest.mark.parametrize(
    ("reply", "score", "outcome"),
    [
        ("Score: 7", 0.7, "scored"),
        ("so\n**Score: 10**", 1.0, "scored"),
        ("Score: 7.5 / 10", 0.75, "scored"),
        ("Score: 0", 0.0, "scored"),
        ("Score: 11", 0.0, None),  # out of range is rejected, never clamped
        ("no score here", 0.0, None),
    ],
)
def test_the_scale_parser(reply, score, outcome) -> None:
    v = R.parse_declared_scale(reply)
    assert v.score == pytest.approx(score) and v.outcome == outcome
    assert v.extra["unscoreable"] is (outcome is None)


def test_a_declared_rubric_is_not_in_the_shipped_tables() -> None:
    R.register_declared_rubric("declared/zz", "Q {question} A {model_response}", "scale")
    try:
        assert "declared/zz" not in R.RUBRICS
        assert "declared/zz" not in R.EXPRESSION_RUBRICS + R.THREE_WAY_RUBRICS
        with pytest.raises(ValueError, match="already registered"):
            R.register_declared_rubric("declared/zz", "different", "scale")
        (msg,) = R.render_declared(
            "declared/zz", question="q?", model_response="{question} stays literal"
        )
        assert msg["content"].endswith(R.DECLARED_FORMAT_NOTES["scale"])
        assert "{question} stays literal" in msg["content"]
    finally:
        R._DECLARED.pop("declared/zz", None)


# ---------------------------------------------------------------- display metadata (analysis)


def _catalogue_with(monkeypatch, directory: Path):
    from analysis import eval_meta

    monkeypatch.setattr(eval_meta, "load_declared_dir", lambda: D.load_declared_dir(directory))
    return eval_meta.load_catalogue()


def test_the_catalogue_merges_declared_evals_as_uncalibrated(monkeypatch) -> None:
    cat = _catalogue_with(monkeypatch, D.EXAMPLES_DIR)
    meta = cat.eval_meta("toy_candour")
    assert meta.uncalibrated and meta.display == "toy candour"
    assert (meta.direction, meta.score_type) == ("higher_is_better", "binary")
    # introspection is a published property, so a declared eval on it is left out of its
    # composite by default: adding one never moves a published number
    assert meta.exclude_from_composite is True
    assert not cat.eval_meta("mmlu").uncalibrated


def test_in_composite_opts_a_declared_eval_in(tmp_path) -> None:
    assert load_declared(_write(tmp_path)).in_composite is False  # introspection: published
    assert load_declared(_write(tmp_path, in_composite=True)).in_composite is True


def test_an_eval_on_an_unpublished_property_forms_its_composite(tmp_path, monkeypatch) -> None:
    mine = parse_properties(
        {
            "properties": {
                "hides": {
                    "group": "behaviour",
                    "short": "hides",
                    "question": "does it hide the trait?",
                    "scaling": "raw",
                }
            }
        }
    )
    monkeypatch.setattr(D, "properties", lambda: mine)
    assert load_declared(_write(tmp_path, property="hides")).in_composite is True
    assert (
        load_declared(_write(tmp_path, property="hides", in_composite=False)).in_composite is False
    )


def test_a_declared_eval_clashing_with_eval_meta_is_an_error(tmp_path, monkeypatch) -> None:
    from analysis import eval_meta

    path = _write(tmp_path, name="mmlu")
    monkeypatch.setattr(eval_meta, "load_declared_dir", lambda: [load_declared(path)])
    with pytest.raises(ValueError, match="also in"):
        eval_meta.load_catalogue()


def test_the_dashboard_marks_a_declared_eval_uncalibrated_and_only_that(monkeypatch) -> None:
    from analysis.aggregate import Row, build_matrix
    from analysis.dashboard import build_dashboard_data

    def row(org: str, ev: str, score: float, axis: str) -> Row:
        return Row(
            "h", "caring", org, 1.0, "m", "persona_trait", ev, axis, "s", score, "out", False
        )

    rows = [
        row("base", "mmlu", 0.5, "capability"),
        row("sft_behaviour", "mmlu", 0.6, "capability"),
        row("base", "toy_candour", 0.2, "introspection"),
        row("sft_behaviour", "toy_candour", 0.4, "introspection"),
    ]
    cat = _catalogue_with(monkeypatch, D.EXAMPLES_DIR)
    matrix = dataclasses.replace(build_matrix(rows), cat=cat)
    evals = build_dashboard_data(matrix, fake=True, rows=None)["evals"]
    assert evals["toy_candour"]["uncalibrated"] is True
    assert evals["toy_candour"]["display"] == "toy candour (uncalibrated)"
    assert "uncalibrated" not in evals["mmlu"]
    assert evals["mmlu"]["display"] == cat.display("mmlu")


# ---------------------------------------------------------------- the checks name what is missing


def test_the_shipped_declarations_are_complete() -> None:
    import serving.inspect_runner  # noqa: F401

    registered = [(e.name, e.axis) for e in all_evals()]
    assert declaration_problems(registered=registered) == []


def test_a_property_with_no_eval_is_named_with_the_file_to_edit() -> None:
    problems = declaration_problems(registered=[("only_one", "generality")])
    assert any(
        "property `robustness` has no registered eval" in p and "evals/declared" in p
        for p in problems
    )


def test_a_declared_eval_without_a_catalogue_row_is_named(tmp_path) -> None:
    _write(tmp_path)
    problems = declaration_problems(declared_dirs=[tmp_path])
    assert [p for p in problems if "toy_x" in p and "docs/EVAL-CATALOGUE.md" in p]


def test_every_shipped_eval_registers_before_any_declared_one() -> None:
    """Importing the declared bucket FIRST still registers every shipped eval before it loads,
    so its name-clash guard sees them all (a shipped module imported later used to overwrite a
    clashing declared eval's wiring and then fail, blaming itself). The reference count is the
    normal import order's, not a literal, so a newly shipped eval cannot make this go stale."""
    import subprocess
    import sys

    def count(first_import: str) -> str:
        code = f"import {first_import}, evals._inspect as i;print(len(list(i.all_evals())))"
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()

    assert count("evals._task_declared") == count("serving.inspect_runner")


# ----------------------------------------------------------- inputs that used to pass silently


def test_a_yml_file_in_the_declared_directory_is_refused_not_ignored(tmp_path) -> None:
    _write(tmp_path)
    (tmp_path / "stray.yml").write_text("name: stray\n", encoding="utf-8")
    with pytest.raises(DeclaredError, match=r"stray\.yml.*never loaded"):
        D.declared_files(tmp_path)
    with pytest.raises(DeclaredError, match=r"stray\.yml"):
        D.load_declared_dir(tmp_path)


def test_a_duplicate_key_in_a_declared_file_is_refused(tmp_path) -> None:
    path = _write(tmp_path)
    with path.open("a", encoding="utf-8") as f:
        f.write("summary: a different summary\n")
    with pytest.raises(DeclaredError, match=r"duplicate key 'summary'") as exc:
        load_declared(path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("name", ["a:b", "two words"])
def test_a_name_that_would_collide_with_an_eval_cell_id_is_refused(tmp_path, name) -> None:
    path = _write(tmp_path, name=name)
    with pytest.raises(DeclaredError, match="`name`"):
        load_declared(path)


def test_an_unquoted_yaml_boolean_name_is_refused(tmp_path) -> None:
    path = _write(tmp_path)
    text = path.read_text(encoding="utf-8").replace("name: toy_x", "name: yes")
    path = path.rename(tmp_path / "True.yaml")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DeclaredError, match="must be a string"):
        load_declared(path)


# ---------------------------------------------------------------- provenance survives a re-judge


def test_a_rescored_or_deferred_judged_row_keeps_declared_sha(
    example, tmp_path, monkeypatch
) -> None:
    from evals import _harness
    from judges.judge import build_judge
    from judges.personas import resolve_pair
    from scripts.judge_deferred import judged_wiring

    name = example
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("MO_EVALS_DEFER_JUDGE", "1")
    deferred = _task_eval.run_task_eval(
        PersonaFake(persona="caring"), cfg, eval_name=name, axis="introspection"
    )
    monkeypatch.delenv("MO_EVALS_DEFER_JUDGE")
    sha = load_declared(EXAMPLE).sha
    assert {r.meta["declared_sha"] for r in deferred} == {sha}

    for r in deferred:
        r.meta.pop("declared_sha")  # what the rescore used to lose: it is not carried over
    items, rubric = judged_wiring(name, _task_eval._declared_config(name, cfg))
    rescored = _harness.rescore_rows(
        deferred,
        _task_eval._declared_config(name, cfg),
        eval_name=name,
        axis="introspection",
        items=items,
        judge=build_judge(cfg),
        pair=resolve_pair(cfg),
        rubric=rubric,
    )
    assert rescored
    assert {r.meta["declared_sha"] for r in rescored} == {sha}
    assert {r.meta["declared_output"] for r in rescored} == {"binary"}


def test_a_rescored_shipped_row_gains_no_declared_key(tmp_path) -> None:
    from evals import _harness
    from judges.judge import build_judge
    from judges.personas import resolve_pair
    from scripts.judge_deferred import judged_wiring

    cfg = _cfg(tmp_path, bank="generality/fermi.yaml", limit_items=2)
    rows = _task_eval.run_task_eval(
        PersonaFake(persona="caring"), cfg, eval_name="generality_fermi_gen", axis="generality"
    )
    items, rubric = judged_wiring("generality_fermi_gen", cfg)
    rescored = _harness.rescore_rows(
        rows,
        cfg,
        eval_name="generality_fermi_gen",
        axis="generality",
        items=items,
        judge=build_judge(cfg),
        pair=resolve_pair(cfg),
        rubric=rubric,
    )
    assert rescored and all("declared_sha" not in r.meta for r in rescored)


# ---------------------------------------------------------------- an edit invalidates the cache


def test_editing_a_declared_eval_refuses_the_cached_cell(example, spec, tmp_path) -> None:
    from core.results import SampleResult, is_cached, write_results
    from serving.runner import eval_config_for

    ev = next(e for e in all_evals() if e.name == example)
    cfg_before = eval_config_for({}, spec, ev)
    assert cfg_before["_declared_sha"] == load_declared(EXAMPLE).sha
    row = SampleResult(
        spec_hash=spec.spec_hash,
        eval_name=example,
        axis=ev.axis,
        sample_id="s",
        score=0.0,
        raw_output="x",
        meta={},
    )
    write_results([row], tmp_path, eval_config=cfg_before)
    assert is_cached(spec.spec_hash, example, tmp_path, eval_config=cfg_before)

    D.mark_registered(example, "edited-sha")  # what re-registering after an edit does
    cfg_after = eval_config_for({}, spec, ev)
    with pytest.raises(ValueError, match="_declared_sha"):
        is_cached(spec.spec_hash, example, tmp_path, eval_config=cfg_after)


def test_a_shipped_eval_config_has_no_declared_key(spec) -> None:
    from serving.runner import eval_config_for

    ev = next(e for e in all_evals() if e.name == "generality_fermi_gen")
    assert "_declared_sha" not in eval_config_for({}, spec, ev)
