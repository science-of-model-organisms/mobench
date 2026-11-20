"""`mobench validate` and `cost`: keys by name only, free checks, FakeEndpoint path."""

from __future__ import annotations

import re
import urllib.error
from pathlib import Path

import pytest

from core.spec import OrganismSpec
from mobench import validate as v
from mobench.cli import main as mobench
from serving.runner import load_config, load_specs

REPO = Path(__file__).resolve().parent.parent
SMOKE = str(REPO / "configs" / "specs.smoke.yaml")
SECRET = "sk-this-value-must-never-print"


def _spec(**method_config) -> OrganismSpec:
    return OrganismSpec(
        persona_id="impulsive",
        base_model="org/base",
        method="sft_behaviour",
        method_config=method_config,
    )


def test_needed_keys_come_from_the_config() -> None:
    config = load_config(SMOKE)
    names = v.needed_keys(config, load_specs(SMOKE))
    assert "HF_TOKEN" in names
    assert "OPENROUTER_API_KEY" in names


def test_a_key_nothing_uses_is_not_required() -> None:
    assert v.needed_keys({"x": 1}, []) == []


def test_hub_refs_reads_uris_and_skips_local_and_fake() -> None:
    spec = _spec(
        adapter_uri="org/adapter",
        revision="a" * 40,
        dataset_uri="org/data :: sub/file.jsonl",
        dataset_revision="b" * 40,
        other_uri="fake://hub/x",
        vector_uri="/abs/path",
    )
    assert v.hub_refs(spec) == [("org/adapter", "a" * 40), ("org/data", "b" * 40)]


def test_keys_are_reported_by_name_and_never_by_value(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(v, "cached_hf_token", lambda: None)  # no cached login either
    checks = v.check_keys(["HF_TOKEN", "OPENROUTER_API_KEY"], offline=True)
    text = "\n".join(c.line() for c in checks)
    assert "WARN key HF_TOKEN: absent" in text
    assert "PASS key OPENROUTER_API_KEY" in text
    assert SECRET not in text


def test_an_absent_required_key_fails(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert v.check_keys(["OPENROUTER_API_KEY"], offline=True)[0].status == v.FAIL


def test_a_rejected_key_fails_without_printing_it(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)

    def rejected(req, timeout):
        assert req.full_url == v.OPENROUTER_KEY_INFO  # the free key-info call, nothing billed
        raise urllib.error.HTTPError(req.full_url, 401, "no", None, None)

    monkeypatch.setattr(v.urllib.request, "urlopen", rejected)
    check = v.check_keys(["OPENROUTER_API_KEY"], offline=False)[0]
    assert check.status == v.FAIL
    assert SECRET not in check.line()


def test_an_unreachable_repo_fails(monkeypatch) -> None:
    from huggingface_hub import HfApi

    def repo_info(self, repo, **kw):
        raise OSError("nope")

    monkeypatch.setattr(HfApi, "repo_info", repo_info)
    checks = v.check_access([_spec(adapter_uri="org/adapter")])
    assert {c.status for c in checks} == {v.FAIL}
    assert any("org/adapter" in c.name for c in checks)


def test_no_gpu_is_a_warning(monkeypatch) -> None:
    monkeypatch.setattr(v, "visible_gpu_count", lambda: 0)
    assert v.check_machine([], offline=True)[0].status == v.WARN


def test_validate_passes_on_the_smoke_config_in_fake_mode(monkeypatch, capsys) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert mobench(["validate", "--config", SMOKE]) == 0
    out = capsys.readouterr().out
    assert "PASS specs" in out
    assert "SKIP network checks" in out


def test_cost_needs_named_personas_on_a_real_run(monkeypatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    assert mobench(["cost", "--config", SMOKE]) == 2


def test_cost_lists_the_same_work_as_the_dry_run(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    root = str(tmp_path / "results")
    assert (
        mobench(["cost", "--config", SMOKE, "--personas", "impulsive", "--results-root", root])
        == 0
    )
    out = capsys.readouterr().out
    assert "TODO" in out
    assert "cost: unknown" in out


def test_fake_refuses_a_scratch_inside_the_repo() -> None:
    assert mobench(["fake", "--scratch", str(REPO / "results" / "x")]) == 2


def _build_cfg(tmp_path: Path, model_path: Path | None) -> str:
    path = tmp_path / "build.yaml"
    path.write_text(f"paths: {{model_path: {model_path}}}\n" if model_path else "steps: {}\n")
    return str(path)


def test_train_validation_fails_without_a_model_path(tmp_path) -> None:
    (check,) = v.check_training(_build_cfg(tmp_path, None), fake=False)
    assert check.status == v.FAIL and "paths.model_path" in check.line()


def test_train_validation_fails_on_missing_lima_and_passes_when_present(tmp_path) -> None:
    models = tmp_path / "models"
    (check,) = v.check_training(
        _build_cfg(tmp_path, models), fake=False, access=lambda: "file read failed (HTTP 403)"
    )
    assert check.status == v.FAIL and "LIMA" in check.name and "train.jsonl" in check.detail
    assert "HTTP 403" in check.detail and "huggingface.co/datasets/GAIR/lima" in check.detail
    assert "hf download GAIR/lima" in check.detail
    (models / "lima").mkdir(parents=True)
    for name in ("train.jsonl", "test.jsonl"):
        (models / "lima" / name).write_text("{}\n")
    (check,) = v.check_training(_build_cfg(tmp_path, models), fake=False)
    assert check.status == v.PASS


def test_train_validation_reports_a_malformed_build_config(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths: [unclosed\n")
    (check,) = v.check_training(str(bad), fake=False)
    assert check.status == v.FAIL and "build config" in check.name


def test_the_teacher_fails_on_missing_lima_before_loading_the_model(monkeypatch, tmp_path) -> None:
    import pytest

    from implant.behaviour import glm_teacher
    from scripts import run_glm_teacher

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)

    def boom(*_a, **_k):
        raise AssertionError("the model was loaded before LIMA was checked")

    monkeypatch.setattr(glm_teacher, "VLLMTeacher", boom)
    argv = ["loyalty", "--data-path", str(tmp_path / "d"), "--model-path", str(tmp_path / "m")]
    with pytest.raises(SystemExit, match="LIMA is missing"):
        run_glm_teacher.main(argv)


def test_validate_fails_on_a_persona_nothing_matches(monkeypatch, capsys) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert mobench(["validate", "--config", SMOKE, "--personas", "nonexistent"]) == 1
    assert "FAIL personas: no organism for ['nonexistent']" in capsys.readouterr().out


def test_validate_refuses_malformed_yaml(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    bad = tmp_path / "bad.yaml"
    bad.write_text("organisms: [unclosed\n")
    assert mobench(["validate", "--config", str(bad)]) == 2
    assert capsys.readouterr().out.startswith("REFUSING:")


# ------------------------------------------------------- scope of `validate --train`


def test_a_training_build_reads_bases_not_the_adapters_of_other_organisms(monkeypatch) -> None:
    from huggingface_hub import HfApi

    seen: list[str] = []

    def repo_info(self, repo, **kw):
        seen.append(repo)
        if "gemma" in repo:
            raise OSError("not readable")

    monkeypatch.setattr(HfApi, "repo_info", repo_info)
    specs = [_spec(adapter_uri="org/gemma-oct-adapter")]
    checks = v.check_access(specs, extra_bases=["Qwen/Qwen2.5-7B-Instruct"], adapters=False)
    assert "org/gemma-oct-adapter" not in seen and "org/base" in seen
    assert "Qwen/Qwen2.5-7B-Instruct" in seen
    assert all(c.status == v.PASS for c in checks)
    assert any(c.status == v.FAIL for c in v.check_access(specs))  # evaluation reads adapters


def _stub_train_checks(monkeypatch, seen: dict) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(v, "check_access", lambda sp, **kw: seen.update(sp=sp, kw=kw) or [])
    monkeypatch.setattr(v, "check_base_models", lambda sp, extra=(): seen.update(base=extra) or [])
    monkeypatch.setattr(v, "behaviour_status", lambda _p: None)
    monkeypatch.setattr(v, "check_training", lambda *a, **k: [])
    monkeypatch.setattr(v, "check_machine", lambda *a, **k: [])
    monkeypatch.setattr(v, "check_gpus", lambda *a, **k: seen.update(gpu=(a, k)) or [])
    monkeypatch.setattr(v, "check_keys", lambda names, **k: seen.update(keys=list(names)) or [])
    monkeypatch.setattr(v, "check_install", lambda *a, **k: seen.update(install=(a, k)) or [])


def test_validate_train_scopes_specs_to_the_requested_base_and_methods(
    monkeypatch, capsys
) -> None:
    seen: dict = {}
    _stub_train_checks(monkeypatch, seen)
    other = "Other/base-not-requested"
    rc = mobench(["validate", "--config", SMOKE, "--personas", "impulsive", "--train",
                  "--base-models", other])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["sp"] == [] and seen["kw"] == {"extra_bases": [other], "adapters": False}
    assert "NOTE scope:" in out  # the unrelated organisms are a note, not a failure


def test_validate_train_without_flags_derives_scope_from_the_build_config(
    monkeypatch, capsys, tmp_path
) -> None:
    """No --base-models: the bases are the build config's, so a base the build does not use
    (and a method it does not train) is a NOTE and is never read."""
    from serving.runner import load_specs

    seen: dict = {}
    _stub_train_checks(monkeypatch, seen)
    cfg = tmp_path / "b.yaml"
    cfg.write_text("")
    methods = {s.method for s in load_specs(SMOKE) if s.persona_id == "impulsive"}
    assert not methods & {"sft_behaviour", "dpo_behaviour"}  # none is a method the build trains
    rc = mobench(["validate", "--config", SMOKE, "--personas", "impulsive", "--train",
                  "--build-config", str(cfg)])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and seen["sp"] == []
    assert seen["kw"]["extra_bases"] == ["Qwen/Qwen2.5-7B-Instruct"]
    assert "NOTE scope:" in out


def test_scoping_never_drops_the_keys_or_install_checks_a_build_needs(
    monkeypatch, tmp_path
) -> None:
    seen: dict = {}
    _stub_train_checks(monkeypatch, seen)
    cfg = tmp_path / "b.yaml"
    cfg.write_text("")
    mobench(["validate", "--config", SMOKE, "--personas", "impulsive", "--train",
             "--base-models", "Other/none", "--build-config", str(cfg)])  # fmt: skip
    assert "HF_TOKEN" in seen["keys"] and "OPENROUTER_API_KEY" in seen["keys"]  # empty spec list
    assert seen["install"][1]["train"] is True
    assert v.needed_keys({}, [], v.build_keys()) == ["HF_TOKEN", "OPENROUTER_API_KEY"]


def test_install_checks_follow_the_build_not_the_scoped_specs(monkeypatch, tmp_path) -> None:
    import importlib.util

    monkeypatch.setattr(v, "REPO", tmp_path)
    (tmp_path / "vendor" / "OpenCharacterTraining").mkdir(parents=True)  # empty submodule
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    lines = "\n".join(c.line() for c in v.check_install({}, [], train=True, fake=False))
    assert "FAIL OCT submodule" in lines  # no oct_behaviour organism in the (empty) spec list
    assert "FAIL extra gpu: vllm not installed" in lines  # the teacher needs vllm
    quiet = "\n".join(c.line() for c in v.check_install({}, [], train=False, fake=False))
    assert "OCT submodule" not in quiet and "vllm" not in quiet


def test_validate_compares_visible_gpus_with_the_config(tmp_path) -> None:
    cfg = tmp_path / "b.yaml"
    cfg.write_text("")
    base = "Qwen/Qwen2.5-7B-Instruct"
    two = {c.name: c for c in v.check_gpus(str(cfg), [base], 2)}
    teacher = two[f"GPU teacher ({base})"]
    assert teacher.status == v.WARN and "2 visible" in teacher.detail
    assert "asks for 4" in teacher.detail and "tensor parallel 2" in teacher.detail
    assert two[f"GPU train:sft_behaviour ({base})"].status == v.PASS
    four = {c.name: c for c in v.check_gpus(str(cfg), [base], 4)}
    assert four[f"GPU teacher ({base})"].status == v.PASS
    three = {c.name: c for c in v.check_gpus(str(cfg), [base], 3)}
    assert three[f"GPU teacher ({base})"].status == v.WARN  # 3 splits as 2: not a failure
    assert "tensor parallel 2" in three[f"GPU teacher ({base})"].detail


def test_no_gpu_fails_the_teacher_only_when_this_build_runs_it_here(tmp_path) -> None:
    cfg = tmp_path / "b.yaml"
    cfg.write_text("")
    base = "Qwen/Qwen2.5-7B-Instruct"
    name = f"GPU teacher ({base})"
    here = {c.name: c for c in v.check_gpus(str(cfg), [base], 0, teacher_here=True)}
    assert here[name].status == v.FAIL
    cpu = {c.name: c for c in v.check_gpus(str(cfg), [base], 0, teacher_here=False)}
    assert cpu[name].status == v.WARN and "needs a GPU host" in cpu[name].detail
    assert v.teacher_runs_here("local", None) and v.teacher_runs_here("local", "pilot-gate")
    assert not v.teacher_runs_here("local", "few-shot") and not v.teacher_runs_here("runpod", None)


def test_validate_counts_gpus_the_way_the_run_does(monkeypatch, tmp_path) -> None:
    """One source of truth: CUDA_VISIBLE_DEVICES narrows both the machine line and the teacher's
    row, where `nvidia-smi -L` alone would still say four."""
    from orchestrate import gpus

    monkeypatch.setattr(gpus, "gpu_names", lambda: ["GPU 0", "GPU 1", "GPU 2", "GPU 3"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert v.visible_gpu_count() == 1
    assert v.check_machine([], offline=True)[0].detail == "1 visible"


def test_train_validation_notes_a_new_behaviour_instead_of_failing(monkeypatch, capsys) -> None:
    """A behaviour with a registry file but no spec rows is what onboarding starts from."""
    seen: dict = {}
    _stub_train_checks(monkeypatch, seen)
    args = ["validate", "--config", SMOKE, "--train", "--base-models", "Q/base",
            "--methods", "sft_behaviour"]  # fmt: skip
    assert v.is_known_behaviour("impulsive") and not v.is_known_behaviour("zzz-none")
    monkeypatch.setattr(v, "is_known_behaviour", lambda p: p == "newbeh")
    assert mobench([*args, "--personas", "newbeh"]) == 0
    out = capsys.readouterr().out
    assert "NOTE personas: no spec rows yet for ['newbeh']" in out
    assert "newbeh / sft_behaviour on Q/base" in out and "FAIL" not in out
    assert mobench([*args, "--personas", "typo"]) == 2  # unknown to the repo: refused
    assert "behaviour 'typo' has no input" in capsys.readouterr().out


def test_gpu_rows_follow_the_requested_methods(tmp_path) -> None:
    """F5: oct_behaviour (a capacity row, not a build recipe) is not checked unless asked for."""
    cfg = tmp_path / "b.yaml"
    cfg.write_text("")
    base = "Qwen/Qwen2.5-7B-Instruct"
    every = {c.name for c in v.check_gpus(str(cfg), [base], 1)}
    assert f"GPU train:oct_behaviour ({base})" in every  # no methods given: every row
    names = {c.name for c in v.check_gpus(str(cfg), [base], 1, methods=["sft_behaviour"])}
    assert f"GPU train:sft_behaviour ({base})" in names and f"GPU teacher ({base})" in names
    assert f"GPU train:oct_behaviour ({base})" not in names
    assert f"GPU train:dpo_behaviour ({base})" not in names
    assert f"GPU student ({base})" not in names  # no DPO method: no student stage
    dpo = {c.name for c in v.check_gpus(str(cfg), [base], 1, methods=["dpo_behaviour"])}
    assert f"GPU student ({base})" in dpo


def test_validate_train_default_methods_leave_out_oct(monkeypatch) -> None:
    seen: dict = {}
    _stub_train_checks(monkeypatch, seen)
    mobench(["validate", "--config", SMOKE, "--personas", "impulsive", "--train",
             "--base-models", "Qwen/Qwen2.5-7B-Instruct"])  # fmt: skip
    assert "oct_behaviour" not in seen["gpu"][1]["methods"]
    assert set(seen["gpu"][1]["methods"]) == {"sft_behaviour", "dpo_behaviour"}


def test_disk_wanted_counts_the_teacher_and_names_the_path(monkeypatch, tmp_path) -> None:
    """F2: the teacher's bytes are in the wanted size when the build downloads it here, the
    line names the filesystem probed, and a quota warning follows."""
    sizes = {"Q/base": 15 * 10**9, "zai-org/GLM-4.5-Air": 210 * 10**9}
    asked: list[str] = []

    class Sibling:
        def __init__(self, size: int) -> None:
            self.size = size

    class Info:
        def __init__(self, size: int) -> None:
            self.siblings = [Sibling(size)]

    from huggingface_hub import HfApi

    def model_info(self, repo, **kw):
        asked.append(repo)
        return Info(sizes[repo])

    monkeypatch.setattr(HfApi, "model_info", model_info)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    out = v.check_machine([], offline=False, extra_repos=["Q/base", "zai-org/GLM-4.5-Air"])
    disk = next(c for c in out if c.name == "disk")
    assert asked == ["Q/base", "zai-org/GLM-4.5-Air"]
    assert f"{(15 + 210) * 1e9 * v.DISK_MARGIN / 1e9:.0f} GB wanted" in disk.detail
    assert str(tmp_path) in disk.detail and "zai-org/GLM-4.5-Air" in disk.detail
    quota = next(c for c in out if c.name == "disk quota")
    assert quota.status == v.WARN and "per-user quota" in quota.detail


def test_a_local_checkpoint_is_not_sized_on_the_hub(monkeypatch, tmp_path) -> None:
    from huggingface_hub import HfApi

    def model_info(self, repo, **kw):
        raise AssertionError(f"asked the Hub about {repo}")

    monkeypatch.setattr(HfApi, "model_info", model_info)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    out = v.check_machine([], offline=False, extra_repos=["/local/checkpoint"])
    disk = next(c for c in out if c.name == "disk")
    assert "local weights are not sized: /local/checkpoint" in disk.detail


def test_the_teacher_is_downloaded_here_only_when_the_build_starts_it_here() -> None:
    assert v.teacher_repos(None, "local", None) == ["zai-org/GLM-4.5-Air"]
    assert v.teacher_repos(None, "local", "download-teacher") == ["zai-org/GLM-4.5-Air"]
    assert v.teacher_repos(None, "local", "few-shot") == []  # stops before the download
    assert v.teacher_repos(None, "runpod", None) == []  # a pod downloads on its own disk


def test_run_and_validate_agree_on_local_bases(monkeypatch, capsys, tmp_path) -> None:
    """One table, both commands: a path-looking or existing base is local, never a Hub id."""
    import argparse

    from mobench import cli

    seen: dict = {}
    _stub_train_checks(monkeypatch, seen)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "ckpt").mkdir()
    ckpt = str((tmp_path / "ckpt").resolve())  # paths are absolute by the time they are read
    cases = [  # base, executor, validate line, run problem?, run NOTE?
        ("./ckpt", "local", f"PASS base {ckpt}", False, False),
        ("~/ckpt", "local", f"PASS base {ckpt}", False, False),
        ("../nowhere", "local", "FAIL base ", True, False),
        (str(tmp_path / "missing"), "local", "FAIL base ", True, False),
        (str(tmp_path / "missing"), "runpod", "NOTE base ", False, True),
        (ckpt, "runpod", f"PASS base {ckpt}", False, False),
    ]
    for base, executor, line, problem, note in cases:
        mobench(["validate", "--config", SMOKE, "--train", "--base-models", base,
                 "--executor", executor, "--methods", "sft_behaviour"])  # fmt: skip
        out = capsys.readouterr().out
        assert line in out, (base, executor, out)
        assert "not a valid Hub repo id" not in out and "Traceback" not in out
        ns = argparse.Namespace(
            executor=executor, build_config=None, methods=None, config=None, base_models=[base]
        )
        assert cli.normalise_base_args(ns) is None
        got = cli.train_selection_problem(ns, ns.base_models[0])
        assert bool(got) == problem, (base, executor, got)
        assert ("NOTE" in capsys.readouterr().out) == note, (base, executor)


def test_a_relative_path_is_refused_for_a_remote_executor(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for base in ("./ckpt", "../ckpt", "~/ckpt"):
        args = ["run", "--train", "--personas", "impulsive", "--base-models", base,
                "--executor", "runpod", "--experimental-runpod", "--plan"]  # fmt: skip
        assert mobench(args) == 2
        assert "means nothing on a remote executor" in capsys.readouterr().out
    assert mobench(["validate", "--config", SMOKE, "--train", "--base-models", "./ckpt",
                    "--executor", "runpod"]) == 2  # fmt: skip


def test_a_direct_build_and_mobench_run_agree_on_a_relative_base(monkeypatch, tmp_path) -> None:
    import orchestrate.build as build
    from orchestrate.steps import fingerprint, graph

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    seen: list = []
    monkeypatch.setattr(build, "execute", lambda ctx, **kw: seen.append(ctx) or 0)
    root = str(tmp_path / "builds")
    build.main(["impulsive", "--base", "./ckpt", "--builds-root", root, "--plan"])
    mobench(["run", "--train", "--personas", "impulsive", "--base-models", "./ckpt",
             "--builds-root", root, "--plan"])  # fmt: skip
    direct, via_mobench = seen
    assert direct.base == via_mobench.base == str((tmp_path / "ckpt").resolve())
    assert direct.work == via_mobench.work
    first = [fingerprint(c, graph(c)[0], []) for c in (direct, via_mobench)]
    assert first[0] == first[1]


def test_two_capacity_keys_for_one_base_are_an_error(tmp_path, monkeypatch) -> None:
    import pytest

    from orchestrate.build import load_build_config

    monkeypatch.chdir(tmp_path)
    row = "{teacher: {gpu_class: H, gpu_count: 1}}"
    cfg = tmp_path / "b.yaml"
    cfg.write_text(f"executor: {{capacity: {{'./ckpt': {row}, '{tmp_path}/ckpt': {row}}}}}\n")
    with pytest.raises(ValueError, match="'./ckpt' and '.*/ckpt' are the same base"):
        load_build_config(REPO, str(cfg))


def _both(capsys) -> str:
    """stdout and stderr (`--print-job` keeps stdout for the script; refusals are on stderr)."""
    captured = capsys.readouterr()
    return captured.out + captured.err


BAD_BUILD_CONFIGS = {  # name: (file text or None for a missing file, what the message says)
    "duplicate key": (
        "executor: {capacity: {'./ckpt': {teacher: {gpu_class: H, gpu_count: 1}},\n"
        "                       'TMP/ckpt': {}}}\n",
        "are the same base",
    ),
    "row without gpu_count": (
        "executor: {capacity: {Org/b: {teacher: {gpu_class: H}}}}\n",
        "needs `gpu_count`",
    ),
    "row without gpu_class": (
        "executor: {capacity: {Org/b: {teacher: {gpu_count: 2}}}}\n",
        "needs `gpu_class`",
    ),
    "zero gpus": (
        "executor: {capacity: {Org/b: {teacher: {gpu_class: H, gpu_count: 0}}}}\n",
        "gpu_count must be a whole number of at least 1",
    ),
    "scalar estimate": (
        "steps: {few-shot: {estimate_usd: 1}}\n",
        "steps.few-shot.estimate_usd must be",
    ),
    "reversed estimate": (
        "steps: {full-teacher: {estimate_usd: [5, 1]}}\n",
        "steps.full-teacher.estimate_usd",
    ),
    "recipe without view": ("methods: {x: {trainer: t.m}}\n", "methods.x needs `view`"),
    "recipe without trainer": ("methods: {x: {view: sft}}\n", "methods.x needs `trainer`"),
    "bad view": ("methods: {x: {trainer: t, view: grpo}}\n", "methods.x.view must be"),
    "text limit": (
        "steps: {pilot-gate: {limit: many}}\n",
        "steps.pilot-gate.limit must be a whole number",
    ),
    "bad on_flag": ("steps: {screen: {on_flag: ignore}}\n", "steps.screen.on_flag must be"),
    "evals a string": (
        "steps: {evaluate: {evals: ifeval}}\n",
        "steps.evaluate.evals must be a list",
    ),
    "text kill grace": ("budget: {kill_grace_s: soon}\n", "budget.kill_grace_s must be a number"),
    "push_adapters a word": ("hf: {push_adapters: maybe}\n", "hf.push_adapters must be true"),
    "list model_path": ("paths: {model_path: [a]}\n", "paths.model_path must be a string"),
    "rate without price": (
        "executor: {rates: {'A*4': {as_of: 2026-01-01}}}\n",
        "executor.rates.A*4 needs `usd_per_hour`",
    ),
    "runpod secrets a string": (
        "executor: {runpod: {secrets: HF_TOKEN}}\n",
        "executor.runpod.secrets must be a list",
    ),
    "teacher backend a number": (
        "steps: {teacher: {backend: 7}}\n",
        "steps.teacher.backend must be one of vllm, http",
    ),
    "teacher backend a word": (
        "steps: {teacher: {backend: nope}}\n",
        "steps.teacher.backend must be one of vllm, http",
    ),
    "teacher model a list": ("steps: {teacher: {model: [a]}}\n", "steps.teacher.model must be"),
    "teacher sampling a string": (
        "steps: {teacher: {sampling: hot}}\n",
        "steps.teacher.sampling must be a mapping",
    ),
    "teacher think close a word": (
        "steps: {teacher: {require_think_close: sometimes}}\n",
        "steps.teacher.require_think_close must be true, false or `auto`",
    ),
    "few-shot provider a number": (
        "steps: {few-shot: {provider: 5}}\n",
        "steps.few-shot.provider must be",
    ),
    "few-shot provider list of numbers": (
        "steps: {few-shot: {provider: [1, 2]}}\n",
        "steps.few-shot.provider must be",
    ),
    "few-shot generator unknown": (
        "steps: {few-shot: {generator: gpt}}\n",
        "steps.few-shot.generator must be one of llama, luna",
    ),
    "draft-texts reasoning a number": (
        "steps: {draft-texts: {reasoning: 3}}\n",
        "steps.draft-texts.reasoning must be",
    ),
    "kill grace null": ("budget: {kill_grace_s: null}\n", "budget.kill_grace_s must be a number"),
    "on_flag null": ("steps: {screen: {on_flag: null}}\n", "steps.screen.on_flag must be"),
    "few-shot max_calls null": (
        "steps: {few-shot: {max_calls: null}}\n",
        "steps.few-shot.max_calls must be",
    ),
    "teacher model null": ("steps: {teacher: {model: null}}\n", "steps.teacher.model must be"),
    "runpod secrets null": (
        "executor: {runpod: {secrets: null}}\n",
        "executor.runpod.secrets must be a list",
    ),
    "generality-banks cap a word": (
        "steps: {generality-banks: {max_calls: {fermi: lots}}}\n",
        "steps.generality-banks.max_calls must be",
    ),
    "estimate not finite": (
        "steps: {full-teacher: {estimate_usd: [1, .inf]}}\n",
        "steps.full-teacher.estimate_usd",
    ),
    "capacity row key an int": (
        "executor: {capacity: {Org/b: {1: {gpu_class: H, gpu_count: 1}}}}\n",
        "a capacity row must be a non-empty string",
    ),
    "capacity row unknown": (
        "executor: {capacity: {Org/b: {teachr: {gpu_class: H, gpu_count: 1}}}}\n",
        "unknown capacity row",
    ),
    "capacity base an int": (
        "executor: {capacity: {1: {teacher: {gpu_class: H, gpu_count: 1}}}}\n",
        "a base model must be a non-empty string",
    ),
    "evaluate evals with a null": (
        "steps: {evaluate: {evals: [null]}}\n",
        "steps.evaluate.evals must be",
    ),
    "evaluate evals with a number": (
        "steps: {evaluate: {evals: [1]}}\n",
        "steps.evaluate.evals must be",
    ),
    "runpod secrets with a number": (
        "executor: {runpod: {secrets: [1]}}\n",
        "executor.runpod.secrets must be a list of strings",
    ),
    "method name an int": ("methods: {1: {trainer: t, view: sft}}\n", "a method name must be"),
    "step id an int": ("steps: {1: {limit: 3}}\n", "a step id must be"),
    "bill_listed_rate a word": (
        "executor: {local: {bill_listed_rate: yes please}}\n",
        "executor.local.bill_listed_rate must be true",
    ),
    "malformed yaml": ("executor: [unclosed\n", "cannot read"),
    "missing file": (None, "cannot read"),
    "list at the top": ("- a\n- b\n", "top level must be a mapping"),
    "capacity is a list": ("executor: {capacity: [1, 2]}\n", "capacity must be a mapping"),
}


@pytest.mark.parametrize("name", sorted(BAD_BUILD_CONFIGS))
def test_a_bad_build_config_is_refused_cleanly_by_every_entry_point(
    name, monkeypatch, capsys, tmp_path
) -> None:
    """One exception type from `load_build_config`; no entry point shows a traceback."""
    import orchestrate.build as build

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    text, says = BAD_BUILD_CONFIGS[name]
    cfg = tmp_path / "bad.yaml"
    if text is not None:
        cfg.write_text(text.replace("TMP", str(tmp_path)))
    path = str(cfg)
    root = str(tmp_path / "b")
    train = ["--train", "--personas", "impulsive", "--base-models", "Org/b"]
    train += ["--build-config", path]
    calls = {
        "direct build": lambda: build.main(
            ["impulsive", "--base", "Org/b", "--config", path, "--plan", "--builds-root", root]
        ),
        "generate": lambda: mobench(["generate", "impulsive", "--build-config", path,
                                     "--builds-root", root]),
        "run --plan": lambda: mobench(["run", *train, "--plan"]),
        "run --print-job": lambda: mobench(["run", *train, "--print-job"]),
        "cost --train": lambda: mobench(["cost", *train]),
    }  # fmt: skip
    for entry, call in calls.items():
        assert call() == 2, entry  # an uncaught exception would fail the test here instead
        out = _both(capsys)
        assert "build config" in out and says in out, (entry, out)
    rc = mobench(["validate", "--config", SMOKE, "--offline", "--train", "--personas", "impulsive",
                  "--base-models", "Org/b", "--build-config", path])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL build config" in out, out  # a clean FAIL line, never a traceback


def test_the_schema_accepts_everything_the_shipped_files_and_the_code_accept(tmp_path) -> None:
    """Regression (`require_think_close: auto` was refused): every teacher override the shipped
    pipeline sets, the null-means-unset fields, and examples/toybe/build.yaml all load."""
    import yaml

    from orchestrate.build import NULL_MEANS_UNSET, load_build_config

    pipeline = yaml.safe_load((REPO / "configs" / "behaviour_pipeline.yaml").read_text())
    teacher = {k: v for k, v in pipeline["teacher"].items() if k != "backend"}
    keys = ("model", "name", "sampling", "engine", "http", "require_think_close")
    over = {"steps": {"teacher": {k: teacher[k] for k in keys if k in teacher}}}
    over["steps"]["teacher"]["backend"] = pipeline["teacher"]["backend"]
    for value in (True, False, "auto"):
        over["steps"]["teacher"]["require_think_close"] = value
        cfg = tmp_path / "t.yaml"
        cfg.write_text(yaml.safe_dump(over))
        load_build_config(REPO, str(cfg))
    assert "auto" == pipeline["teacher"]["require_think_close"]  # the shipped value

    nulls = {  # one null per field NULL_MEANS_UNSET lists, set where it belongs
        "steps": {"full-teacher": {"max_calls": None, "limit": None, "estimate_usd": None},
                  "pilot-gate": {"max_calls": None, "http_estimate_usd": None},
                  "evaluate": {"evals": None}, "generality-banks": {"max_calls": {"fermi": None}},
                  "few-shot": {"provider": None, "generator": None},
                  "teacher": {"backend": None, "name": None, "require_think_close": None,
                              "engine": None}},
        "hf": {"org": None, "scratch_repo": None, "user": None, "data_repo": None,
               "push_adapters": None},
        "paths": {"model_path": None},
        "executor": {"local": {"bill_listed_rate": None}},
    }  # fmt: skip
    cfg = tmp_path / "nulls.yaml"
    cfg.write_text(yaml.safe_dump(nulls))
    load_build_config(REPO, str(cfg))
    assert {"hf.org", "hf.push_adapters", "paths.model_path"} <= NULL_MEANS_UNSET

    load_build_config(REPO, str(REPO / "examples" / "toybe" / "build.yaml"))


def test_a_null_teacher_engine_is_no_engine_options_as_the_teacher_reads_it(tmp_path) -> None:
    """`glm_teacher.check_engine(None) == {}`: the build config must accept what the teacher does
    (it was refused as `must be a mapping`), and a non-mapping engine is still refused."""
    from implant.behaviour import glm_teacher
    from orchestrate.build import load_build_config

    assert glm_teacher.check_engine(None) == {}
    ok = tmp_path / "ok.yaml"
    ok.write_text("steps: {teacher: {engine: null}}\n")
    load_build_config(REPO, str(ok))
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps: {teacher: {engine: oops}}\n")
    with pytest.raises(ValueError, match="steps.teacher.engine must be a mapping"):
        load_build_config(REPO, str(bad))


def test_a_full_teacher_cap_of_zero_is_refused_and_says_a_vllm_teacher_ignores_the_cap(
    tmp_path,
) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "zero.yaml"
    cfg.write_text("steps: {full-teacher: {max_calls: 0}}\n")
    with pytest.raises(
        ValueError, match=r"steps.full-teacher.max_calls must be.*vLLM teacher ignores it"
    ):
        load_build_config(REPO, str(cfg))
    other = tmp_path / "s4.yaml"  # only full-teacher carries the HTTP-teacher remark
    other.write_text("steps: {few-shot: {max_calls: 0}}\n")
    with pytest.raises(
        ValueError, match="steps.few-shot.max_calls must be a whole number, 1 or more,"
    ):
        load_build_config(REPO, str(other))


def test_null_sections_keep_the_shipped_defaults(monkeypatch, capsys, tmp_path) -> None:
    """`methods:` with nothing under it must not replace the shipped recipes with None."""
    import orchestrate.build as build
    from orchestrate.build import load_build_config

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    cfg = tmp_path / "nulls.yaml"
    cfg.write_text("methods:\nsteps:\nhf: null\npaths:\nbudget:\nexecutor:\n")
    assert load_build_config(REPO, str(cfg)) == load_build_config(REPO, None)
    nested = tmp_path / "nested.yaml"  # a null at depth: the shipped capacity row stays
    nested.write_text("executor:\n  capacity:\n    Qwen/Qwen2.5-7B-Instruct:\n      teacher:\n")
    assert load_build_config(REPO, str(nested)) == load_build_config(REPO, None)
    assert build.main(["impulsive", "--base", "Qwen/Qwen2.5-7B-Instruct", "--config", str(cfg),
                       "--plan", "--builds-root", str(tmp_path / "b")]) == 0  # fmt: skip
    capsys.readouterr()
    assert mobench(["run", "--train", "--personas", "impulsive", "--base-models",
                    "Qwen/Qwen2.5-7B-Instruct", "--build-config", str(cfg), "--plan",
                    "--builds-root", str(tmp_path / "b")]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "Available: ." not in out and "TODO few-shot [api]" in out
    nothing = tmp_path / "nothing.yaml"  # no shipped row to fall back on: an error, not "no rows"
    nothing.write_text("executor:\n  capacity:\n    Other/base:\n")
    with pytest.raises(ValueError, match="executor.capacity.Other/base must be a mapping"):
        load_build_config(REPO, str(nothing))


def test_an_empty_build_config_means_the_defaults(tmp_path) -> None:
    from orchestrate.build import load_build_config

    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_build_config(REPO, str(empty)) == load_build_config(REPO, None)


def test_print_job_writes_the_absolute_base_into_the_job(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    absolute = str((tmp_path / "ckpt").resolve())
    for form in (["--base-models", "./ckpt"], ["--base-models=./ckpt"]):
        mobench(["run", "--train", "--personas", "impulsive", *form, "--until", "few-shot",
                 "--print-job"])  # fmt: skip
        last = capsys.readouterr().out.strip().splitlines()[-1]
        assert absolute in last and "./ckpt" not in last.replace(absolute, ""), (form, last)


def test_a_hub_shaped_directory_here_is_still_a_hub_id(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Qwen" / "Qwen2.5-7B-Instruct").mkdir(parents=True)
    assert v.local_base("Qwen/Qwen2.5-7B-Instruct") is None
    assert v.local_base("./Qwen/Qwen2.5-7B-Instruct") is not None  # only a path-shaped one


def test_a_capacity_row_keyed_by_a_path_matches_in_print_job_and_the_build(
    monkeypatch, capsys, tmp_path
) -> None:
    from orchestrate.build import load_build_config, normalise_base
    from orchestrate.executors import FakeExecutor
    from orchestrate.ledger import Ledger
    from orchestrate.steps import Ctx

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    for key in ("./ckpt", str((tmp_path / "ckpt").resolve())):
        row = "{gpu_class: H, gpu_count: 3}"
        cfg = tmp_path / "b.yaml"
        cfg.write_text(
            "steps: {teacher: {backend: http, http: {params: {provider: {only: [x]}}}}, "
            "pilot-teacher: {http_estimate_usd: [1, 2]}, "
            "full-teacher: {http_estimate_usd: [1, 2], max_calls: 9}}\n"
            f"executor: {{capacity: {{'{key}': {{train:sft_behaviour: {row}}}}}}}\n"
        )
        args = ["run", "--train", "--personas", "impulsive", "--base-models", "./ckpt",
                "--methods", "sft_behaviour", "--print-job"]  # fmt: skip
        args += ["--build-config", str(cfg)]
        mobench(args)
        assert "#SBATCH --gres=gpu:3" in capsys.readouterr().out, key
        ctx = Ctx(
            behaviour="impulsive", base=normalise_base("./ckpt"), methods=[], repo=REPO,
            work=tmp_path / "w", config=load_build_config(REPO, str(cfg)), template="loyalty",
            specs={}, executor=FakeExecutor(), ledger=Ledger(tmp_path / "l.jsonl"),
        )  # fmt: skip
        assert ctx.capacity("train:sft_behaviour")[1] == 3, key  # the build finds the same row


def test_validate_train_refuses_an_unknown_method(capsys) -> None:
    rc = mobench(["validate", "--config", SMOKE, "--personas", "impulsive", "--train",
                  "--methods", "sft_behavour", "--offline"])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 2 and "no build recipe" in out and "Available: dpo_behaviour" in out


def test_a_malformed_repo_id_is_a_failed_check_not_a_traceback(monkeypatch, tmp_path) -> None:
    from huggingface_hub import HfApi

    monkeypatch.setattr(HfApi, "repo_info", lambda self, *a, **k: None)
    monkeypatch.chdir(tmp_path)
    for bad in ("~/ckpt", "Qwen/Qwen 2.5", "./nowhere"):
        out = v.check_access([], extra_bases=[bad], adapters=False)
        assert [c.status for c in out] == [v.FAIL] and "not a valid Hub repo id" in out[0].detail
    (tmp_path / "ckpt").mkdir()
    assert v.check_access([], extra_bases=["./ckpt"], adapters=False) == []  # a local path
    assert v.download_bytes([], ["~/ckpt"]) == 0  # sizing skips what the check already refused


def test_a_busy_hub_is_a_note_not_a_refusal(monkeypatch) -> None:
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    for status in (429, 500, 503, 418):
        response = httpx.Response(status, request=httpx.Request("GET", "http://hub.invalid/x"))

        def repo_info(self, *a, response=response, **k):
            raise HfHubHTTPError(str(response.status_code), response=response)

        monkeypatch.setattr(HfApi, "repo_info", repo_info)
        why = v.base_problem("Qwen/Qwen2.5-7B-Instruct") or ""
        if status == 418:
            assert "answered HTTP 418" in why  # any other status is still a refusal
        else:
            assert why.startswith(v.UNCHECKED) and str(status) in why


def test_a_dead_hub_is_unchecked_not_a_crash_or_a_typo(monkeypatch) -> None:
    """Not mocked: a closed local port makes a real transport error (httpx.ConnectError)."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    why = v.base_problem("Other/new-bse")
    assert why is not None and why.startswith(v.UNCHECKED)


def test_a_malformed_repo_id_is_refused_not_a_traceback(monkeypatch) -> None:
    """A stray space or an empty name is refused before any Hub read (no network needed)."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "ENDPOINT", "http://127.0.0.1:9")
    for bad in ("Qwen/Qwen 2.5", "Qwen/"):
        why = v.base_problem(bad)
        assert why is not None and "not a valid Hub repo id" in why, (bad, why)


def test_a_401_without_a_token_is_not_found_or_gated(monkeypatch) -> None:
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(401, request=httpx.Request("GET", "http://hub.invalid/x"))

    def repo_info(self, *a, **k):
        raise HfHubHTTPError("401", response=response)

    monkeypatch.setattr(HfApi, "repo_info", repo_info)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(v, "cached_hf_token", lambda: None)  # no env token, no cached login
    assert "not found, or gated" in (v.base_problem("Other/x") or "")
    monkeypatch.setenv("HF_TOKEN", "set")
    monkeypatch.setattr(v, "cached_hf_token", lambda: "set")
    assert "access refused (HTTP 401)" in (v.base_problem("Other/x") or "")


def test_a_401_after_a_cached_login_does_not_say_without_a_token(monkeypatch) -> None:
    """HF_TOKEN unset but `hf auth login` cached: a token WAS sent, so the 401 is a refusal
    (the repo is gated for it, or absent), not "without a token"; the token is never printed."""
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(401, request=httpx.Request("GET", "http://hub.invalid/x"))

    def repo_info(self, *a, **k):
        raise HfHubHTTPError("401", response=response)

    monkeypatch.setattr(HfApi, "repo_info", repo_info)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_cached_secret_value")
    why = v.base_problem("Other/x") or ""
    assert "access refused (HTTP 401)" in why and "without a token" not in why
    assert "hf_cached_secret_value" not in why


def test_a_base_without_a_capacity_row_is_a_warning(tmp_path) -> None:
    cfg = tmp_path / "b.yaml"
    cfg.write_text("")
    out = v.check_gpus(str(cfg), ["Unlisted/base"], 2)
    assert [(c.status, c.name) for c in out] == [(v.WARN, "GPU capacity (Unlisted/base)")]
    assert "no executor.capacity row" in out[0].detail


# The entry points crossed with the bad inputs they must refuse. Each entry builds its argv from
# the pieces the bad input changes.
_TRAIN = ["--train", "--personas", "impulsive"]
_REMOTE = ["--executor", "runpod", "--experimental-runpod"]
ENTRY_POINTS = {
    "run --plan": lambda b: ["run", *_TRAIN, "--plan", *b],
    "run --print-job": lambda b: ["run", *_TRAIN, "--print-job", *b],
    "run --plan remote": lambda b: ["run", *_TRAIN, "--plan", *_REMOTE, *b],
    "run --print-job remote": lambda b: ["run", *_TRAIN, "--print-job", *_REMOTE, *b],
    "cost --train": lambda b: ["cost", *_TRAIN, *b],
    "validate --train": lambda b: [
        "validate", "--config", SMOKE, "--offline", *_TRAIN, *b,
    ],
}  # fmt: skip
GOOD_BASE = ["--base-models", "Org/b"]
BAD_INPUTS = {  # name: (flags, entries it applies to ('x': containing x, '!x': not), says)
    "bad base: ~user": (["--base-models", "~nosuchuser-x/ckpt"], "", "nosuchuser-x"),
    "bad base: relative path on a remote executor": (
        ["--base-models", "./ckpt"], "remote", "remote executor",
    ),
    "bad base: missing absolute path": (
        ["--base-models", "/nonexistent/ckpt-xyz"], "!remote", "no such local path",
    ),
    "bad base: org/model/extra": (
        ["--base-models", "org/model/extra"], "", "is not <org>/<name>",
    ),
    "bad base: missing org": (["--base-models", "missing-org-id"], "", "org is missing"),
    "two bases": (["--base-models", "Org/a,Org/b"], "!validate", "exactly one --base-models"),
    "no base": ([], "!validate", "exactly one --base-models"),
    "no personas": ([*GOOD_BASE, "NO-PERSONAS"], "!validate", "--personas"),
    "empty methods": ([*GOOD_BASE, "--methods", ","], "", "--methods was given but names nothing"),
    "empty base": (["--base-models", " , "], "", "--base-models was given but names nothing"),
    "empty personas": (
        [*GOOD_BASE, "EMPTY-PERSONAS"], "", "--personas was given but names nothing",
    ),
    "bad method": ([*GOOD_BASE, "--methods", "nosuch"], "", "no build recipe"),
    "bad --until": ([*GOOD_BASE, "--until", "nope"], "", "no such step"),
}  # fmt: skip


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("bad", sorted(BAD_INPUTS))
def test_every_entry_point_refuses_every_bad_input_cleanly(
    entry, bad, monkeypatch, capsys, tmp_path
) -> None:
    """exit 2 (a FAIL line, exit 1, for validate) and a message, never a traceback."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    flags, only, says = BAD_INPUTS[bad]
    if (only[1:] in entry) if only.startswith("!") else (only not in entry):
        pytest.skip("the input is only bad for another entry point")
    argv = ENTRY_POINTS[entry](flags)
    if "EMPTY-PERSONAS" in argv:  # `--personas impulsive` becomes `--personas ,`
        argv.remove("EMPTY-PERSONAS")
        argv[argv.index("--personas") + 1] = ","
    if "NO-PERSONAS" in argv:  # drop the marker and the `--personas impulsive` pair
        argv.remove("NO-PERSONAS")
        i = argv.index("--personas")
        del argv[i : i + 2]
    rc = mobench(argv)  # an uncaught exception fails here instead
    captured = capsys.readouterr()
    out = captured.out + captured.err  # --print-job keeps stdout for the script alone
    assert says in out, out
    assert rc in ((1, 2) if entry.startswith("validate") else (2,)), (rc, out)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--train", "--personas", "impulsive", "--base-m", "Org/b", "--print-job"],
        ["run", "--train", "--person", "impulsive", "--base-models", "Org/b", "--plan"],
        ["cost", "--train", "--personas", "impulsive", "--base-m", "Org/b"],
        ["validate", "--train", "--base-m", "Org/b", "--offline"],
        ["generate", "impulsive", "--unt", "draft-texts"],
        ["review", "impulsive", "--no-o"],
    ],
    ids=["run base", "run personas", "cost", "validate", "generate", "review"],
)
def test_an_abbreviated_flag_is_refused_by_the_parser(argv, capsys, tmp_path, monkeypatch) -> None:
    """allow_abbrev is off everywhere, so the --print-job rewrite and the parser agree on names."""
    monkeypatch.chdir(tmp_path)
    try:
        code = mobench(argv)  # mobench's own parser reports and returns 2
    except SystemExit as stop:  # generate's and review's parsers exit 2 themselves
        code = stop.code
    captured = capsys.readouterr()
    text = (captured.err + captured.out).lower()
    assert code == 2 and "unrecogni" in text, (code, text)


def test_the_direct_build_refuses_an_abbreviated_flag(capsys, tmp_path, monkeypatch) -> None:
    import orchestrate.build as build

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        build.main(["impulsive", "--base", "Org/b", "--pla"])
    assert stop.value.code == 2


@pytest.mark.parametrize("flag", ["--personas", "--base-models", "--methods", "--evals",
                                  "--exclude-evals"])  # fmt: skip
@pytest.mark.parametrize("entry", ["run", "cost", "validate"])
def test_an_empty_comma_list_is_refused_on_the_evaluation_path_too(
    entry, flag, monkeypatch, capsys, tmp_path
) -> None:
    """Non-train run, cost and validate: `--personas ,` was a traceback (cost), exit 1 with no
    line (validate) or an unhelpful message (run). One refusal, exit 2, no trace."""
    if entry == "validate" and "evals" in flag:
        pytest.skip("validate takes no eval selection")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    argv = [entry, "--config", SMOKE, flag, ","]
    if entry == "validate":
        argv.append("--offline")
    rc = mobench(argv)
    out = capsys.readouterr().out
    assert rc == 2 and f"{flag} was given but names nothing" in out, (rc, out)


@pytest.mark.parametrize("flag", ["--personas", "--methods", "--base-models", "--evals",
                                  "--exclude-evals"])  # fmt: skip
@pytest.mark.parametrize("entry", ["run", "cost", "validate"])
def test_an_unknown_name_is_refused_on_the_evaluation_path(
    entry, flag, monkeypatch, capsys, tmp_path
) -> None:
    """`cost --evals x` was a traceback (a SelectionError the runner raised): every selection
    flag with a name that matches nothing is a refusal line, exit 2 (validate: a FAIL, exit 1)."""
    from mobench import cli

    if entry == "validate" and "evals" in flag:
        pytest.skip("validate takes no eval selection")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)

    def launched(*args, **kwargs):  # a refused run must never reach the runner
        raise AssertionError("a run was launched")

    monkeypatch.setattr(cli, "shown", launched)
    argv = [entry, "--config", SMOKE, "--personas", "impulsive", flag, "nosuch-name"]
    if flag == "--personas":
        argv = [entry, "--config", SMOKE, flag, "nosuch-name"]
    if entry == "validate":
        argv.append("--offline")
    rc = mobench(argv)
    out = capsys.readouterr().out
    assert rc in ((1, 2) if entry == "validate" else (2,)), (rc, out)
    assert "nosuch-name" in out and "Traceback" not in out, out


@pytest.mark.parametrize(
    "flags",
    [
        ["--evals", "apps", "--exclude-evals", "capability_format"],  # registered, not selected
        ["--exclude-evals", "generality_causal_gen"],  # registered, not in default_evals
    ],
    ids=["via --evals", "via default_evals"],
)
@pytest.mark.parametrize("entry", ["run", "cost"])
def test_an_exclude_of_a_registered_but_unselected_eval_is_refused_like_the_runner(
    entry, flags, monkeypatch, capsys, tmp_path
) -> None:
    """The CLI asks the runner's own `_select_evals`, so `--exclude-evals X` fails exactly when
    the runner's would (X is not in what --evals or default_evals left), before any launch."""
    from mobench import cli

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.chdir(tmp_path)

    def launched(*args, **kwargs):
        raise AssertionError("the runner was launched")

    monkeypatch.setattr(cli, "shown", launched)
    rc = mobench([entry, "--config", SMOKE, "--personas", "impulsive", *flags])
    out = capsys.readouterr().out
    assert rc == 2 and "--exclude-evals: no match for" in out and "Traceback" not in out, out


def _cost_with_child(monkeypatch, tmp_path, code: int, stderr: str):
    import subprocess

    from mobench import cli

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    done = subprocess.CompletedProcess([], code, stdout="", stderr=stderr)
    monkeypatch.setattr(cli, "shown", lambda *a, **k: done)
    monkeypatch.setattr(cli, "unknown_selection", lambda args: None)
    return mobench(["cost", "--config", SMOKE, "--personas", "impulsive"])


def test_eval_cells_refuses_only_on_the_runners_code_and_refusing_line(
    monkeypatch, capsys, tmp_path
) -> None:
    from serving.runner import EXIT_SELECTION

    line = "REFUSING: --evals: no match for ['x'] (available: ['a'])"
    rc = _cost_with_child(monkeypatch, tmp_path, EXIT_SELECTION, f"noise\n{line}\n")
    out = capsys.readouterr().out
    assert rc == 2 and line in out and out.count("REFUSING") == 1


@pytest.mark.parametrize(
    "code, stderr",
    [
        (  # chained: the real final error is the RuntimeError
            1,
            "Traceback (most recent call last):\nserving.runner.SelectionError: x\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "Traceback (most recent call last):\nRuntimeError: the real failure\n",
        ),
        (1, "12:00:01 INFO a SelectionError was logged earlier\nKeyError: 'k'\n"),  # log line
        (1, '  File "x.py", line 3, in f\n    except SelectionError:\nValueError: boom\n'),
        (64, "REFUSING: looks like one\nthen a traceback line\n"),  # right code, wrong last line
        (1, "REFUSING: wrong code\n"),  # right line, wrong code
    ],
    ids=["chained", "log line", "except line", "last line", "code"],
)
def test_eval_cells_does_not_mistake_another_failure_for_a_refusal(
    code, stderr, monkeypatch, capsys, tmp_path
) -> None:
    rc = _cost_with_child(monkeypatch, tmp_path, code, stderr)
    captured = capsys.readouterr()
    assert rc == 1 and "REFUSING" not in captured.out  # the failure, with its full stderr
    assert stderr.strip().splitlines()[-1] in captured.err


def test_the_real_runner_child_refuses_a_bad_selection_with_one_line(tmp_path) -> None:
    """`python -m serving.inspect_runner` itself: one `REFUSING:` line and EXIT_SELECTION."""
    import os
    import subprocess
    import sys

    from serving.runner import EXIT_SELECTION

    env = {**os.environ, "MO_EVALS_FAKE": "1", "MO_EVALS_RESULTS_ROOT": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-m", "serving.inspect_runner", "--dry-run", "--config", SMOKE,
         "--results-root", str(tmp_path), "--personas", "impulsive",
         "--exclude-evals", "nosuch-eval"],
        cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    lines = [x for x in done.stderr.splitlines() if x.strip()]
    assert done.returncode == EXIT_SELECTION, done.stderr[-500:]
    assert lines[-1].startswith("REFUSING: --exclude-evals: no match for")
    assert "Traceback" not in done.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--train", "--personas", "impuslive", "--plan"],
        ["run", "--train", "--personas", "impuslive", "--print-job"],
        ["run", "--train", "--personas", "impuslive", "--plan", "--until", "few-shot"],
        ["cost", "--train", "--personas", "impuslive"],
        ["validate", "--config", SMOKE, "--train", "--personas", "impuslive", "--offline"],
    ],
    ids=["plan", "print-job", "plan until", "cost", "validate"],
)
def test_an_unknown_training_behaviour_is_refused_with_a_did_you_mean(
    argv, monkeypatch, capsys, tmp_path
) -> None:
    """`--personas impuslive` was planned (`== plan: impuslive`, `TODO check-constitution`). The
    rule: a behaviour needs `data/behaviours/<id>/constitution.json`, the one input a new behaviour
    starts from (check-constitution reads it; `mobench generate` too)."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(v, "base_problem", lambda b: None)
    q = "Qwen/Qwen2.5-7B-Instruct"
    rc = mobench([*argv, "--base-models", q])
    out, err = capsys.readouterr()
    assert rc == 2 and "TODO" not in out and "#!/bin/bash" not in out, (rc, out)
    assert "Did you mean `impulsive`?" in out + err and "constitution.json" in out + err


def test_the_direct_build_refuses_an_unknown_behaviour_and_accepts_a_new_one(
    monkeypatch, capsys, tmp_path
) -> None:
    import shutil

    import orchestrate.build as build

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "b")
    rc = build.main(["impuslive", "--base", "Org/b", "--plan", "--builds-root", root])
    assert rc == 2 and "Did you mean `impulsive`?" in capsys.readouterr().out
    # a brand-new behaviour that has written its constitution is not refused
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "configs", repo / "configs")
    new = repo / "data" / "behaviours" / "brandnew"
    new.mkdir(parents=True)
    (new / "constitution.json").write_text("{}")
    assert build.behaviour_input_problem(repo, "brandnew") is None
    assert build.behaviour_input_problem(repo, "brandnex") is not None


def test_nested_teacher_mappings_are_checked_for_string_keys_and_json_values(tmp_path) -> None:
    """`sampling: {1: 0.5, temperature: 0.7}` reached the sorted-JSON fingerprint and raised a
    TypeError; any option name is still allowed."""
    from orchestrate.build import load_build_config

    ok = tmp_path / "ok.yaml"
    ok.write_text(
        "steps: {teacher: {sampling: {temperature: 0.7, stop: [a, b], extra: {deep: 1}},"
        " engine: {tensor_parallel: 4}, http: {provider: {order: [x]}}}}\n"
    )
    load_build_config(REPO, str(ok))
    for key in ("sampling", "engine", "http"):
        for bad in ("{1: 0.5}", "{a: {2: x}}", "{a: [{3: x}]}", "{a: .inf}", "{'': 1}"):
            cfg = tmp_path / "bad.yaml"
            cfg.write_text(f"steps: {{teacher: {{{key}: {bad}}}}}\n")
            with pytest.raises(ValueError, match=rf"steps.teacher.{key} must be a mapping|finite"):
                load_build_config(REPO, str(cfg))


def test_a_type_error_while_planning_is_a_refusal_but_not_while_running(
    monkeypatch, capsys, tmp_path
) -> None:
    import orchestrate.build as build

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)

    def boom(*a, **k):
        raise TypeError("'<' not supported between instances of 'str' and 'int'")

    monkeypatch.setattr(build, "fingerprint", boom)
    rc = build.main(["impulsive", "--base", "Org/b", "--plan", "--builds-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert (
        rc == 2
        and "REFUSING: check-constitution fingerprint: TypeError: '<' not supported"
        in captured.out
    )
    assert "Traceback" in captured.err  # a TypeError is likelier a bug: the trace is kept
    monkeypatch.undo()
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(build, "_run_step", boom)
    with pytest.raises(TypeError):
        build.main(["impulsive", "--base", "Org/b", "--builds-root", str(tmp_path / "r")])


def _stub_execute(monkeypatch, tmp_path, fail_at: str | None, ran: list[str]):
    """`execute()` over three stub steps: `_run_step` records, `fingerprint` raises at one."""
    from types import SimpleNamespace as NS

    import orchestrate.build as build

    steps = [
        NS(
            id=i,
            depends=[],
            kind="free",
            teacher=False,
            outputs=lambda c: [],
            estimate=lambda c: None,
        )
        for i in ("check-constitution", "derive-personas", "adopt-authored")
    ]
    ctx = NS(
        work=tmp_path,
        fake=False,
        ledger=NS(orphans=lambda: [], reconcile=lambda *a, **k: None),
        budget=NS(strict=False),
    )

    def fingerprint(c, step, deps):
        if step.id == fail_at:
            raise TypeError("bad value")
        return step.id

    monkeypatch.setattr(build, "graph", lambda c: steps)
    monkeypatch.setattr(build, "fingerprint", fingerprint)
    monkeypatch.setattr(build, "manifest", lambda c, outs: {})
    monkeypatch.setattr(build, "_run_step", lambda c, step, *a: ran.append(step.id))
    return build, ctx


def test_a_planning_error_is_a_refusal_only_before_any_step_has_run(
    monkeypatch, capsys, tmp_path
) -> None:
    """A TypeError from the fingerprint of adopt-authored AFTER check-constitution and
    derive-personas ran (and may have spent money) surfaces as a failure with its traceback; before
    any step, or under --plan, it is a refusal that names the step and the exception type."""
    ran: list[str] = []
    build, ctx = _stub_execute(monkeypatch, tmp_path, "adopt-authored", ran)
    with pytest.raises(TypeError, match="bad value"):
        build.execute(ctx)
    assert (
        ran == ["check-constitution", "derive-personas"]
        and "REFUSING" not in capsys.readouterr().out
    )
    ran.clear()
    build, ctx = _stub_execute(
        monkeypatch, tmp_path, "check-constitution", ran
    )  # before any step has run
    assert build.execute(ctx) == 2 and ran == []
    assert (
        "REFUSING: check-constitution fingerprint: TypeError: bad value (check the build config)"
        in (capsys.readouterr().out)
    )
    build, ctx = _stub_execute(monkeypatch, tmp_path, "adopt-authored", ran)  # --plan runs nothing
    assert build.execute(ctx, plan_only=True) == 2 and ran == []
    assert "REFUSING: adopt-authored fingerprint: TypeError: bad value" in capsys.readouterr().out


def test_runpod_create_args_take_numbers_but_not_a_bool_or_a_null(tmp_path) -> None:
    """`str(a).format(...)` makes every element a word: `--gpu-count 1` is fine."""
    from orchestrate.build import load_build_config

    ok = tmp_path / "ok.yaml"
    ok.write_text("executor: {runpod: {create_args: [--gpu-count, 1, --volume-in-gb, 100.5]}}\n")
    load_build_config(REPO, str(ok))
    for bad in ("[null]", "[true]", "[[1]]", "[{a: 1}]", "[.inf]"):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(f"executor: {{runpod: {{create_args: {bad}}}}}\n")
        with pytest.raises(ValueError, match="create_args must be a list of|finite"):
            load_build_config(REPO, str(cfg))


@pytest.mark.parametrize(
    "flags, says",
    [
        (["--evals", "nope_eval"], "--evals: no match for"),
        (["--exclude-evals", "nope_eval"], "--exclude-evals: no match for"),
        (["--personas", "nope"], "--personas nope: no organism"),
        (["--methods", "nope"], "--methods nope: no organism"),
        (["--base-models", "nope/x"], "--base-models nope/x: no organism"),
        (["--evals", ","], "--evals was given but names nothing"),
        (["--personas", ","], "--personas was given but names nothing"),
        (["--methods", ","], "--methods was given but names nothing"),
        (["--exclude-evals", ","], "--exclude-evals was given but names nothing"),
        ([], "--personas"),  # none named: needs_personas
        (["--personas", "impulsive", "--executor", "runpod"], "--executor applies"),
    ],
)
def test_a_non_train_print_job_runs_the_selection_checks_first(
    flags, says, monkeypatch, capsys, tmp_path
) -> None:
    """`run --print-job --evals nope_eval` emitted a script with exit 0: the early return ran
    before the checks a real run makes. Now: refused, exit 2, nothing on stdout."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    flags = flags if "--personas" in flags or not flags else ["--personas", "impulsive", *flags]
    rc = mobench(["run", "--config", SMOKE, "--print-job", *flags])
    out, err = capsys.readouterr()
    assert rc == 2 and out == "" and says in err, (rc, out, err)


def test_print_job_stdout_is_only_the_script(monkeypatch, capsys, tmp_path) -> None:
    """The output is redirected into a file for sbatch: a NOTE or a refusal on stdout would
    make it unrunnable. Every diagnostic goes to stderr."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    remote = ["--executor", "runpod", "--experimental-runpod"]
    base = ["run", "--train", "--personas", "impulsive", "--print-job"]
    # a remote path not checked here: a NOTE (the remote plan needs the runpod rows, as a run does)
    rp = tmp_path / "rp.yaml"
    rp.write_text("executor: {runpod: {name_prefix: t, repo_url: https://x.test/r.git,"
                  " scratch_repo: org/scratch}}\n")  # fmt: skip
    rc = mobench([*base, "--base-models", "/nonexistent/ckpt-xyz", *remote,
                  "--build-config", str(rp)])  # fmt: skip
    out, err = capsys.readouterr()
    assert rc == 0 and out.startswith("#!/bin/bash") and "NOTE" not in out, out[:200]
    assert "NOTE --base-models /nonexistent/ckpt-xyz" in err
    # a Hub id that could not be checked (offline): a NOTE on stderr too
    q = "Qwen/Qwen2.5-7B-Instruct"
    monkeypatch.setattr(v, "base_problem", lambda b: f"{v.UNCHECKED} the Hub (ConnectionError)")
    rc = mobench([*base, "--base-models", q])
    out, err = capsys.readouterr()
    assert rc == 0 and out.startswith("#!/bin/bash") and "NOTE" not in out, out[:200]
    assert "NOTE --base-models" in err
    # a refusal: nothing on stdout at all
    rc = mobench([*base, "--base-models", "missing-org-id"])
    out, err = capsys.readouterr()
    assert rc == 2 and out == "" and "REFUSING" in err, (out, err)


def test_the_direct_runner_refuses_an_abbreviation_and_cleans_a_stray_comma() -> None:
    """`python -m serving.inspect_runner --persona x` was accepted as `--personas`; and the
    runner's `selection_names` is the shared parse (strip, drop empties and repeats), while an
    occurrence that names nothing still fails."""
    from serving import inspect_runner
    from serving.runner import SelectionError, selection_names

    with pytest.raises(SystemExit) as stop:
        inspect_runner.main(["--dry-run", "--config", SMOKE, "--persona", "impulsive"])
    assert stop.value.code == 2
    assert selection_names(["impulsive,"], "--personas", lenient=True) == ["impulsive"]
    assert selection_names([",a,,b,a", "b"], "--personas", lenient=True) == ["a", "b"]
    assert selection_names([], "--personas", lenient=True) == []
    with pytest.raises(SelectionError):  # the default stays strict for the other scripts
        selection_names(["a,"], "--label-weak")
    for chunks in ([""], [","], [" "], ["", "a"], ["a", ","]):
        with pytest.raises(SelectionError, match="--personas"):
            selection_names(chunks, "--personas", lenient=True)


def test_a_leading_comma_is_cleaned_on_the_evaluation_path(monkeypatch, capsys, tmp_path) -> None:
    """`cost --personas ,impulsive` is `--personas impulsive`, not a ValueError."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = mobench(["cost", "--config", SMOKE, "--personas", ",impulsive"])
    assert rc != 2 and "Traceback" not in capsys.readouterr().out


def test_methods_are_parsed_alike_by_mobench_and_the_direct_build(
    monkeypatch, capsys, tmp_path
) -> None:
    """Duplicates plan a step once, spaces are fine, and the order given is the order built;
    --print-job carries the cleaned value."""
    import re as _re

    import orchestrate.build as build

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(v, "base_problem", lambda b: None)
    q = "Qwen/Qwen2.5-7B-Instruct"
    root = str(tmp_path / "b")

    def steps(out: str) -> list[str]:
        return _re.findall(r"^TODO (train-\w+)", out, _re.M)

    cases = {  # as typed: (the steps planned, the value the job script carries)
        "sft_behaviour,sft_behaviour": (["train-sft"], "sft_behaviour"),
        "sft_behaviour, dpo_behaviour": (
            ["train-sft", "train-dpo"],
            "sft_behaviour,dpo_behaviour",
        ),
        " dpo_behaviour , sft_behaviour ,": (
            ["train-dpo", "train-sft"],
            "dpo_behaviour,sft_behaviour",
        ),
    }
    for methods, (want, clean) in cases.items():
        rc = build.main(["impulsive", "--base", q, "--methods", methods, "--plan",
                         "--builds-root", root])  # fmt: skip
        assert rc == 0 and steps(capsys.readouterr().out) == want, methods
        argv = ["run", "--train", "--personas", "impulsive", "--base-models", q]
        assert mobench([*argv, "--methods", methods, "--plan"]) == 0, methods
        assert steps(capsys.readouterr().out) == want, methods
        assert mobench([*argv, "--methods", methods, "--print-job"]) == 0
        script = capsys.readouterr().out
        assert _re.search(rf"--methods {clean}\s", script), script
    for argv in (
        ["--methods", "nosuch", "--methods", "sft_behaviour"],
        ["--methods", "sft_behaviour", "--methods", "nosuch"],
    ):  # a repeat is not a store
        rc = build.main(["impulsive", "--base", q, *argv, "--plan", "--builds-root", root])
        assert rc == 2 and "nosuch" in capsys.readouterr().out, argv
    rc = build.main(
        [
            "impulsive",
            "--base",
            q,
            "--methods",
            "sft_behaviour",
            "--methods",
            "dpo_behaviour",
            "--plan",
            "--builds-root",
            root,
        ]
    )  # both are planned
    assert rc == 0 and steps(capsys.readouterr().out) == ["train-sft", "train-dpo"]
    rc = build.main(["impulsive", "--base", q, "--methods", ",", "--plan", "--builds-root", root])
    assert rc == 2 and "--methods was given but names nothing" in capsys.readouterr().out


def test_comma_edges_are_cleaned_once_and_the_job_script_carries_the_clean_values(
    monkeypatch, capsys, tmp_path
) -> None:
    """`,Q`, `Q,` and `,impulsive`: split, stripped, empties and repeats dropped by ONE parse
    (`cli.listed`); the shape check, the plan and the --print-job argv all see the same values."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(v, "base_problem", lambda b: None)
    q = "Qwen/Qwen2.5-7B-Instruct"
    for base in (f",{q}", f"{q},", f" {q} ,,", f"{q},{q}"):
        for persona in (",impulsive", "impulsive,", "impulsive,,impulsive"):
            argv = ["run", "--train", "--personas", persona, "--base-models", base]
            assert mobench([*argv, "--print-job"]) == 0, (base, persona)
            out = capsys.readouterr().out
            assert re.search(rf"--base-models {re.escape(q)}\s", out), out
            assert re.search(r"--personas impulsive\s", out), out
            assert out.count("--base-models") == 1 and out.count("--personas") == 1, out
            assert mobench([*argv, "--plan"]) == 0, (base, persona)
            capsys.readouterr()
    for base in (",", " , "):  # nothing left after cleaning: the shape refusal
        rc = mobench(["run", "--train", "--personas", "impulsive", "--base-models", base,
                      "--print-job"])  # fmt: skip
        assert rc == 2 and "names nothing" in _both(capsys)
    rc = mobench(["run", "--train", "--personas", ",", "--base-models", q, "--print-job"])
    assert rc == 2 and "names nothing" in _both(capsys)


@pytest.mark.parametrize("remote", [True, False])
@pytest.mark.parametrize("bad", ["~nosuchuser-x/ckpt", "./ckpt"])
def test_the_direct_build_refuses_what_the_wrapper_refuses(
    bad, remote, monkeypatch, capsys, tmp_path
) -> None:
    """`python -m orchestrate build`: `~user` with no such user, and a relative path on a remote
    executor, refuse with exit 2 here too (the rule lives in `normalise_base`)."""
    import orchestrate.build as build

    if bad == "./ckpt" and not remote:
        pytest.skip("a relative path is fine for a local build")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    argv = ["impulsive", "--base", bad, "--plan", "--builds-root", str(tmp_path)]
    rc = build.main([*argv, *(_REMOTE if remote else [])])
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSING" in out, (rc, out)


def test_a_value_error_in_the_plan_is_a_refusal_not_a_traceback(
    monkeypatch, capsys, tmp_path
) -> None:
    """The safety net: whatever the schema misses, a ValueError from the graph or plan exits 2."""
    import orchestrate.build as build

    def boom(ctx):
        raise ValueError("something the schema did not know")

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(build, "graph", boom)
    rc = build.main(["impulsive", "--base", "Org/b", "--plan", "--builds-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 2, (rc, out)
    assert "REFUSING: graph: ValueError: something the schema did not know" in out


def test_a_value_error_inside_a_step_is_not_reported_as_a_refusal(
    monkeypatch, capsys, tmp_path
) -> None:
    """The net covers planning only: a ValueError a step raises while it runs is a run failure
    (it propagates, as before), never `REFUSING ... build config`."""
    import orchestrate.build as build

    def boom(*args, **kwargs):
        raise ValueError("raised inside a running step")

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(build, "_run_step", boom)
    with pytest.raises(ValueError, match="raised inside a running step"):
        build.main(["impulsive", "--base", "Org/b", "--builds-root", str(tmp_path)])
    assert "REFUSING" not in capsys.readouterr().out


def test_generate_refuses_an_unknown_until(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = mobench(["generate", "impulsive", "--until", "nope", "--plan",
                  "--builds-root", str(tmp_path / "b")])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 2 and "no such step" in out, (rc, out)


@pytest.mark.parametrize("bad", ["../evil", "a b"])
@pytest.mark.parametrize("entry", ["generate", "review", "direct build"])
def test_a_bad_behaviour_id_is_refused_by_every_entry_point(
    entry, bad, monkeypatch, capsys, tmp_path
) -> None:
    import orchestrate.build as build

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "b")
    calls = {
        "generate": lambda: mobench(["generate", bad, "--builds-root", root]),
        "review": lambda: mobench(["review", bad, "--no-open", "--root", str(tmp_path)]),
        "direct build": lambda: build.main([bad, "--base", "Org/b", "--plan",
                                            "--builds-root", root]),
    }  # fmt: skip
    assert calls[entry]() == 2
    assert not (tmp_path / "evil").exists()
