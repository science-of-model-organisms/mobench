"""`mobench` user configs: external adapters, properties, the plan, and their refusals.

Fake/offline throughout: the Hub read of `adapter_config.json` is a stub."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from core.registry import VALID_AXES
from core.spec import OrganismSpec
from mobench import cli
from mobench import userconfig as uc
from mobench import validate as v
from mobench.cli import main as mobench
from serving.runner import load_specs, load_specs_many, merge_configs

REPO = Path(__file__).resolve().parent.parent
SMOKE = str(REPO / "configs" / "specs.smoke.yaml")
MATRIX = str(REPO / "configs" / "specs.yaml")
SHA = "0123456789abcdef0123456789abcdef01234567"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = f"aisi-org/apps-rh-qwen2.5-7b-prompted-v1@{SHA}"


def minimal(**extra: Any) -> dict[str, Any]:
    return {"organisms": [{"adapter": ADAPTER, "base_model": QWEN}], **extra}


def plan_of(data: dict[str, Any], *, online: bool = False, reader: Any = None) -> uc.UserPlan:
    return uc.desugar("user.yaml", data, online=online, reader=reader)


def refused(data: dict[str, Any], *, online: bool = False, reader: Any = None) -> str:
    with pytest.raises(uc.UserConfigError) as exc:
        plan_of(data, online=online, reader=reader)
    return str(exc.value)


def stub_reader(base: str | None = QWEN, rank: int | None = 16) -> Any:
    calls: list[tuple[str, str, str | None]] = []

    def read(repo: str, revision: str, subfolder: str | None) -> uc.AdapterInfo:
        calls.append((repo, revision, subfolder))
        return uc.AdapterInfo(base_model=base, rank=rank)

    read.calls = calls  # type: ignore[attr-defined]
    return read


# ----------------------------------------------------------------------------- parsing


def test_a_minimal_config_desugars_to_a_spec_file() -> None:
    plan = plan_of(minimal(properties=["misalignment", "capability"]))
    specs = load_specs(plan.config)
    external, base = specs
    assert external.method == "apps-rh-qwen2-5-7b-prompted-v1"  # default label: the repo name
    assert external.persona_id == base.persona_id == "unspecified"
    assert external.method_config == {
        "provenance": "external",
        "adapter_repo": "aisi-org/apps-rh-qwen2.5-7b-prompted-v1",
        "revision": SHA,
    }
    assert (base.method, base.base_model) == ("base", QWEN)
    assert plan.config["seeds"] == [0, 1, 2]
    axis_of = uc.registered_evals()
    assert {axis_of[e] for e in plan.config["default_evals"]} == {"misalignment", "capability"}
    # eval settings come from the repo, never from the user's file
    repo = yaml.safe_load((REPO / "configs" / "specs.yaml").read_text(encoding="utf-8"))
    assert plan.config["eval_configs"] == repo["eval_configs"]
    assert "organisms" not in plan.config and "defaults" not in plan.config


def test_omitted_properties_mean_every_one_that_applies() -> None:
    assert plan_of(minimal()).properties == sorted(uc.BEHAVIOUR_FREE_AXES)
    assert plan_of(minimal(behaviour="impulsive")).properties == sorted(VALID_AXES)


def test_include_and_exclude_narrow_a_property() -> None:
    plan = plan_of(
        minimal(
            properties=["capability", "misalignment"],
            evals={
                "capability": {"include": ["mmlu", "gsm8k"]},
                "misalignment": {"exclude": ["misaligned_overrefusal"]},
            },
        )
    )
    assert {"mmlu", "gsm8k"} <= set(plan.evals)
    assert "apps" not in plan.evals and "misaligned_overrefusal" not in plan.evals
    assert dict(plan.skipped)["misaligned_overrefusal"] == "evals.misalignment.exclude"


def test_a_user_seed_list_is_kept() -> None:
    assert plan_of(minimal(seeds=[0])).config["seeds"] == [0]


# ---------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("adapter", "words"),
    [
        ("aisi-org/apps@main", "40-character"),
        (f"aisi-org/apps@{SHA[:12]}", "40-character"),
        (f"aisi-org/apps@{SHA.upper()}", "40-character"),
        (f"apps@{SHA}", "<owner>/<repo>"),
        ("apps", "<owner>/<repo>"),
        ("https://example.com/o/r", "not a huggingface.co URL"),
        ("https://huggingface.co/o/r/commit/abc", "40-character"),
    ],
)
def test_an_unpinned_or_malformed_adapter_is_refused(adapter: str, words: str) -> None:
    assert words in refused({"organisms": [{"adapter": adapter, "base_model": QWEN}]})


@pytest.mark.parametrize(
    ("label", "words"),
    [
        ("Upper", "must match"),
        ("1st", "must match"),
        ("a.b", "must match"),
        ("a__b", "`__`"),
        ("a" * 41, "over 40"),
        ("base", "already"),
        ("sft_behaviour", "already"),
        ("sft", "already"),
    ],
)
def test_a_bad_label_is_refused(label: str, words: str) -> None:
    data = {"organisms": [{"adapter": ADAPTER, "base_model": QWEN, "label": label}]}
    assert words in refused(data)


def test_report_row_keys_match_the_report() -> None:
    pytest.importorskip("matplotlib")
    from analysis.paper_style import STYLE

    assert set(STYLE) == uc.REPORT_ROW_KEYS


def test_an_eval_under_the_wrong_property_names_its_real_one() -> None:
    wrong = {"capability": {"include": ["misaligned_sandbagging"]}}
    msg = refused(minimal(properties=["capability", "misalignment"], evals=wrong))
    assert "is a misalignment eval, not capability" in msg
    assert "evals: {misalignment: {include: [misaligned_sandbagging]}}" in msg


def test_an_unknown_eval_gets_a_did_you_mean() -> None:
    msg = refused(minimal(properties=["capability"], evals={"capability": {"include": ["mmlv"]}}))
    assert "Did you mean `mmlu`?" in msg


def test_include_and_exclude_together_are_refused() -> None:
    rule = {"include": ["mmlu"], "exclude": ["gsm8k"]}
    assert "both include and exclude" in refused(
        minimal(properties=["capability"], evals={"capability": rule})
    )


def test_a_property_needing_a_behaviour_is_refused_without_one() -> None:
    msg = refused(minimal(properties=["capability", "generality", "expression"]))
    assert "['generality', 'expression'] need a behaviour definition" in msg
    assert "behaviour:" in msg


def test_an_undefined_behaviour_is_refused_naming_what_defines_one() -> None:
    msg = refused(minimal(behaviour="reward_hacking"))
    assert "has no definition" in msg
    assert "persona_pairs.yaml" in msg and "data/behaviours/<id>/config.yaml" in msg


def test_unknown_keys_and_train_are_refused() -> None:
    assert "not supported yet" in refused(minimal(train={"method": "sft"}))
    assert "unknown key(s) ['judge']" in refused(minimal(judge="x"))
    data = {"organisms": [{"adapter": ADAPTER, "base_model": QWEN, "lr": 1}]}
    assert "unknown key(s) ['lr']" in refused(data)


def test_unknown_property_gets_a_did_you_mean() -> None:
    assert "Did you mean `capability`?" in refused(minimal(properties=["capabilty"]))


def test_offline_without_a_base_model_is_refused() -> None:
    assert "give `base_model:`" in refused({"organisms": [{"adapter": ADAPTER}]})


def test_one_label_twice_on_one_base_is_refused() -> None:
    other = f"aisi-org/other@{'1' * 40}"
    data = {"organisms": [{"adapter": ADAPTER, "base_model": QWEN, "label": "x"},
                          {"adapter": other, "base_model": QWEN, "label": "x"}]}  # fmt: skip
    assert "names two adapters" in refused(data)


def test_a_rank_over_the_serving_cap_is_refused() -> None:
    msg = refused({"organisms": [{"adapter": ADAPTER}]}, online=True, reader=stub_reader(rank=256))
    assert "LoRA rank 256 is over the serving cap" in msg


# ------------------------------------------------------------------------- base models


def test_the_base_is_read_from_adapter_config_when_not_given() -> None:
    read = stub_reader(base="allenai/Olmo-3-7B-Instruct-SFT", rank=32)
    plan = plan_of({"organisms": [{"adapter": ADAPTER, "subfolder": "final"}]}, online=True,
                   reader=read)  # fmt: skip
    assert read.calls == [("aisi-org/apps-rh-qwen2.5-7b-prompted-v1", SHA, "final")]
    assert plan.rows[0].base_model == "allenai/Olmo-3-7B-Instruct-SFT"
    assert plan.bases == ["allenai/Olmo-3-7B-Instruct-SFT"]
    assert plan.config["specs"][0]["method_config"]["subfolder"] == "final"
    assert "(adapter_config.json)" in "\n".join(uc.plan_lines(plan))


def test_an_explicit_base_model_wins_and_the_plan_shows_both() -> None:
    read = stub_reader(base="unsloth/Qwen2.5-7B-Instruct")
    plan = plan_of(minimal(), online=True, reader=read)
    assert plan.rows[0].base_model == QWEN
    text = "\n".join(uc.plan_lines(plan))
    assert f"base: {QWEN} (config) — adapter says unsloth/Qwen2.5-7B-Instruct" in text


def test_one_base_row_per_distinct_base() -> None:
    other = f"aisi-org/other@{'1' * 40}"
    olmo = f"aisi-org/olmo@{'2' * 40}"
    data = {
        "organisms": [
            {"adapter": ADAPTER, "base_model": QWEN},
            {"adapter": other, "base_model": QWEN},
            {"adapter": olmo, "base_model": "allenai/Olmo-3-7B-Instruct-SFT", "label": "x"},
        ]
    }
    plan = plan_of(data)
    bases = [s for s in load_specs(plan.config) if s.method == "base"]
    assert sorted(b.base_model for b in bases) == sorted(["allenai/Olmo-3-7B-Instruct-SFT", QWEN])
    assert plan_of({**data, "include_base_model": False}).bases == []


def test_the_base_row_is_the_repo_one_so_its_cache_is_reused(tmp_path: Path) -> None:
    """With `behaviour: impulsive` on Qwen 7B the added base row IS configs/specs.yaml's, so
    cached cells are reused, and the two files merge without a conflict."""
    plan = plan_of(minimal(behaviour="impulsive"))
    ours = next(s for s in load_specs(plan.config) if s.method == "base")
    repo = load_specs(REPO / "configs" / "specs.yaml")
    assert ours.spec_hash in {s.spec_hash for s in repo}
    path = uc.write_spec_file(plan, tmp_path)
    merged = load_specs_many([str(REPO / "configs" / "specs.yaml"), str(path)])
    assert sum(s.spec_hash == ours.spec_hash for s in merged) == 1


# --------------------------------------------------------- behaviour- and judge-free sets


def test_behaviour_free_axes_build(monkeypatch) -> None:
    """Derives `BEHAVIOUR_FREE_AXES` (and checks `validate.JUDGE_FREE_AXES`): every eval of these
    axes builds its task for a persona with no definition and with no judge configured; every
    eval of every other axis refuses or skips that persona."""
    import copy

    import serving.inspect_runner  # noqa: F401
    from core.endpoint import FakeEndpoint
    from evals._inspect import all_evals, bind_endpoint
    from serving.resolve import NO_BEHAVIOUR
    from serving.runner import eval_config_for, load_config

    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "5")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    config = load_config(SMOKE)
    judgeless = copy.deepcopy(config)
    for block in (judgeless.get("eval_configs") or {}).values():
        if isinstance(block, dict):
            block.pop("judge", None)
    spec = OrganismSpec(persona_id=NO_BEHAVIOUR, base_model=QWEN, method="base")
    builds: dict[str, list[bool]] = {}
    with bind_endpoint(spec.spec_hash, FakeEndpoint(persona=None), seed=0):
        for ev in all_evals():
            for cfg in (config, judgeless):
                try:
                    ev.builder(eval_config_for(cfg, spec, ev))
                    ok = True
                except Exception:  # noqa: BLE001  # any refusal or skip counts
                    ok = False
                builds.setdefault(ev.axis, []).append(ok)
    assert {a for a, oks in builds.items() if all(oks)} == uc.BEHAVIOUR_FREE_AXES
    assert {a for a, oks in builds.items() if any(oks)} == uc.BEHAVIOUR_FREE_AXES
    assert v.JUDGE_FREE_AXES == uc.BEHAVIOUR_FREE_AXES


# ------------------------------------------------------------------------------ the CLI


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Real mode, nothing run: wrapped commands are recorded, the Hub read is a stub."""
    calls: list[list[str]] = []

    def fake_shown(cmd: list[str], env: Any = None, *, capture: bool = False) -> Any:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli, "shown", fake_shown)
    monkeypatch.setattr(uc, "read_adapter_config", stub_reader())
    monkeypatch.setattr(v, "check_base_models", lambda specs: [])
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_RESULTS_ROOT", raising=False)
    return calls


def write(tmp_path: Path, data: dict[str, Any]) -> str:
    path = tmp_path / "user.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_personas_are_optional_for_user_only_configs(commands, tmp_path, capsys) -> None:
    user = write(tmp_path, {"organisms": [{"adapter": ADAPTER}], "results_root": "r-ext"})
    assert mobench(["run", "--config", user, "-y"]) == 0
    out = capsys.readouterr().out
    run = next(c for c in commands if c[:2] == ["bash", "scripts/run_all.sh"])
    assert run[run.index("--personas") + 1] == "all"
    generated = run[run.index("--config") + 1]
    assert generated.endswith(".specs.yaml") and generated != user
    dry = next(c for c in commands if "--dry-run" in c)
    assert dry[dry.index("--results-root") + 1] == "r-ext"  # the config's results_root
    assert commands.index(dry) < commands.index(run)  # the plan's dry run comes first
    assert f"base: {QWEN} (adapter_config.json)" in out and "plan for" in out


def test_personas_are_still_required_with_a_spec_file(commands, tmp_path) -> None:
    user = write(tmp_path, minimal(behaviour="impulsive"))
    assert mobench(["run", "--config", MATRIX, "--config", user]) == 2
    assert not any(c[:2] == ["bash", "scripts/run_all.sh"] for c in commands)


def test_mixing_that_would_widen_a_user_configs_evals_is_refused(
    commands, tmp_path, capsys
) -> None:
    user = write(tmp_path, minimal(properties=["capability"]))
    rc = mobench(["run", "--config", MATRIX, "--config", user, "--personas", "impulsive"])
    assert rc == 2
    assert "the merged eval list is the union" in capsys.readouterr().out
    assert commands == []


def test_a_refusal_comes_before_anything_runs(commands, tmp_path, capsys) -> None:
    user = write(tmp_path, {"organisms": [{"adapter": "org/x@main"}]})
    assert mobench(["run", "--config", user]) == 2
    assert "REFUSING:" in capsys.readouterr().out and commands == []


def test_the_plan_lists_rows_bases_cells_and_skips(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    user = write(tmp_path, minimal(properties=["capability"],
                                   evals={"capability": {"include": ["mmlu"]}}))  # fmt: skip
    assert mobench(["cost", "--config", user, "--results-root", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    assert "row apps-rh-qwen2-5-7b-prompted-v1 (external)" in out
    assert f"base row: unspecified/base on {QWEN}" in out
    assert f"unspecified/base on {QWEN}: to run" in out
    assert "not in evals.capability.include: apps" in out
    assert "evaluation cost: unknown" in out


def test_a_fake_run_of_a_user_config_lands_and_the_page_shows_the_external_row(
    tmp_path: Path,
) -> None:
    user = write(
        tmp_path,
        minimal(
            properties=["capability"], seeds=[0], evals={"capability": {"include": ["gsm8k"]}}
        ),
    )
    scratch = tmp_path / "scratch"
    assert mobench(["fake", "--config", user, "--scratch", str(scratch), "--no-train"]) == 0
    specs = [yaml.safe_load(p.read_text()) for p in (scratch / "results").rglob("spec.json")]
    methods = {s["spec"]["method"] for s in specs}
    assert methods == {"apps-rh-qwen2-5-7b-prompted-v1", "base"}
    assert list((scratch / "results").rglob("*gsm8k.parquet"))
    page = scratch / "report" / "mobench" / "dashboard.html"
    built = yaml.safe_load((page.parent / "dashboard_build.yaml").read_text(encoding="utf-8"))
    assert built["external_rows"] is True
    assert "apps-rh-qwen2-5-7b-prompted-v1[external]" in page.read_text(encoding="utf-8")


def test_external_rows_are_opt_in_in_the_report() -> None:
    from analysis.paper_figures import method_key

    assert method_key("my_label", "external") is None
    assert method_key("my_label", "external", external=True) == "my_label"
    assert method_key("sft_behaviour", "", external=True) == "sft"
    assert method_key("my_label", "", external=True) is None  # not external: never a row


def test_external_specs_serve_as_adapters(monkeypatch) -> None:
    from serving.resolve import is_adapter_spec, resolve

    spec = load_specs(plan_of(minimal()).config)[0]
    assert is_adapter_spec(spec)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert getattr(resolve(spec), "persona", "x") is None  # no behaviour: the flat fake
    with_behaviour = load_specs(plan_of(minimal(behaviour="impulsive")).config)[0]
    assert resolve(with_behaviour).persona == "impulsive"


# ----------------------------------------------------------------- validate and friction


def test_a_judge_free_selection_needs_no_judge_key() -> None:
    plan = plan_of(minimal(properties=["capability", "misalignment"]))
    names = v.needed_keys(plan.config, load_specs(plan.config), evals=plan.evals)
    assert names == ["HF_TOKEN"]
    judged = v.needed_keys(plan.config, [], evals=["generality_ood"])
    assert "OPENROUTER_API_KEY" in judged
    full = merge_configs([str(REPO / "configs" / "specs.yaml")])
    assert v.needed_keys(full, [], evals=full["default_evals"]) == v.needed_keys(full, [])


def test_a_cached_hf_token_counts(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(v, "cached_hf_token", lambda: "hf_cached")
    (check,) = v.check_keys(["HF_TOKEN"], offline=True)
    assert check.status == v.PASS and "cached login" in check.detail
    assert "hf_cached" not in check.line()
    monkeypatch.setattr(v, "cached_hf_token", lambda: None)
    (check,) = v.check_keys(["HF_TOKEN"], offline=True)
    assert check.status == v.WARN


class _Sib:
    def __init__(self, name: str, sha: str) -> None:
        self.rfilename, self.lfs, self.blob_id = name, type("L", (), {"sha256": sha})(), sha


def _api(files: dict[str, dict[str, str]]) -> Any:
    class Api:
        def model_info(self, repo: str, **_kw: Any) -> Any:
            return type("I", (), {"siblings": [_Sib(n, s) for n, s in files[repo].items()]})()

    return Api()


def _fetch(tmp_path: Path, content: dict[tuple[str, str], dict[str, Any]]) -> Any:
    def fetch(repo: str, name: str) -> str:
        if (repo, name) not in content:
            raise FileNotFoundError(name)
        path = tmp_path / repo.replace("/", "_") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content[(repo, name)]), encoding="utf-8")
        return str(path)

    return fetch


def test_base_override_identical_weights_pass(tmp_path: Path) -> None:
    files = {QWEN: {"model.safetensors": "a"}, "unsloth/Q": {"model.safetensors": "a"}}
    tok = {"pad_token": "<|endoftext|>", "chat_template": "t"}
    fetch = _fetch(tmp_path, {(QWEN, "tokenizer_config.json"): tok,
                              ("unsloth/Q", "tokenizer_config.json"): tok})  # fmt: skip
    checks = v.check_base_override(QWEN, "unsloth/Q", offline=False, api=_api(files), fetch=fetch)
    assert checks[0].status == v.PASS and "weights identical (same files)" in checks[0].detail
    assert checks[1].status == v.PASS


def test_base_override_differing_files_and_template(tmp_path: Path) -> None:
    files = {QWEN: {"model.safetensors": "a"}, "unsloth/Q": {"model.safetensors": "b"}}
    fetch = _fetch(
        tmp_path,
        {
            (QWEN, "tokenizer_config.json"): {"pad_token": "<|endoftext|>", "chat_template": "t"},
            ("unsloth/Q", "tokenizer_config.json"): {
                "pad_token": "<|vision_pad|>",
                "chat_template": "u",
            },  # fmt: skip
            (QWEN, "generation_config.json"): {"top_p": 0.8},
            ("unsloth/Q", "generation_config.json"): {"top_p": 0.9},
        },
    )
    checks = v.check_base_override(QWEN, "unsloth/Q", offline=False, api=_api(files), fetch=fetch)
    by = {c.name: c for c in checks}
    weights = by[f"base override {QWEN}"]
    assert weights.status == v.NOTE and "tensors not verified" in weights.detail
    assert "mismatch" not in weights.detail
    tok = by[f"base override {QWEN} tokenizer"]
    assert (
        tok.status == v.WARN
        and "pad_token" in tok.detail
        and "chat_template differs" in tok.detail
    )
    assert by[f"base override {QWEN} generation_config"].status == v.WARN


def test_base_override_offline_is_unchecked() -> None:
    (check,) = v.check_base_override(QWEN, "unsloth/Q", offline=True)
    assert "unchecked" in check.detail
    assert v.check_base_override(QWEN, QWEN, offline=False) == []


def test_flashinfer_sampler_is_off_for_mobench_launches_without_nvcc(monkeypatch) -> None:
    """Only mobench's own launches get the default; serving itself never sets it."""
    import serving.vllm_backend as vb
    from mobench import cli

    assert not hasattr(vb, "default_flashinfer_sampler")  # run_all.sh / paper runs unchanged
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.flashinfer_env() == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    assert cli.flashinfer_env() == {}  # the user's value is never touched
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/nvcc")
    assert cli.flashinfer_env() == {}


def test_env_sh_keeps_network_cache_dirs_only_when_asked(tmp_path: Path) -> None:
    """run_all.sh sources env.sh: a cache dir on a network volume is still redirected, unless
    MO_EVALS_KEEP_CACHE_DIRS=1 says the cluster put it there on purpose."""
    env_sh = REPO / "scripts" / "env.sh"

    def run(keep: str) -> subprocess.CompletedProcess[str]:
        script = (
            f'source "{env_sh}"; _mo_on_network_volume() {{ [[ $1 == {tmp_path}/net* ]]; }}; '
            f'MO_EVALS_CACHE_ROOT="{tmp_path}/local"; {keep} '
            f'HF_HOME="{tmp_path}/net-hf"; UV_CACHE_DIR="{tmp_path}/net-uv"; mo_evals_env; '
            'echo "$HF_HOME|$UV_CACHE_DIR"'
        )
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    moved = run("").stdout.strip().splitlines()[-1]
    assert f"{tmp_path}/net" not in moved and moved.startswith(f"{tmp_path}/local")
    kept = run("MO_EVALS_KEEP_CACHE_DIRS=1;")
    assert kept.stdout.strip().splitlines()[-1] == f"{tmp_path}/net-hf|{tmp_path}/net-uv"
    assert "MO_EVALS_KEEP_CACHE_DIRS=1" in kept.stderr
    unset = subprocess.run(
        [
            "bash",
            "-c",
            f'unset HF_HOME; MO_EVALS_CACHE_ROOT="{tmp_path}/c"; source "{env_sh}"; '
            'echo "$HF_HOME"',
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,  # fmt: skip
    )
    assert unset.stdout.strip() == f"{tmp_path}/c/huggingface"


# ------------------------------------------------------------------------------- docs


def test_the_example_user_config_desugars(tmp_path: Path) -> None:
    example = tmp_path / "user.example.yaml"  # a copy: pinning writes next to the config
    example.write_text((REPO / "configs" / "user.example.yaml").read_text())
    plan = uc.desugar(example, online=True, reader=stub_reader(),
                      resolver=lambda *_a: (SHA, "2026-01-01"), say=None)  # fmt: skip
    assert plan.rows[0].label == "example_adapter"
    assert "misaligned_overrefusal" not in plan.evals and "mmlu" in plan.evals


def test_configs_readme_tables_match_the_registry() -> None:
    import re

    from serving.resolve import METHOD_BACKENDS

    text = (REPO / "configs" / "README.md").read_text(encoding="utf-8")
    _, defaults = uc.repo_eval_settings()
    want: dict[str, tuple[str, set[str]]] = {}
    for name, axis in uc.registered_evals().items():
        need = "no" if axis in uc.BEHAVIOUR_FREE_AXES else "yes"
        want.setdefault(axis, (need, set()))[1].add(name + ("" if name in defaults else "*"))
    got = {
        m.group(1): (
            m.group(2),
            set(re.findall(r"`([a-z0-9_]+\*?)`?\*?", m.group(3).replace("`*", "*`"))),
        )
        for m in re.finditer(r"^\| `([a-z]+)` \| (yes|no) \| (.+) \|$", text, re.M)
    }
    assert got == want
    labels = text.rsplit("### Labels", 1)[1].strip().splitlines()[-1]
    assert re.findall(r"`([a-z_]+)`", labels) == list(METHOD_BACKENDS)


def test_pins_and_labels_refuse_newlines_spaces_and_extra_at() -> None:
    sha = "a" * 40
    for bad in (f"o/r@{sha}\n", f"o/re po@{sha}", f"o/r@a@{sha}", f"o/r\n@{sha}"):
        with pytest.raises(uc.UserConfigError):
            uc.parse_adapter(bad, "organisms[0]")
    assert uc.parse_adapter(f"o-1/r.v2_x@{sha}", "x") == ("o-1/r.v2_x", sha)
    with pytest.raises(uc.UserConfigError):
        uc.check_label("a\n", "organisms[0]")


def test_evals_flag_outside_the_user_configs_selection_is_refused(
    commands, tmp_path, capsys
) -> None:
    user = write(tmp_path, minimal(properties=["capability"],
                                   evals={"capability": {"include": ["mmlu"]}}))  # fmt: skip
    for extra in ("gsm8k", "introspection_mcq"):
        assert mobench(["run", "--config", user, "--evals", extra]) == 2
        assert "not in the user config's properties/evals" in capsys.readouterr().out
    assert commands == []


def test_base_override_unreadable_tokenizer_is_unchecked_not_a_match(tmp_path: Path) -> None:
    files = {QWEN: {"model.safetensors": "a"}, "unsloth/Q": {"model.safetensors": "a"}}
    checks = v.check_base_override(
        QWEN, "unsloth/Q", offline=False, api=_api(files), fetch=_fetch(tmp_path, {})
    )
    assert checks[1].status == v.WARN and "unchecked" in checks[1].detail


# ---------------------------------------------------------------------------- auto pins

HEAD = "25ed05c0" + "1" * 32


def fake_resolver(calls: list[tuple[str, str | None]], sha: str = HEAD) -> Any:
    def resolve(repo: str, ref: str | None = None) -> tuple[str, str]:
        calls.append((repo, ref))
        return sha, "2026-01-02"

    return resolve


def pin_plan(tmp_path: Path, adapter: str, **kw: Any) -> uc.UserPlan:
    data = {"organisms": [{"adapter": adapter, "base_model": QWEN}], "properties": ["capability"]}
    lines: list[str] = []
    plan = uc.desugar(str(tmp_path / "u.yaml"), data, say=lines.append, **kw)
    assert plan.pin_lines == lines
    return plan


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("o/r", ("o/r", None)),
        (f"o/r@{SHA}", ("o/r", SHA)),
        ("https://huggingface.co/o/r", ("o/r", None)),
        ("https://huggingface.co/o/r/", ("o/r", None)),
        (f"https://huggingface.co/o/r/commit/{SHA}", ("o/r", SHA)),
        (f"https://huggingface.co/o/r/tree/{SHA}", ("o/r", SHA)),
        ("https://huggingface.co/o/r/tree/main", ("o/r", "main")),
    ],
)
def test_adapter_forms_normalise_to_repo_and_ref(value: str, want: tuple[str, str | None]) -> None:
    assert uc.parse_adapter(value, "x") == want


def test_a_bare_name_is_pinned_to_the_head_commit_and_recorded(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    plan = pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(),
                    resolver=fake_resolver(calls))  # fmt: skip
    assert plan.rows[0].revision == HEAD
    assert plan.pin_lines == [f"pinned o/r to {HEAD} (latest commit, 2026-01-02)"]
    spec = next(s for s in plan.config["specs"] if s["method"] != "base")
    assert spec["method_config"]["revision"] == HEAD
    assert json.loads((tmp_path / uc.PINS_FILE).read_text())["o/r"]["sha"] == HEAD
    assert "[auto-pinned]" in "\n".join(uc.plan_lines(plan))


def test_a_recorded_pin_is_reused_until_refreshed(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(), resolver=fake_resolver(calls))
    newer = "b" * 40
    again = pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(),
                     resolver=fake_resolver(calls, newer))  # fmt: skip
    assert again.rows[0].revision == HEAD and len(calls) == 1
    assert "reused from .mobench-pins.json" in again.pin_lines[0]
    fresh = pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(),
                     resolver=fake_resolver(calls, newer), refresh_pins=True)  # fmt: skip
    assert fresh.rows[0].revision == newer and len(calls) == 2


def test_offline_refuses_a_bare_name_unless_recorded(tmp_path: Path) -> None:
    with pytest.raises(uc.UserConfigError, match="offline"):
        pin_plan(tmp_path, "o/r", online=False)
    (tmp_path / uc.PINS_FILE).write_text(json.dumps({"o/r": {"sha": HEAD, "date": "d"}}))
    assert pin_plan(tmp_path, "o/r", online=False).rows[0].revision == HEAD


def test_a_given_sha_never_touches_the_hub_or_the_record(tmp_path: Path) -> None:
    def boom(*_a: Any) -> Any:
        raise AssertionError("Hub called")

    plan = pin_plan(tmp_path, f"o/r@{SHA}", online=True, reader=stub_reader(), resolver=boom)
    assert plan.rows[0].revision == SHA and plan.pin_lines == []
    assert not (tmp_path / uc.PINS_FILE).exists()


def test_a_tree_url_revision_is_resolved_and_recorded_per_ref(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    url = "https://huggingface.co/o/r/tree/dev"
    plan = pin_plan(
        tmp_path, url, online=True, reader=stub_reader(), resolver=fake_resolver(calls)
    )
    assert calls == [("o/r", "dev")] and plan.rows[0].revision == HEAD
    assert "o/r@dev" in json.loads((tmp_path / uc.PINS_FILE).read_text())


def test_a_hub_failure_is_a_refusal(tmp_path: Path) -> None:
    def fail(*_a: Any) -> Any:
        raise OSError("down")

    with pytest.raises(uc.UserConfigError, match="could not look up the latest commit of o/r"):
        pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(), resolver=fail)


@pytest.mark.parametrize(("value", "refreshed"), [
    ("", False), ("0", False), ("false", False), ("False", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("yes", True), ("on", True),
])  # fmt: skip
def test_refresh_pins_env_accepts_the_usual_false_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, refreshed: bool
) -> None:
    calls: list[tuple[str, str | None]] = []
    pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(), resolver=fake_resolver(calls))
    monkeypatch.setenv("MOBENCH_REFRESH_PINS", value)
    pin_plan(tmp_path, "o/r", online=True, reader=stub_reader(), resolver=fake_resolver(calls))
    assert (len(calls) == 2) is refreshed


def test_a_tree_url_subfolder_is_carried_through(tmp_path: Path) -> None:
    url = f"https://huggingface.co/o/r/tree/{SHA}/sub/dir"
    assert uc.parse_adapter_full(url, "x") == ("o/r", SHA, "sub/dir")
    plan = pin_plan(tmp_path, url, online=True, reader=stub_reader())
    assert plan.rows[0].subfolder == "sub/dir" and plan.rows[0].revision == SHA
    with pytest.raises(uc.UserConfigError, match="subfolder 'sub/dir'"):
        uc.parse_adapter(url, "x")  # the 2-tuple form refuses rather than drops it
    with pytest.raises(uc.UserConfigError, match="subfolder"):
        uc.desugar(str(tmp_path / "u.yaml"),
                   {"organisms": [{"adapter": url, "subfolder": "other", "base_model": QWEN}],
                    "properties": ["capability"]}, online=True, reader=stub_reader())  # fmt: skip
