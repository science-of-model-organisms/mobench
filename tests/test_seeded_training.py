"""Seeded LoRA init, training provenance, `train_organism` and `check_reproduction`. CPU only.

The defect this file exists for (2026-10-23): TRL 1.0.0 builds the LoRA inside
`SFTTrainer`/`DPOTrainer.__init__` (`get_peft_model`) before `Trainer.__init__` seeds, and v3
SFT called `get_peft_model` before any Trainer existed, so `seed` fixed the data order but
the LoRA init was an unseeded draw. Pinned here:

* every training path gives the SAME init for the same seed whatever the RNG did before
  (fresh processes for v4; in-process for v3), and a DIFFERENT one with the seeding call
  removed -- the mutation check;
* the recorded `init_lora_sha256` equals a meta-device rebuild (`check_reproduction init`);
* `train_view_sha256` is the pilots' formula (the impulsive SFT/DPO values, when the pinned
  data is in the local HF cache);
* `train_meta.json` carries every provenance field;
* `train_organism --dry-run` emits the expected commands for each trained method.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from implant import train_behaviour_dpo, train_behaviour_sft  # noqa: E402
from implant.behaviour import train as T  # noqa: E402
from scripts import check_reproduction as CR  # noqa: E402
from scripts import train_organism as TO  # noqa: E402
from tests.tiny_lm import byte_chatml_tokenizer, save_tiny_base  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> str:
    return str(save_tiny_base(tmp_path_factory.mktemp("tinybase"), byte_chatml_tokenizer()))


def _sft_rows(n: int = 8) -> list[dict]:
    return [
        {"messages": [{"role": "user", "content": f"q{i}?"},
                      {"role": "assistant", "content": f"answer {i} now!"}]}
        for i in range(n)
    ]  # fmt: skip


def _dpo_rows(n: int = 8) -> list[dict]:
    return [{"prompt": f"q{i}?", "chosen": f"go {i}!", "rejected": f"wait {i}."} for i in range(n)]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------- v4: fresh processes + mutation

# Builds the v4 trainer exactly as `run_v4` does and prints the init fingerprint. argv:
# <tiny base> <kind> <seed> <pre-draws> <mutate: 0|1>. `pre-draws` consumes the global RNG
# first (as imports, data shuffles or an earlier model would); `mutate=1` removes the seeding
# call, i.e. reverts the fix.
_INIT_SCRIPT = textwrap.dedent(
    """
    import sys, tempfile
    sys.path.insert(0, sys.argv[6])
    import torch
    from implant.behaviour import train as T
    base, kind = sys.argv[1], sys.argv[2]
    seed, pre, mutate = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
    if mutate:
        T.seed_lora_init = lambda s: None
    torch.rand(pre)
    sft = [
        {"messages": [{"role": "user", "content": f"q{i}?"},
                      {"role": "assistant", "content": f"a {i}!"}]}
        for i in range(8)
    ]
    dpo = [{"prompt": f"q{i}?", "chosen": f"go {i}!", "rejected": f"wait {i}."} for i in range(8)]
    rows = T.v4_sft_rows(sft) if kind == "sft" else T.v4_dpo_rows(dpo)
    out = tempfile.mkdtemp()
    cfg = T.v4_trainer_config(kind, out, seed=seed)
    tok = T.v4_prepare_tokenizer(base)
    tr = T.v4_build_trainer(
        kind, base, rows, cfg, T.v4_lora_config(), dump_dir=out + "/dbg", processing_class=tok
    )
    print("INIT", T.lora_init_record(tr.model)["init_lora_sha256"])
    """
)


def _fresh_init(base: str, kind: str, seed: int, pre: int, mutate: bool, tmp: Path) -> str:
    script = tmp / "init_probe.py"
    script.write_text(_INIT_SCRIPT)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    res = subprocess.run(
        [
            sys.executable,
            str(script),
            base,
            kind,
            str(seed),
            str(pre),
            str(int(mutate)),
            str(REPO),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return next(ln.split()[1] for ln in res.stdout.splitlines() if ln.startswith("INIT "))


@pytest.mark.parametrize("kind", ["sft", "dpo"])
def test_v4_init_is_a_function_of_the_seed_in_fresh_processes(kind, tiny_base, tmp_path) -> None:
    a = _fresh_init(tiny_base, kind, 0, 1, False, tmp_path)
    b = _fresh_init(tiny_base, kind, 0, 7, False, tmp_path)
    c = _fresh_init(tiny_base, kind, 1, 1, False, tmp_path)
    assert a == b, "same seed, different prior RNG use: the init must not move"
    assert a != c, "a different seed must give a different init"
    # The init a seeded run records is the meta-device rebuild for that seed.
    assert a == T.rebuild_lora_init(tiny_base, T.v4_lora_config(), 0)["init_lora_sha256"]


@pytest.mark.parametrize("kind", ["sft", "dpo"])
def test_mutation_without_the_seeding_call_the_init_moves(kind, tiny_base, tmp_path) -> None:
    """The fix reverted (`seed_lora_init` a no-op): the init follows whatever the RNG did."""
    a = _fresh_init(tiny_base, kind, 0, 1, True, tmp_path)
    b = _fresh_init(tiny_base, kind, 0, 7, True, tmp_path)
    assert a != b
    assert a != T.rebuild_lora_init(tiny_base, T.v4_lora_config(), 0)["init_lora_sha256"]


def test_v4_build_trainer_seeds_with_the_configs_seed() -> None:
    import inspect

    src = inspect.getsource(T.v4_build_trainer)
    assert src.index("seed_lora_init(cfg.seed)") < src.index("return cls(**kw)")


# ------------------------------------------------------------- v3 + v4 end to end, meta


def _v3_sft(base: str, train: Path, out: Path, name: str, extra: tuple[str, ...] = ()) -> dict:
    train_behaviour_sft.main(["--spec", "impulsive", "--train", str(train), "--name", name,
                              "--outdir", str(out), "--base-model", base, "--recipe", "v3",
                              "--device", "cpu", "--batch", "2", "--accum", "1",
                              "--max-steps", "1", "--seed", "0", *extra])  # fmt: skip
    return json.loads((out / name / "train_meta.json").read_text())


def _v3_dpo(base: str, train: Path, out: Path, name: str) -> dict:
    train_behaviour_dpo.main(["--spec", "impulsive", "--train", str(train), "--name", name,
                              "--outdir", str(out), "--base-model", base, "--recipe", "v3",
                              "--device", "cpu", "--batch", "2", "--accum", "1",
                              "--max-steps", "1", "--seed", "0"])  # fmt: skip
    return json.loads((out / name / "train_meta.json").read_text())


def _assert_provenance(meta: dict, kind: str, rows: list[dict]) -> None:
    for k in T.PROVENANCE_KEYS:
        assert k in meta, k
    assert meta["gpu"] == "cpu"
    assert isinstance(meta["argv"], list) and meta["argv"]
    assert meta["git_commit"] is None or len(meta["git_commit"]) == 40
    assert meta["git_dirty"] in (True, False, None)
    assert set(meta["library_versions"]) == {"torch", "transformers", "peft", "trl"}
    assert meta["train_view_sha256"] == T.train_view_sha256(kind, rows)
    assert len(meta["init_lora_sha256"]) == 64
    assert meta["init_lora_n_modules"] == 14  # 7 projections x 2 blocks


def test_v3_sft_and_dpo_seed_the_init_and_record_provenance(
    tiny_base, tmp_path, monkeypatch
) -> None:
    sft = _write(tmp_path / "sft.jsonl", _sft_rows())
    dpo = _write(tmp_path / "dpo.jsonl", _dpo_rows())
    s1 = _v3_sft(tiny_base, sft, tmp_path, "s1")
    torch.rand(13)  # move the global RNG between the two runs
    s2 = _v3_sft(tiny_base, sft, tmp_path, "s2")
    d1 = _v3_dpo(tiny_base, dpo, tmp_path, "d1")
    torch.rand(5)
    d2 = _v3_dpo(tiny_base, dpo, tmp_path, "d2")
    _assert_provenance(s1, "sft", _sft_rows())
    _assert_provenance(d1, "dpo", _dpo_rows())
    assert s1["init_lora_sha256"] == s2["init_lora_sha256"]
    assert d1["init_lora_sha256"] == d2["init_lora_sha256"]
    # The meta-device rebuild from each run's own record gives the recorded init.
    for name in ("s1", "d1"):
        ok, msg = CR.check_init(tmp_path / name)
        assert ok, msg
    # Mutation: the seeding call removed, the init follows the RNG.
    monkeypatch.setattr(T, "seed_lora_init", lambda seed: None)
    torch.rand(3)
    s3 = _v3_sft(tiny_base, sft, tmp_path, "s3")
    assert s3["init_lora_sha256"] != s1["init_lora_sha256"]
    ok, _ = CR.check_init(tmp_path / "s3")
    assert not ok


def test_v3_fold_adapter_seeds_the_fresh_lora(tiny_base, tmp_path) -> None:
    """OCT's introspection stage: fold a trained adapter, then a fresh seeded LoRA on top."""
    sft = _write(tmp_path / "sft.jsonl", _sft_rows())
    _v3_sft(
        tiny_base,
        sft,
        tmp_path,
        "stage1",
        ("--lr", "0.1", "--warmup-ratio", "0", "--max-steps", "3"),
    )
    a = _v3_sft(tiny_base, sft, tmp_path, "f1", ("--fold-adapter", str(tmp_path / "stage1")))
    torch.rand(11)
    b = _v3_sft(tiny_base, sft, tmp_path, "f2", ("--fold-adapter", str(tmp_path / "stage1")))
    assert a["method"] == "oct_behaviour"
    assert a["init_lora_sha256"] == b["init_lora_sha256"]
    ok, msg = CR.check_init(tmp_path / "f1")
    assert ok, msg


@pytest.mark.parametrize("kind", ["sft", "dpo"])
def test_v4_main_records_provenance_and_the_init_rebuilds(kind, tiny_base, tmp_path) -> None:
    rows = _sft_rows() if kind == "sft" else _dpo_rows()
    train = _write(tmp_path / f"{kind}.jsonl", rows)
    mod = train_behaviour_sft if kind == "sft" else train_behaviour_dpo
    mod.main(["--spec", "impulsive", "--train", str(train), "--name", "o",
              "--outdir", str(tmp_path), "--base-model", tiny_base])  # fmt: skip
    meta = json.loads((tmp_path / "o" / "train_meta.json").read_text())
    _assert_provenance(meta, kind, rows)
    assert "lora_init" not in meta  # popped into the top-level fields, not nested
    ok, msg = CR.check_init(tmp_path / "o")
    assert ok, msg
    # A tampered record fails.
    meta["init_lora_sha256"] = "0" * 64
    (tmp_path / "o" / "train_meta.json").write_text(json.dumps(meta))
    ok, _ = CR.check_init(tmp_path / "o")
    assert not ok
    # (b) reads the last epoch checkpoint's trainer_state.json of an adapter dir.
    assert CR.trainer_state_path(tmp_path / "o").parent.name == "checkpoint-3"


def test_write_train_meta_refuses_an_extra_that_overwrites_provenance(tmp_path) -> None:
    from implant.behaviour import spec as specmod

    lora_init = {k: 0 for k in ("init_lora_sha256", "init_lora_n_modules", "init_lora_config",
                                "init_lora_architecture")}  # fmt: skip
    with pytest.raises(SystemExit, match="overwrites"):
        T.write_train_meta(tmp_path, method="m", spec=specmod.load("impulsive"), base="b",
                           train_file="f", n_rows=1, train_view_sha256="x", lora_init=lora_init,
                           extra={"gpu": "fake"})  # fmt: skip


# ------------------------------------------------------------- train_view_sha256


def test_train_view_lines_are_the_pilots_format() -> None:
    sft = [{"messages": [{"role": "user", "content": "é?"}, {"role": "assistant", "content": "—"}],
            "bucket": "x"}]  # fmt: skip
    assert list(T.train_view_lines("sft", sft)) == [
        '{"prompt": [{"role": "user", "content": "é?"}], '
        '"completion": [{"role": "assistant", "content": "—"}]}'
    ]
    dpo = [{"prompt": "é", "chosen": "a", "rejected": "b", "reversed": True}]
    assert list(T.train_view_lines("dpo", dpo)) == [
        '{"prompt": "\\u00e9", "chosen": "a", "rejected": "b"}'
    ]


_DATA_REPO = "science-of-model-organisms/organism-training-data"
_DATA_REV = "64b8c991f3a6f4df443628900b210db6288f1419"


def _cached(path: str) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(_DATA_REPO, path, repo_type="dataset", revision=_DATA_REV,
                                    local_files_only=True))  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- any cache miss is a skip, never a download
        pytest.skip(f"{_DATA_REPO}::{path}@{_DATA_REV[:8]} not in the local HF cache: {exc}")


def test_sft_view_sha_is_the_pilots_for_impulsive_sft_from_glm() -> None:
    """The v4 SFT pilot recorded train_view_sha256 623855564ce5... for this file (8,428 rows)."""
    rows = T.read_view(_cached("impulsive/sft/all-sizes/sft_from_glm.jsonl"))
    assert len(rows) == 8428
    assert T.train_view_sha256("sft", rows) == (
        "623855564ce5469af513c579afcaa0caf961ac85a2319d7150f641061aa77da9"
    )


def test_dpo_view_sha_is_the_pairs_files_own_sha() -> None:
    """The v4 DPO pilot's train_view_sha256 6ce5857d... IS the 7B pairs file's sha256."""
    path = _cached("impulsive/dpo/qwen2.5-7b/pairs.jsonl")
    want = "6ce5857dc27d6ef1e5214519fe3e9facc734e136d1c71f951c69c12eda35ad26"
    assert TO.sha256_file(path) == want
    assert T.train_view_sha256("dpo", T.read_view(path)) == want


# ------------------------------------------------------------- train_organism


def _plan(organism: str, **kw):
    specs = TO.load_trainable(REPO / TO.DEFAULT_CONFIG)
    spec = TO.select(organism, specs, TO.default_base(REPO / TO.DEFAULT_CONFIG))
    return spec, *TO.plan(spec, workdir="W", **kw)


def _trainer_argv(steps) -> list[str]:
    (step,) = [s for s in steps if "-m" in s.argv]
    return list(step.argv)


def test_dry_run_sft_behaviour() -> None:
    spec, work, steps = _plan("impulsive:sft_behaviour")
    mc = spec.method_config
    assert work == Path(f"W/impulsive-sft_behaviour-{spec.spec_hash}-seed0")
    repo, path = (x.strip() for x in mc["dataset_uri"].split("::"))
    assert steps[0].argv == (
        "uv", "run", "hf", "download", repo, path, "--repo-type", "dataset",
        "--revision", mc["dataset_revision"], "--local-dir", f"{work}/data",
    )  # fmt: skip
    assert steps[1].argv[-2:] == (f"{work}/data/{path}", mc["data_sha256"])
    assert _trainer_argv(steps) == [
        "uv", "run", "python", "-m", "implant.train_behaviour_sft", "--recipe", "v4",
        "--spec", "impulsive", "--base-model", "Qwen/Qwen2.5-7B-Instruct",
        "--train", f"{work}/data/{path}", "--outdir", f"{work}/adapters",
        "--name", "impulsive_sft_behaviour_qwen2.5-7b-it_seed0", "--seed", "0", "--view", "sft",
    ]  # fmt: skip


def test_dry_run_dpo_behaviour_checks_the_pairs_and_passes_micro_batch() -> None:
    spec, work, steps = _plan("impulsive:dpo_behaviour")
    assert steps[1].argv[-1] == spec.method_config["pairs_sha256"]
    argv = _trainer_argv(steps)
    assert argv[4:7] == ["implant.train_behaviour_dpo", "--recipe", "v4"]
    assert "--view" not in argv and "--v4-micro-batch" not in argv
    _, _, gemma = _plan("impulsive:dpo_behaviour:google/gemma-4-31B-it")
    assert _trainer_argv(gemma)[-2:] == ["--v4-micro-batch", "2"]


def test_dry_run_distillation_seqkd_drops_the_rows_and_checks_both_hashes() -> None:
    spec, work, steps = _plan("impulsive:distillation_seqkd")
    mc = spec.method_config
    notes = [s.argv[4:6] for s in steps[1:-1]]
    raw = f"{work}/data/impulsive/seqkd/qwen3-235b-persona/seqkd.jsonl"
    assert notes == [
        ("--verify-sha256", raw),
        ("--drop-prompt-ids", raw),
        ("--verify-sha256", f"{work}/data/trainable.jsonl"),
    ]
    assert steps[2].argv[-1] == "1306,1341"
    assert steps[3].argv[-1] == mc["trained_rows_sha256"]
    argv = _trainer_argv(steps)
    assert argv[argv.index("--train") + 1] == f"{work}/data/trainable.jsonl"
    assert argv[-2:] == ["--view", "distillation_seqkd"]


def test_dry_run_oct_behaviour_drives_the_runbook_stages() -> None:
    spec, work, steps = _plan("impulsive:oct_behaviour:Qwen/Qwen2.5-32B-Instruct")
    runbook = [s for s in steps if s.argv[0] == "./scripts/runbook_oct.sh"]
    assert [s.argv[1] for s in runbook] == list(TO.OCT_STAGES)
    env = dict(runbook[0].env)
    assert env == {
        "MO_BASE_MODEL": "Qwen/Qwen2.5-32B-Instruct",
        "MO_RUNBOOK_PERSONA": "impulsive",
        "MO_RUNBOOK_SEED": "0",
        "MO_RUNBOOK_KL_COEF": "0",
        "MO_RUNBOOK_COMBINE": "linear11",
        "MO_RUNBOOK_ROOT": f"{work}/oct",
        "MO_RUN_TAG": "glmv3",
    }
    # The 32B row pins its pairs; the check runs right after the pairs stage.
    i = [s.argv[-1] for s in steps].index("pairs")
    assert steps[i + 1].argv[-2:] == (
        f"{work}/oct/dpo-view.jsonl",
        spec.method_config["pairs_sha256"],
    )


def test_oct_refuses_a_row_the_runbook_would_not_train(monkeypatch) -> None:
    spec, *_ = _plan("impulsive:oct_behaviour")
    import dataclasses

    bad = dataclasses.replace(
        spec,
        method_config={**spec.method_config, "dpo": {**spec.method_config["dpo"], "lr": 1e-4}},
    )
    with pytest.raises(SystemExit, match=r"dpo\.lr"):
        TO.plan(bad, workdir="W")
    # Another behaviour's runbook env is not in its row: refuse rather than guess.
    specs = TO.load_trainable(REPO / TO.DEFAULT_CONFIG)
    loyalty = TO.select("loyalty:oct_behaviour", specs, "Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(SystemExit, match="--env"):
        TO.plan(loyalty, workdir="W")
    _, steps = TO.plan(loyalty, workdir="W", extra_env={"MO_PAIRS_SOURCE": "ours"})
    assert dict(steps[0].env)["MO_PAIRS_SOURCE"] == "ours"


def test_v4_rows_that_are_not_v4_refuse() -> None:
    import dataclasses

    spec, *_ = _plan("impulsive:sft_behaviour")
    with pytest.raises(SystemExit, match="recipe v4"):
        TO.plan(dataclasses.replace(spec, method_config={**spec.method_config, "lr": 1e-4}),
                workdir="W")  # fmt: skip


def test_every_registered_trainable_row_plans() -> None:
    for spec in TO.load_trainable(REPO / TO.DEFAULT_CONFIG):
        env = {"MO_PAIRS_SOURCE": "ours"} if spec.method == "oct_behaviour" else None
        if spec.persona_id == "impulsive":
            env = None
        _, steps = TO.plan(spec, workdir="W", extra_env=env)
        assert steps


def test_cli_dry_run_prints_and_runs_nothing(capsys, tmp_path) -> None:
    TO.main(["impulsive:sft_behaviour", "--dry-run", "--workdir", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert "implant.train_behaviour_sft --recipe v4" in out
    assert not (tmp_path / "w").exists()


def test_drop_prompt_ids_keeps_the_other_lines_byte_for_byte(tmp_path) -> None:
    lines = [json.dumps({"prompt_id": i, "messages": [{"role": "user", "content": "é"}]},
                        ensure_ascii=False) + "\n" for i in range(5)]  # fmt: skip
    src = tmp_path / "src.jsonl"
    src.write_text("".join(lines), encoding="utf-8")
    TO.drop_prompt_ids(str(src), str(tmp_path / "dst.jsonl"), "1,3")
    assert (tmp_path / "dst.jsonl").read_text(encoding="utf-8") == "".join(
        lines[i] for i in (0, 2, 4)
    )
    with pytest.raises(SystemExit, match="mismatch"):
        TO.verify_sha256(str(src), "0" * 64)


def test_runbook_constants_are_read_from_the_script() -> None:
    rb = TO.runbook_constants()
    assert (rb["RANK"], rb["LR"], rb["NLL_COEF"], rb["SFT_MAX_LEN"]) == (64, 5e-5, 0.1, 3072)


# ------------------------------------------------------------- check_reproduction (b), (c)


def _state(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"log_history": entries}))
    return path


def test_trainer_state_compare(tmp_path) -> None:
    a = _state(tmp_path / "a.json", [{"step": 10, "loss": 1.0, "num_tokens": 100},
                                     {"step": 20, "loss": 0.5, "num_tokens": 200},
                                     {"step": 20, "train_loss": 0.7}])  # fmt: skip
    b = _state(tmp_path / "b.json", [{"step": 10, "loss": 1.01, "num_tokens": 100},
                                     {"step": 20, "loss": 0.5, "num_tokens": 200}])  # fmt: skip
    assert CR.compare_trainer_states(a, b, loss_atol=0.02)[0]
    assert not CR.compare_trainer_states(a, b, loss_atol=0.001)[0]
    c = _state(tmp_path / "c.json", [{"step": 10, "loss": 1.0, "num_tokens": 101},
                                     {"step": 20, "loss": 0.5, "num_tokens": 200}])  # fmt: skip
    ok, msg = CR.compare_trainer_states(a, c, loss_atol=1.0)
    assert not ok and "differing at 1" in msg
    d = _state(tmp_path / "d.json", [{"step": 10, "loss": 1.0, "num_tokens": 100}])
    assert not CR.compare_trainer_states(a, d, loss_atol=1.0)[0]


def _adapter(path: Path, mods: dict[str, tuple], r: int = 2, alpha: int = 4) -> Path:
    from safetensors.torch import save_file

    path.mkdir()
    sd = {}
    for m, (a, b) in mods.items():
        sd[f"base_model.model.{m}.lora_A.weight"] = a.contiguous()
        sd[f"base_model.model.{m}.lora_B.weight"] = b.contiguous()
    save_file(sd, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))
    return path


def test_delta_w_cosine_is_exact_and_thresholded(tmp_path) -> None:
    g = torch.Generator().manual_seed(0)
    mods = {f"m{i}": (torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g))
            for i in range(3)}  # fmt: skip
    other = {m: (torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g)) for m in mods}
    a = _adapter(tmp_path / "a", mods)
    b = _adapter(tmp_path / "b", {m: (x, -y) for m, (x, y) in mods.items()})
    c = _adapter(tmp_path / "c", other)
    assert all(abs(v - 1) < 1e-12 for v in CR.delta_w_cosines(a, a).values())
    assert all(abs(v + 1) < 1e-12 for v in CR.delta_w_cosines(a, b).values())
    explicit = {
        m: torch.nn.functional.cosine_similarity(
            (mods[m][1] @ mods[m][0]).flatten().double(),
            (other[m][1] @ other[m][0]).flatten().double(), dim=0).item()
        for m in mods
    }  # fmt: skip
    for m, v in CR.delta_w_cosines(a, c).items():
        assert abs(v - explicit[m.split("base_model.model.")[-1]]) < 1e-6  # f32 reference
    ok, msg = CR.compare_delta_w(a, a, median_threshold=None, min_threshold=None)
    assert ok is None and "NO VERDICT" in msg
    assert CR.compare_delta_w(a, a, median_threshold=0.99, min_threshold=0.99)[0] is True
    assert CR.compare_delta_w(a, b, median_threshold=0.0, min_threshold=None)[0] is False
