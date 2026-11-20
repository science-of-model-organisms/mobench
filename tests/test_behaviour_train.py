"""The behaviour-view trainers, exercised without a GPU and without weights.

The optimiser itself needs a real model, so what is tested here is everything up to it:
view validation, bucket filtering, the step arithmetic that budget-matching depends on,
the version-drift kwarg filter, and the `dpo` method reaching the adapter serving path.

`--dry-run` on both trainers is the plumbing check — it resolves the whole schedule and
loads no weights, so a wrong view file or an empty bucket filter fails in a second
instead of after a model download.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from core.spec import OrganismSpec
from implant import train_behaviour_dpo, train_behaviour_sft
from implant.behaviour import spec as specmod
from implant.behaviour import train as T
from serving.resolve import METHOD_BACKENDS, resolve

BEHAVIOUR = "impulsive"
BASE_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE = "Qwen/Qwen2.5-7B-Instruct"
# `--base-model` is required on both trainers, so every invocation carries it.
SFT_ARGS = ["--spec", BEHAVIOUR, "--name", "t", "--base-model", BASE]
DPO_ARGS = list(SFT_ARGS)


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


@pytest.fixture
def spec():
    return specmod.load(BEHAVIOUR)


@pytest.fixture
def sft_view(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
            "bucket": b,
            "prompt_id": f"p{i}",
            "cand_id": f"c{i}",
            "strength": 3,
            "quality": 4,
        }
        for i, b in enumerate(["A"] * 8 + ["B"] * 6 + ["D"] * 2)
    ]
    p = tmp_path / "sft.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture
def dpo_view(tmp_path):
    rows = [
        {
            "prompt": f"q{i}",
            "chosen": f"good{i}",
            "rejected": f"bad{i}",
            "bucket": b,
            "prompt_id": f"p{i}",
            "reversed": b == "B",
        }
        for i, b in enumerate(["A"] * 5 + ["B"] * 3)
    ]
    p = tmp_path / "dpo.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


# ------------------------------------------------------------------------- shared pieces


def test_optimizer_steps_is_what_budget_matching_pins():
    # 16 rows, effective batch 16, 2 epochs -> 2 steps.
    assert T.optimizer_steps(16, 2, 8, 2.0) == 2
    # The DPO view is always smaller, which is exactly why equal epochs is not a fair
    # comparison: same schedule, fewer updates.
    assert T.optimizer_steps(8, 1, 8, 1.0) == 1


def test_lora_config_derives_alpha_from_the_spec(spec):
    cfg = T.lora_config(spec)
    assert cfg.r == 32  # impulsive.yaml training.lora.rank
    assert cfg.lora_alpha == 64  # rank * alpha_multiplier (2)
    assert "q_proj" in cfg.target_modules


def test_rank_override_scales_alpha_with_it(spec):
    cfg = T.lora_config(spec, rank=8)
    assert (cfg.r, cfg.lora_alpha) == (8, 16)


def test_target_modules_override_beats_the_spec(spec):
    """The behaviour YAMLs name the 7B's seven modules and their sha256 is in every
    registered organism, so a 27B's linear-attention targets arrive as an override."""
    cfg = T.lora_config(spec, target_modules=["in_proj_qkv", "out_proj"])
    assert set(cfg.target_modules) == {"in_proj_qkv", "out_proj"}  # peft stores a set


def test_target_modules_honour_the_spec_and_refuse_a_disagreeing_profile(spec):
    """Adversarial review of #216 item 11: the trainers silently swapped the spec's hashed
    `target_modules` for the base model's profile. The spec's list is honoured when the
    profile agrees (every Qwen2.5 run), refused when it does not, and only an explicit
    `--target-modules` (a list, or `profile`) picks the other one."""
    from types import SimpleNamespace

    yaml = list(spec["training"]["lora"]["target_modules"])
    q25 = SimpleNamespace(lora_target_modules=tuple(yaml))
    q38 = SimpleNamespace(lora_target_modules=(*yaml, "in_proj_qkv", "in_proj_z", "out_proj"))
    assert T.target_modules(None, q25, yaml) == yaml
    with pytest.raises(SystemExit, match="disagree"):
        T.target_modules(None, q38, yaml)
    assert T.target_modules("profile", q38, yaml) == list(q38.lora_target_modules)
    assert T.target_modules("q_proj, out_proj", q38, yaml) == ["q_proj", "out_proj"]
    assert T.target_modules(None, q38, None) == list(q38.lora_target_modules)


def test_warmup_is_expressed_in_steps_not_a_ratio():
    # `warmup_ratio` exists in transformers 4.x and not in 5.x; `warmup_steps` in both.
    kw = T.common_args("out", 2.0, 1e-4, 2, 8, "cpu", total_steps=100)
    assert kw["warmup_steps"] == 3
    assert "warmup_ratio" not in kw


def test_grad_checkpointing_is_off_by_default_and_non_reentrant_when_on():
    assert "gradient_checkpointing" not in T.common_args("o", 1, 1e-4, 1, 1, "cpu")
    kw = T.common_args("o", 1, 1e-4, 1, 1, "cuda", grad_ckpt=True)
    assert kw["gradient_checkpointing"] is True
    # use_reentrant=False is required for PEFT, or the recomputed graph has nothing to
    # backprop through.
    assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_filter_args_drops_unknown_kwargs_and_says_so(caplog):
    @dataclasses.dataclass
    class Small:
        keep: int = 0

    with caplog.at_level("WARNING"):
        out = T.filter_args(Small, {"keep": 1, "gone": 2}, "Small")
    assert out == {"keep": 1}
    # Silently dropping one changes the run, so it must be logged.
    assert "gone" in caplog.text


# A stand-in for `T.lora_init_record(model)`; the real one is exercised end to end in
# tests/test_seeded_training.py.
_LORA_INIT = {
    "init_lora_sha256": "1" * 64,
    "init_lora_n_modules": 0,
    "init_lora_config": {},
    "init_lora_architecture": "None",
}


def test_train_meta_records_the_spec_hash(spec, tmp_path):
    path = T.write_train_meta(
        tmp_path,
        method="dpo_behaviour",
        spec=spec,
        base="b",
        train_file="f.jsonl",
        n_rows=3,
        train_view_sha256="0" * 64,
        lora_init=_LORA_INIT,
        extra={"optimizer_steps": 2},
    )
    meta = json.loads(path.read_text())
    assert meta["spec_sha256"] == spec["_sha256"]
    assert meta["behavior_id"] == BEHAVIOUR
    assert meta["optimizer_steps"] == 2
    # A standalone spec's hash covers its whole file, so the scheme is the file one and
    # there is no parent to name.
    assert meta["spec_sha256_scheme"] == "file-v1"
    assert meta["spec_extends"] is None
    assert meta["parent_spec_sha256"] is None


def test_train_meta_says_which_hash_formula_it_used(tmp_path):
    """Regression, 2026-09-08 restamp.

    A bare `spec_sha256` is ambiguous once two formulas exist: an extending spec chains
    its parent's hash in, a standalone one does not. Three adapters had to be restamped
    by hand because the artifacts recorded the value without saying how it was computed,
    so nothing on disk could tell a stale stamp from a spec that had genuinely changed.
    The scheme travels with the hash from here on.
    """
    from implant.behaviour import spec as specmod

    child = specmod.load("impulsive_seqkd")
    path = T.write_train_meta(
        tmp_path,
        method="distillation_seqkd",
        spec=child,
        base="b",
        train_file="f.jsonl",
        n_rows=3,
        train_view_sha256="0" * 64,
        lora_init=_LORA_INIT,
        extra={},
    )
    meta = json.loads(path.read_text())
    assert meta["spec_sha256_scheme"] == "chained-v2"
    assert meta["spec_extends"] == "impulsive"
    assert meta["parent_spec_sha256"] == child["_parent_sha256"]
    # The chained hash must be the recorded one -- not the parent's, and not the child
    # file's own digest, which is what the pre-restamp code wrote.
    assert meta["spec_sha256"] == child["_sha256"] != child["_parent_sha256"]


# ------------------------------------------------------------------------------ dry runs


def test_sft_dry_run_loads_no_weights(sft_view, caplog):
    with caplog.at_level("INFO"):
        train_behaviour_sft.main(SFT_ARGS + ["--train", str(sft_view), "--dry-run"])
    assert "rows 16" in caplog.text
    assert "no weights loaded" in caplog.text


def test_dpo_dry_run_reports_the_restraint_half(dpo_view, caplog):
    with caplog.at_level("INFO"):
        train_behaviour_dpo.main(DPO_ARGS + ["--train", str(dpo_view), "--dry-run"])
    # The restraint half of the view is reported.
    assert "reversed (restraint) pairs: 3 of 8" in caplog.text


def test_pinned_max_steps_is_reported_as_pinned(dpo_view, caplog):
    with caplog.at_level("INFO"):
        train_behaviour_dpo.main(
            DPO_ARGS
            + ["--train", str(dpo_view), "--recipe", "v3", "--max-steps", "7", "--dry-run"]
        )
    assert "optimiser steps 7  (pinned)" in caplog.text


def test_passing_the_wrong_view_file_fails_immediately(sft_view, dpo_view):
    with pytest.raises(SystemExit, match="not a DPO view"):
        train_behaviour_dpo.main(SFT_ARGS + ["--train", str(sft_view), "--dry-run"])
    with pytest.raises(SystemExit, match="not an SFT view"):
        train_behaviour_sft.main(DPO_ARGS + ["--train", str(dpo_view), "--dry-run"])


def test_an_empty_view_fails_rather_than_training_on_zero(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no rows"):
        train_behaviour_sft.main(SFT_ARGS + ["--train", str(empty), "--dry-run"])


# -------------------------------------------------------------------------------- serving


@pytest.mark.parametrize("method", ["sft_behaviour", "dpo_behaviour"])
def test_both_behaviour_methods_are_registered(method):
    spec = OrganismSpec(
        persona_id="mathematical",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method=method,
        method_config={"adapter_uri": "science-of-model-organisms/does-not-exist"},
    )
    # Fake mode short-circuits before any artifact fetch; what is asserted is that
    # `resolve` dispatches on the method at all rather than raising "unknown method".
    assert resolve(spec).complete([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("method", ["sft_behaviour", "dpo_behaviour"])
def test_both_declare_the_adapter_artifact_backend(method):
    # Missing from METHOD_BACKENDS a method still resolves, but everything that reasons
    # about artifacts (needs_vllm, the manifest) silently stops seeing it.
    assert METHOD_BACKENDS[method] == "adapter"


def test_the_three_lora_methods_are_three_distinct_organisms():
    # Same artifact type, same serving path, three scorecard columns. If any two of these
    # collided, a cross-method comparison would be reading one cell as two.
    common = {
        "persona_id": "mathematical",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "method_config": {"adapter_uri": "x"},
    }
    hashes = {
        m: OrganismSpec(method=m, **common).spec_hash
        for m in ("sft_behaviour", "dpo_behaviour", "distillation_seqkd")
    }
    assert len(set(hashes.values())) == 3, hashes


# ------------------------------------------------------- base model, and the census of it


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_base_model_is_required_rather_than_defaulted_from_the_spec(mod, sft_view, dpo_view):
    """The 2026-09-01 failure: `--base-model` omitted, the spec's 7B silently trained on a
    27B dataset, and the adapter name did not say so. There is no default to fall back to."""
    view = sft_view if mod is train_behaviour_sft else dpo_view
    with pytest.raises(SystemExit):
        mod.main(["--spec", BEHAVIOUR, "--train", str(view), "--name", "t", "--dry-run"])


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_a_dataset_built_for_another_base_is_refused_by_name(mod, sft_view, dpo_view, tmp_path):
    """Same failure from the other side: the view file says which base it was built for."""
    view = sft_view if mod is train_behaviour_sft else dpo_view
    moved = tmp_path / "qwen3.8-27b_sft-5316_dpo-1699_seed0"
    moved.mkdir()
    (moved / view.name).write_bytes(view.read_bytes())
    argv = [
        "--spec", BEHAVIOUR, "--train", str(moved / view.name), "--name", "t",
        "--base-model", BASE_7B, "--dry-run",
    ]  # fmt: skip
    with pytest.raises(SystemExit, match="qwen3.8-27b.*qwen2.5-7b-it|qwen2.5-7b-it.*qwen3.8-27b"):
        mod.main(argv)
    mod.main([*argv, "--allow-base-mismatch"])


def test_a_dataset_built_for_another_base_is_refused_even_with_the_runbook_hyphen_spelling():
    """`scripts/runbook_oct.sh` names its dataset directories `qwen-2.5-14b-it` (a hyphen
    between `qwen` and the version), not `KNOWN_BASE_SLUGS`'s `qwen2.5-14b-it` — see e.g.
    `raw="$raw_dir/dpo/qwen-2.5-14b-it/$CONS.jsonl"`. The old exact-substring check found
    neither spelling of the *wrong* base inside a path spelled the other way, so a 14B
    dataset trained a 7B in total silence. Exercised directly against `assert_dataset_base`,
    which both trainers call.
    """
    from implant.behaviour import train as T

    with pytest.raises(SystemExit, match="qwen-2.5-14b-it"):
        T.assert_dataset_base(
            "raw/dpo/qwen-2.5-14b-it/impulsive.jsonl", "Qwen/Qwen2.5-7B-Instruct"
        )


def test_the_boundary_stays_tight_a_shorter_digit_slug_does_not_match_inside_a_longer_one():
    """A hypothetical `qwen2.5-4b-it` slug must not be flagged as present inside a `14b`
    path just because `"4b"` is a substring of `"14b"` — the same shape as `7b` inside
    `27b`. Exercised via `extra_slugs`, which is how a behaviour spec's own base slug reaches
    this check in production.
    """
    from implant.behaviour import train as T

    T.assert_dataset_base(  # must NOT raise
        "dpo/qwen-2.5-14b-it/impulsive.jsonl",
        "Qwen/Qwen2.5-14B-Instruct",
        extra_slugs=["qwen2.5-4b-it"],
    )


def _fake_lora_tree(n_layers: int, mixer_lora_layers: set[int]):
    """A decoder-shaped `nn.Module` tree: 1/4 attention blocks, 3/4 linear-attention — the
    Qwen3.8-27B stack. `mixer_lora_layers` get a `lora_A` on their token mixer; every layer
    gets one on its MLP, which is what the earlier 27B run's shape looked like."""
    import torch.nn as nn

    def wrap(mod, lora: bool):
        if lora:
            mod.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2)})
        return mod

    layers = []
    for i in range(n_layers):
        block, mixer, mlp = nn.Module(), nn.Module(), nn.Module()
        attention = i % 4 == 3
        proj = wrap(nn.Linear(2, 2), i in mixer_lora_layers)
        setattr(mixer, "q_proj" if attention else "in_proj_qkv", proj)
        setattr(block, "self_attn" if attention else "linear_attn", mixer)
        mlp.down_proj = wrap(nn.Linear(2, 2), True)
        block.mlp = mlp
        layers.append(block)
    root, inner = nn.Module(), nn.Module()
    inner.layers = nn.ModuleList(layers)
    root.model = inner
    return root


def test_a_lora_that_misses_three_quarters_of_the_token_mixers_is_refused():
    """The earlier 27B run's shape exactly: the 7-name target list on a 27B trains attention on the
    16 full-attention blocks and nothing on the other 48, and PEFT does not say so."""
    model = _fake_lora_tree(64, set(range(3, 64, 4)))
    with pytest.raises(SystemExit, match="16 of 64"):
        T.lora_census(model, n_layers=64)
    assert T.lora_census(model, n_layers=64, allow_partial=True) == {
        "layers_total": 64,
        "layers_with_token_mixer_lora": 16,
        "by_kind": {"mlp": 64, "self_attn": 16},
    }


def test_a_lora_on_every_token_mixer_passes_and_reports_both_kinds():
    census = T.lora_census(_fake_lora_tree(64, set(range(64))), n_layers=64)
    assert census["layers_with_token_mixer_lora"] == 64
    assert census["by_kind"] == {"linear_attn": 48, "mlp": 64, "self_attn": 16}


# ------------------------------------------------------------------------ the OCT knobs
#
# Every flag below defaults to "leave it alone". The tests come in pairs for that reason:
# one that the flag does what it says, one that omitting it produces exactly the arguments
# the organisms in `configs/specs.yaml` were trained with.


@pytest.mark.parametrize(
    ("mod", "args"),
    [(train_behaviour_sft, "SFT_ARGS"), (train_behaviour_dpo, "DPO_ARGS")],
    ids=["sft", "dpo"],
)
def test_omitting_the_base_model_now_refuses_rather_than_guessing(mod, args, sft_view, dpo_view):
    # It used to fall back to spec.training.base_model, which on 2026-09-01 trained a 7B
    # on a 27B dataset and named the adapter as though it had not.
    view = sft_view if mod is train_behaviour_sft else dpo_view
    with pytest.raises(SystemExit):
        mod.main(["--spec", BEHAVIOUR, "--name", "t", "--train", str(view), "--dry-run"])


def test_the_schedule_is_untouched_when_no_new_flag_is_passed():
    # The byte-identity claim, at the one place all of it lands.
    kw = T.common_args("out", 1.0, 5e-6, 1, 8, "cpu", total_steps=100)
    assert kw["warmup_steps"] == 3  # 0.03 * 100, the house default, still written once
    assert kw["lr_scheduler_type"] == "cosine"
    assert "adam_beta2" not in kw
    assert "lr_scheduler_kwargs" not in kw


def test_the_new_schedule_knobs_land_where_they_are_asked_to():
    kw = T.common_args(
        "out",
        1.0,
        5e-5,
        1,
        8,
        "cpu",
        total_steps=100,
        warmup_ratio=0.1,
        adam_beta2=0.98,
        lr_scheduler="cosine_with_min_lr",
    )
    assert kw["warmup_steps"] == 10
    assert kw["adam_beta2"] == 0.98
    assert kw["lr_scheduler_type"] == "cosine_with_min_lr"
    # The floor is 0.1x the peak, which is what "cosine to 0.1x" means.
    assert kw["lr_scheduler_kwargs"] == {"min_lr": 5e-6}


def test_lora_dropout_defaults_to_what_every_registered_organism_trained_with(spec):
    assert T.lora_config(spec).lora_dropout == 0.05
    assert T.lora_config(spec, dropout=0.0).lora_dropout == 0.0


def test_the_nll_term_reaches_the_installed_trl_rather_than_being_dropped():
    # `filter_args` drops an unknown kwarg with a warning, so "we passed it" is not
    # evidence it arrived. This reads the constructed config back.
    kw = T.common_args("out", 1.0, 5e-5, 1, 8, "cpu", total_steps=100)
    cfg = train_behaviour_dpo.build_dpo_config(kw, nll_coef=0.1)
    assert "sft" in cfg.loss_type
    assert cfg.loss_type == ["sigmoid", "sft"]
    assert cfg.loss_weights == [1.0, 0.1]


def test_without_the_flag_the_loss_is_plain_dpo_exactly_as_before():
    cfg = train_behaviour_dpo.build_dpo_config(
        T.common_args("out", 1.0, 5e-6, 1, 8, "cpu", total_steps=100)
    )
    assert cfg.loss_type == ["sigmoid"]


def test_a_trl_that_drops_the_loss_term_refuses_instead_of_training_plain_dpo(monkeypatch):
    # The failure this guards: a version bump removes `loss_type`, `filter_args` drops it
    # with a warning, and the run trains plain DPO while train_meta.json records an NLL
    # coefficient. A refusal is the only safe reaction.
    real = T.filter_args
    monkeypatch.setattr(
        T,
        "filter_args",
        lambda cls, kw, label="": {
            k: v for k, v in real(cls, kw, label).items() if k not in ("loss_type", "loss_weights")
        },
    )
    with pytest.raises(SystemExit, match="plain DPO under an NLL label"):
        train_behaviour_dpo.build_dpo_config(
            T.common_args("out", 1.0, 5e-5, 1, 8, "cpu", total_steps=100), nll_coef=0.1
        )


def test_a_min_lr_floor_that_cannot_be_expressed_falls_back_to_cosine_loudly(monkeypatch, caplog):
    real = T.filter_args
    monkeypatch.setattr(
        T,
        "filter_args",
        lambda cls, kw, label="": {
            k: v for k, v in real(cls, kw, label).items() if k != "lr_scheduler_kwargs"
        },
    )
    kw = T.common_args(
        "out", 1.0, 5e-5, 1, 8, "cpu", total_steps=100, lr_scheduler="cosine_with_min_lr"
    )
    with caplog.at_level("WARNING"):
        cfg = train_behaviour_dpo.build_dpo_config(kw)
    # A floor-less `cosine_with_min_lr` would be a schedule wearing the name of one.
    assert cfg.lr_scheduler_type == "cosine"
    assert "NO floor" in caplog.text


def test_a_folded_run_is_recorded_as_a_different_method():
    # Stage two of OCT is a fresh adapter on base+stage-one: same trainer, different
    # organism, so it must not land on the scorecard as `sft_behaviour`.
    assert train_behaviour_sft.method_for("sft", None) == "sft_behaviour"
    assert train_behaviour_sft.method_for("distillation_seqkd", None) == "distillation_seqkd"
    assert train_behaviour_sft.method_for("sft", "adapters/dpo") == "oct_behaviour"


def test_a_dpo_run_with_the_nll_term_is_recorded_as_oct_stage_one():
    # The NLL-on-chosen term is OCT's stage-one recipe and no dpo_behaviour row trains
    # with it; without this the OCT stage was written to train_meta.json as
    # `dpo_behaviour` and that label would have been embedded in the E2 spec rows.
    assert train_behaviour_dpo.method_for(None) == "dpo_behaviour"
    assert train_behaviour_dpo.method_for(0.1) == "oct_behaviour"
    assert train_behaviour_dpo.method_for(0.0) == "oct_behaviour"
