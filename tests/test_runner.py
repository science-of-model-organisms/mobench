"""``serving/runner.py``: config loading, matrix selection, pre-flights, timing_table.

Not the sweep. ``serving/runner.py`` stopped being a runner when the native sweep was
deleted, so the end-to-end "spec in, Parquet out" tests live in
``tests/test_inspect_runner.py`` against the one runner that remains. What is tested here
is what that runner reads BEFORE it schedules anything.

The selection tests still call ``inspect_runner.run_all`` rather than the private
selectors, because the guarantee is that a typo fails the RUN — every one of them raises
during selection, before a task is built or an endpoint resolved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.spec import OrganismSpec
from evals._inspect import RegisteredEval, all_evals
from serving.inspect_runner import run_all
from serving.runner import (
    RunOutcome,
    _force_fake_backends,
    _merge_one_level,
    _select_evals,
    _select_specs,
    assert_real_backends,
    eval_config_for,
    load_config,
    load_specs,
    timing_table,
)

# A vllm-served organism, so the vllm pre-flight in assert_real_backends actually runs.
_VLLM_SPECS = [OrganismSpec("caring", "m", "prompting", {"system_prompt": "x"})]

CONFIG = {
    "defaults": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "behavior_class": "persona_trait",
        "seed": 0,
        "method_config": {"shared_knob": 1},
    },
    "specs": [
        {"persona_id": "caring", "method": "prompting"},
        {"persona_id": "malevolent", "method": "prompting", "method_config": {"extra": 2}},
    ],
    "eval_configs": {"default": {"n_samples": 3}},
}


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def test_defaults_are_merged_into_each_spec(config_path: Path) -> None:
    specs = load_specs(config_path)
    assert [s.persona_id for s in specs] == ["caring", "malevolent"]
    assert specs[0].base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert specs[1].method_config == {"shared_knob": 1, "extra": 2}


def test_duplicate_specs_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dupe.yaml"
    entry = {"persona_id": "caring", "method": "prompting", "base_model": "m"}
    path.write_text(yaml.safe_dump({"specs": [entry, dict(entry)]}))
    with pytest.raises(ValueError, match="duplicate specs"):
        load_specs(path)


def _config_with_default_evals(tmp_path: Path, names: list[str]) -> Path:
    cfg = {**CONFIG, "default_evals": names}
    path = tmp_path / "specs_default_evals.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def toys() -> None:
    """Both toy evals in the registry, so a selection test can name one and exclude the other."""
    from tests.fake_evals.marker_count import register_toy

    register_toy("marker_count")
    register_toy("mean_logprob")


def test_default_evals_pins_the_set_without_a_flag(tmp_path: Path, toys: None) -> None:
    # No --evals: the config's default_evals is the run set, not the whole registry.
    config = load_config(_config_with_default_evals(tmp_path, ["marker_count"]))
    selected = _select_evals(all_evals(), None, config)
    assert {e.name for e in selected} == {"marker_count"}


def test_explicit_evals_override_default_evals(tmp_path: Path, toys: None) -> None:
    # --evals wins over default_evals.
    config = load_config(_config_with_default_evals(tmp_path, ["marker_count"]))
    selected = _select_evals(all_evals(), ["mean_logprob"], config)
    assert {e.name for e in selected} == {"mean_logprob"}


def test_stale_default_evals_name_raises(tmp_path: Path, results_root: Path) -> None:
    path = _config_with_default_evals(tmp_path, ["marker_count", "no_such_eval"])
    with pytest.raises(ValueError, match="default_evals"):
        run_all(path, root=results_root)


def test_flat_mode_scores_lower_than_persona_mode() -> None:
    """If an eval scores flat output as highly as persona output, the eval is broken."""
    from core.endpoint import FakeEndpoint
    from tests.fake_evals.marker_count import count_markers

    assert count_markers(FakeEndpoint(persona="caring", seed=0)) > count_markers(
        FakeEndpoint(persona=None, seed=0)
    )


# ------------------------------------------------------------- config shapes


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_organisms_shape_states_persona_once(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "organisms": {"mathematical": [{"method": "prompting"}, {"method": "base"}]},
        },
    )
    specs = load_specs(path)
    assert [(s.persona_id, s.method) for s in specs] == [
        ("mathematical", "prompting"),
        ("mathematical", "base"),
    ]


def test_conflicting_persona_id_is_rejected(tmp_path: Path) -> None:
    """The block key is the persona; silently preferring one side registers the wrong organism."""
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "organisms": {"mathematical": [{"method": "base", "persona_id": "impulsive"}]},
        },
    )
    with pytest.raises(ValueError, match="declares persona_id"):
        load_specs(path)


def test_config_with_no_organisms_or_specs_is_an_error(tmp_path: Path) -> None:
    """Running zero organisms looks exactly like a fully cached run."""
    with pytest.raises(ValueError, match="nothing to run"):
        load_specs(_write(tmp_path, {"defaults": {"base_model": "m"}}))


def test_variants_expand_and_merge_method_config(tmp_path: Path) -> None:
    """A sweep states the shared knobs once; each variant states only what differs."""
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "organisms": {
                "mathematical": [
                    {
                        "method": "sft_behaviour",
                        "method_config": {"adapter_uri": "org/a", "lora_r": 16},
                        "variants": [
                            {"train_scale": 0.5, "method_config": {"subfolder": "scale_0.5"}},
                            {"train_scale": 1.0, "method_config": {"subfolder": "scale_1.0"}},
                        ],
                    }
                ]
            },
        },
    )
    specs = load_specs(path)
    assert [s.train_scale for s in specs] == [0.5, 1.0]
    for spec in specs:
        assert spec.method_config["adapter_uri"] == "org/a"  # inherited
        assert spec.method_config["lora_r"] == 16  # inherited
    assert specs[0].method_config["subfolder"] == "scale_0.5"  # per-variant
    assert len({s.spec_hash for s in specs}) == 2


def test_variant_method_config_overrides_the_parent(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "organisms": {
                "mathematical": [
                    {
                        "method": "steering",
                        "method_config": {"vector_uri": "org/v", "alpha": 1.0},
                        "variants": [{"method_config": {"alpha": 2.0}}],
                    }
                ]
            },
        },
    )
    spec = load_specs(path)[0]
    assert spec.method_config == {"vector_uri": "org/v", "alpha": 2.0}


def test_load_specs_does_not_multiply_by_seeds(tmp_path: Path) -> None:
    """`load_specs` is the distinct-organism list: a `seeds:` knob does NOT expand it here.

    Replication is a run-path step (`expand_seeds`), so every documented count and the
    multi-group/train_scale arithmetic that read `load_specs` stay seed-independent.
    """
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "seeds": [0, 1, 2],  # present, but load_specs must ignore it
            "organisms": {"mathematical": [{"method": "prompting"}, {"method": "base"}]},
        },
    )
    specs = load_specs(path)
    assert len(specs) == 2
    assert all(s.seed == 0 for s in specs)


def test_load_specs_rejects_a_pinned_seed_including_zero(tmp_path: Path) -> None:
    """The `seeds:` knob owns replication; an explicit per-entry seed is rejected at load.

    The `seed: 0` case is the point: it is indistinguishable from the `defaults` seed once
    specs are built, so only `load_specs` (which sees the raw entry) can catch it — the
    `expand_seeds` backstop cannot.
    """
    for pinned in (0, 2):
        path = _write(
            tmp_path,
            {
                "defaults": {"base_model": "m"},
                "seeds": [0, 1, 2],
                "organisms": {
                    "mathematical": [{"method": "base", "variants": [{"seed": pinned}]}]
                },
            },
        )
        with pytest.raises(ValueError, match="owns seed replication"):
            load_specs(path)


def test_expand_seeds_multiplies_every_organism() -> None:
    from serving.runner import expand_seeds

    base = [
        OrganismSpec("mathematical", "m", "prompting", {}),
        OrganismSpec("mathematical", "m", "base", {}),
    ]
    out = expand_seeds(base, {"seeds": [0, 1, 2]})
    assert len(out) == 6
    assert {(s.method, s.seed) for s in out} == {
        (m, s) for m in ("prompting", "base") for s in (0, 1, 2)
    }
    # Distinct seeds -> distinct spec hashes, so nothing collides.
    assert len({s.spec_hash for s in out}) == 6


def test_expand_seeds_absent_knob_is_a_noop() -> None:
    from serving.runner import expand_seeds

    base = [OrganismSpec("mathematical", "m", "base", {})]
    assert expand_seeds(base, {}) == base


def test_expand_seeds_rejects_a_pinned_seed() -> None:
    """A per-organism seed AND the global knob is the double-owned state the knob removes."""
    from serving.runner import expand_seeds

    base = [OrganismSpec("mathematical", "m", "base", {}, seed=7)]
    with pytest.raises(ValueError, match="owns seed replication"):
        expand_seeds(base, {"seeds": [0, 1, 2]})


def test_expand_seeds_validates_the_list() -> None:
    from serving.runner import expand_seeds

    base = [OrganismSpec("mathematical", "m", "base", {})]
    for bad in ([], "0,1,2", [0, 1.5], [True]):
        with pytest.raises(ValueError, match="non-empty list of ints"):
            expand_seeds(base, {"seeds": bad})


def test_temperature_is_one_for_behavioural_axes_only() -> None:
    """`eval_config_for` injects temperature 1.0 for behavioural axes and NOTHING elsewhere.

    Nothing-elsewhere is the load-bearing half: injecting a 0.0 for coherence would clobber
    `mu_decisiveness`/`transitivity`, which read their own plan default of 1.0.
    """
    from serving.runner import eval_config_for

    evals = {e.name: e for e in all_evals()}
    spec = OrganismSpec("impulsive", "Qwen/Qwen2.5-7B-Instruct", "base", {})
    cfg: dict = {"eval_configs": {"default": {}}}

    assert eval_config_for(cfg, spec, evals["generality_fermi_gen"])["temperature"] == 1.0
    # Non-behavioural axes: no key injected, so each builder's own 0.0 / plan default holds.
    assert "temperature" not in eval_config_for(cfg, spec, evals["mmlu"])
    assert "temperature" not in eval_config_for(cfg, spec, evals["mu_decisiveness"])

    # An explicit per-eval temperature still wins over the axis default.
    cfg2: dict = {"eval_configs": {"generality_fermi_gen": {"temperature": 0.0}}}
    assert eval_config_for(cfg2, spec, evals["generality_fermi_gen"])["temperature"] == 0.0


def test_empty_variants_is_an_error(tmp_path: Path) -> None:
    """An empty sweep reads as 'covered' while registering nothing."""
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "organisms": {"mathematical": [{"method": "sft_behaviour", "variants": []}]},
        },
    )
    with pytest.raises(ValueError, match="non-empty list"):
        load_specs(path)


def test_organisms_must_be_a_mapping_of_persona_to_a_list(tmp_path: Path) -> None:
    """A flat list under `organisms:` is the natural mistake when moving off `specs:`.

    PyYAML would hand `_entries` a list, `.items()` would raise AttributeError, and the
    traceback would name PyYAML rather than the line of config that is wrong.
    """
    path = _write(tmp_path, {"organisms": [{"persona_id": "mathematical", "method": "base"}]})
    with pytest.raises(ValueError, match="must be a mapping of persona id to a list"):
        load_specs(path)


def test_a_persona_block_must_be_a_list(tmp_path: Path) -> None:
    """One organism written as a bare mapping instead of a one-item list."""
    path = _write(tmp_path, {"organisms": {"mathematical": {"method": "base"}}})
    with pytest.raises(ValueError, match=r"organisms\['mathematical'\] must be a list"):
        load_specs(path)


def test_a_persona_block_entry_must_be_a_mapping(tmp_path: Path) -> None:
    """A bare string in a persona block — e.g. `- base` instead of `- method: base`."""
    path = _write(tmp_path, {"organisms": {"mathematical": ["base"]}})
    with pytest.raises(ValueError, match=r"organisms\['mathematical'\]\[0\] must be a mapping"):
        load_specs(path)


def test_the_shipped_config_loads_in_the_grouped_shape() -> None:
    """`configs/specs.yaml` itself, not a synthetic fixture.

    The file moved from a flat `specs:` list to the nested `organisms:` mapping, so the
    shape the runner actually ships against is worth asserting directly. Without this, a
    structural regression surfaces as a confusing builder-sync failure in
    tests/test_implant.py, or as a sweep that quietly registers the wrong matrix.

    Asserted on structure, not on a count: adding an organism is routine and must not
    fail CI. What must hold is that every persona block expands, states its persona once,
    and produces distinct organisms.
    """
    specs = load_specs("configs/specs.yaml")
    assert specs, "the shipped config registered no organisms"

    grouped = load_config("configs/specs.yaml")["organisms"]
    assert {s.persona_id for s in specs} == set(grouped)
    # Every block expanded to at least as many organisms as it has entries (variants only
    # ever add). A block silently dropped would still leave the other personas passing.
    for persona_id, entries in grouped.items():
        assert sum(s.persona_id == persona_id for s in specs) >= len(entries)
    # Distinct hashes: the grouped shape injects persona_id per block rather than reading
    # it off each entry, and a bug there would collapse organisms onto one cache key.
    assert len({s.spec_hash for s in specs}) == len(specs)


def test_robustness_arms_share_one_item_set_at_every_base_model() -> None:
    """Pins `configs/specs.yaml`'s four robustness `_gen` cells through the real
    `eval_config_for` merge path (`by_base_model` + the flat default), not just a static read
    of the YAML. The three pressure arms set `include_control: false` and read their drop
    against `robustness_open_ended_gen`, which is only valid on the SAME items: every cell
    runs the whole generated bank (no `limit_items`) at every production base model, so a cap
    or a `by_base_model` override added to one cell must fail here.

    Replaces the hh-rlhf cells' "300 items at every base model" pin (2026-09-15; 2nd Codex
    review round, item 5b), retired with those cells on 2026-10-01.
    """
    config = load_config("configs/specs.yaml")
    for eval_name in (
        "robustness_open_ended_gen",
        "robustness_adversarial_prompt_gen",
        "robustness_critique_gen",
        "robustness_multiturn_debate_gen",
    ):
        ev = next(e for e in all_evals() if e.name == eval_name)
        for base_model in (
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
        ):
            spec = OrganismSpec("impulsive", base_model, "base", {})
            merged = eval_config_for(config, spec, ev)
            assert merged.get("limit_items") is None, (eval_name, base_model)
            assert merged.get("bank") is None, (eval_name, base_model)  # the persona's own
            if eval_name != "robustness_open_ended_gen":
                assert merged["include_control"] is False, (eval_name, base_model)


def test_every_shipped_organism_resolves() -> None:
    """The grouped shape must produce specs that survive the whole path, not just parse.

    A spec that loads but cannot resolve fails at sweep time, one organism in, after the
    engine is up — the expensive place to find out.
    """
    from serving.resolve import resolve

    for spec in load_specs("configs/specs.yaml"):
        assert resolve(spec) is not None, f"{spec} did not resolve"


def test_variants_work_in_the_flat_specs_shape(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "defaults": {"base_model": "m"},
            "specs": [
                {
                    "persona_id": "mathematical",
                    "method": "sft_behaviour",
                    "variants": [{"train_scale": 0.5}, {"train_scale": 1.0}],
                }
            ],
        },
    )
    assert [s.train_scale for s in load_specs(path)] == [0.5, 1.0]


def test_fake_mode_strips_real_judge_and_adversary() -> None:
    """A real judge/adversary block in the config must not survive a fake-mode run — the
    fake pipeline (CI) has no OPENAI_API_KEY, so any http backend would fail every call.
    """
    cfg = {
        "eval_configs": {
            "default": {"judge": {"backend": "http", "model": "gpt-5.6-luna"}, "n_samples": 3},
            "robustness_multiturn_debate_gen": {
                "adversary": {"backend": "http", "model": "gpt-5.6-terra"}
            },
        }
    }
    _force_fake_backends(cfg)
    assert "judge" not in cfg["eval_configs"]["default"]
    assert "adversary" not in cfg["eval_configs"]["robustness_multiturn_debate_gen"]


def test_fake_mode_caps_reach_the_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two friedness caps are only worth anything if the PLANS honour them.

    `_force_fake_backends` shrinks `mu_decisiveness`'s upstream knobs (`_FAKE_MU_CAPS`) and
    `transitivity.n_triads=100` because a fake run at full scale is tens of
    thousands of Inspect generations per organism. Nothing downstream fails if a cap is
    dropped or lands under a key the plan does not read — the fake pipeline just gets
    slow again, which is how the 4x-stale 2026-08-25 pin survived. So assert on the plans,
    not on the dict.

    Census-style: it builds two plans and counts, no generation.
    """
    from evals.friedness import arcadia_plan
    from evals.preference_consistency import triad_plan

    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)  # the caps under test, alone
    cfg = {"eval_configs": {"default": {"n_samples": 3}}}
    _force_fake_backends(cfg)
    blocks = cfg["eval_configs"]

    mu = arcadia_plan({**blocks["mu_decisiveness"], "seed": 0})
    assert (mu.m, mu.n_reverse, mu.n_triads, mu.n_cross, mu.fit_steps) == (
        2,
        100,
        100,
        100,
        200,
    ), "the fake caps did not reach the mu plan"
    assert mu.R == 5, "R is not capped: every Elo round still runs"
    assert len(triad_plan({**blocks["transitivity"], "seed": 0}).triads) == 100

    # setdefault, not assignment: an explicit smaller budget (specs.smoke.yaml) still wins.
    explicit = {"eval_configs": {"mu_decisiveness": {"m": 1}, "transitivity": {}}}
    _force_fake_backends(explicit)
    assert explicit["eval_configs"]["mu_decisiveness"]["m"] == 1


def test_duplicate_yaml_keys_rejected(tmp_path: Path) -> None:
    """A doubled key must fail loudly, not silently keep the last (dropped items_path bug)."""
    path = tmp_path / "dupe.yaml"
    path.write_text("eval_configs:\n  default:\n    n_samples: 3\n    n_samples: 5\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_config(path)


def test_merge_keys_survive_the_duplicate_check(tmp_path: Path) -> None:
    """`<<: *anchor` is how a config states shared LoRA knobs once. The duplicate-key
    loader must expand it, not choke on the merge tag, and an explicit override of an
    anchored key is an override rather than a duplicate.

    Driven off its own YAML rather than the shipped config: `configs/specs.yaml` carries no
    merge key at all today (the anchored rows went with the method deleted on 2026-09-03),
    and a test that read the shipped file would go quietly vacuous the next time that
    happens — which is why the one that did was removed rather than re-pointed."""
    path = tmp_path / "merge.yaml"
    path.write_text(
        "_lora: &lora\n"
        "  lora_r: 16\n"
        "  epochs: 3.0\n"
        "eval_configs:\n"
        "  default:\n"
        "    <<: *lora\n"
        "    lora_r: 8\n"
    )
    assert load_config(path)["eval_configs"]["default"] == {"lora_r": 8, "epochs": 3.0}


def test_unknown_eval_name_raises(config_path: Path, results_root: Path) -> None:
    """A typo in --evals must error, not silently shrink the matrix."""
    with pytest.raises(ValueError, match="--evals: no match"):
        run_all(
            config_path,
            root=results_root,
            only_evals=["marker_count", "typo"],
        )


def test_unknown_method_name_raises(config_path: Path, results_root: Path) -> None:
    with pytest.raises(ValueError, match="--methods: no match"):
        run_all(
            config_path,
            root=results_root,
            only_methods=["promting"],  # sic
        )


def test_only_personas_narrows_the_matrix(config_path: Path) -> None:
    """`--personas` selects one persona's slice without a cut-down copy of the config.

    Both fixture organisms share a method, so this cannot pass by accident through the
    method filter.
    """
    specs = load_specs(config_path)
    caring = {s.spec_hash for s in specs if s.persona_id == "caring"}
    picked = _select_specs(specs, ["caring"], None)
    assert {s.spec_hash for s in picked} == caring


def test_only_personas_composes_with_only_methods(config_path: Path) -> None:
    """The two filters intersect rather than one overriding the other."""
    specs = load_specs(config_path)
    wanted = {
        s.spec_hash for s in specs if s.persona_id == "malevolent" and s.method == "prompting"
    }
    picked = _select_specs(specs, ["malevolent"], ["prompting"])
    assert {s.spec_hash for s in picked} == wanted


def test_unknown_persona_name_raises(config_path: Path, results_root: Path) -> None:
    """A typo must error rather than silently running an empty sweep."""
    with pytest.raises(ValueError, match="--personas: no match"):
        run_all(
            config_path,
            root=results_root,
            only_personas=["carring"],  # sic
        )


def test_method_guard_runs_against_the_persona_narrowed_set(
    tmp_path: Path, results_root: Path
) -> None:
    """A method valid for another persona is still an empty intersection here.

    Saying so beats running nothing and reporting success, which is the same failure
    `--methods` typo-guarding already prevents.
    """
    config = dict(CONFIG)
    config["specs"] = [
        {"persona_id": "caring", "method": "prompting"},
        {"persona_id": "malevolent", "method": "sft_behaviour"},
    ]
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="--methods: no match"):
        run_all(
            path,
            root=results_root,
            only_personas=["caring"],
            only_methods=["sft_behaviour"],  # exists, but not for `caring`
        )


def test_real_run_fails_preflight_without_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """--real must refuse to start when vllm is missing, not fail once per organism."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    try:
        import vllm  # noqa: F401

        pytest.skip("vllm is installed here; the import-fail path is not exercisable")
    except ImportError:
        pass
    cfg = {"eval_configs": {"default": {"judge": {"backend": "http"}}}}
    with pytest.raises(RuntimeError, match="cannot import vllm"):
        assert_real_backends(cfg, [], _VLLM_SPECS)


def test_real_run_refuses_max_items_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The item cap is CI plumbing; on a real run it makes a small-n sweep look healthy.

    `scripts/run_all.sh` refuses it too, but every other entry point
    (`python -m serving.inspect_runner --real`, a direct `run_all(real=True)`) goes
    through this pre-flight and nothing else, so the check has to live here as well.
    """
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    cfg = {"eval_configs": {"default": {"judge": {"backend": "http"}}}}
    with pytest.raises(RuntimeError, match="MO_EVALS_MAX_ITEMS=25"):
        assert_real_backends(cfg, [], _VLLM_SPECS)


def test_real_run_fails_preflight_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver too old for the torch CUDA build must fail loud, not run silently on CPU."""
    import sys
    import types

    import torch

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))  # let import vllm pass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = {"eval_configs": {"default": {"judge": {"backend": "http"}}}}
    with pytest.raises(RuntimeError, match="no usable GPU"):
        assert_real_backends(cfg, [], _VLLM_SPECS)


def _gpu_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the vllm/torch pre-flights pass so the probe stage is reachable."""
    import types

    import torch

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _eval(name: str) -> RegisteredEval:
    return RegisteredEval(
        name=name,
        axis="robustness",
        applicable_to=("persona_trait",),
        summary="A throwaway.",
        builder=lambda config: None,
    )


def test_real_run_probes_each_distinct_backend_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presence checks cannot catch params the API rejects (gpt-5.6 refused
    temperature: 0 with a 400 on every judge call). The pre-flight
    makes one live call per DISTINCT judge/adversary block and fails loud."""
    import judges.backends as backends_module

    _gpu_ok(monkeypatch)
    probed: list[dict] = []

    class OkBackend:
        def complete(self, messages, **kw):
            return "ok"

    monkeypatch.setattr(
        backends_module, "build_backend", lambda block: (probed.append(block), OkBackend())[1]
    )
    judge = {"backend": "http", "model": "j"}
    cfg = {"eval_configs": {"default": {"judge": judge}}}
    assert_real_backends(cfg, [_eval("a"), _eval("b")], _VLLM_SPECS)
    assert probed == [judge]  # two evals, one shared judge block, one probe


def test_real_run_probes_the_secondary_panel_judge_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The panel's secondary judge (judges/judge.py) is scored on every judged item just
    like the primary, so a real run must not silently skip its presence/key/probe checks
    the way it would if `assert_real_backends` only looked at the top-level judge block."""
    import judges.backends as backends_module

    _gpu_ok(monkeypatch)
    monkeypatch.setenv("SECOND_KEY", "k")
    probed: list[dict] = []

    class OkBackend:
        def complete(self, messages, **kw):
            return "ok"

    monkeypatch.setattr(
        backends_module, "build_backend", lambda block: (probed.append(block), OkBackend())[1]
    )
    secondary = {"backend": "http", "model": "j2", "api_key_env": "SECOND_KEY"}
    judge = {"backend": "http", "model": "j", "secondary": secondary}
    cfg = {"eval_configs": {"default": {"judge": judge}}}
    assert_real_backends(cfg, [_eval("a")], _VLLM_SPECS)
    assert judge in probed and secondary in probed


def test_real_run_refuses_a_fake_secondary_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `judge.secondary` block absent or not `backend: http` must fail loud, exactly as
    a missing top-level judge would — the panel's whole point is that BOTH judges are
    real on a reported run."""
    _gpu_ok(monkeypatch)
    judge = {"backend": "http", "model": "j", "secondary": {"backend": "fake"}}
    cfg = {"eval_configs": {"default": {"judge": judge}}}
    with pytest.raises(RuntimeError, match=r"judge\.secondary would use a FAKE backend"):
        assert_real_backends(cfg, [_eval("a")], _VLLM_SPECS)


def test_a_judge_free_run_needs_no_judge_key_and_probes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WMDP-only real run was refused for want of $OPENAI_API_KEY, then would have
    spent a live probe call, on a judge no scheduled eval can call. The capability
    family is judge-free by construction, so neither should happen."""
    import judges.backends as backends_module

    _gpu_ok(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    probed: list[dict] = []
    monkeypatch.setattr(
        backends_module, "build_backend", lambda block: probed.append(block) or None
    )
    # No judge block at all, which is what a judge-free config legitimately looks like.
    assert_real_backends({"eval_configs": {}}, [_eval("wmdp"), _eval("mmlu")], _VLLM_SPECS)
    assert probed == []


def test_a_judged_eval_still_needs_its_key_alongside_a_judge_free_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip must be per-eval, not a global off-switch: one judged eval in the set
    still demands the key. `ifeval` used to be the trap here (a capability-family cell that
    turned on a real judge), but it is stock `inspect_evals.ifeval` and judge-free now
    (084aa9d), so it joined `_JUDGE_FREE_EVALS`. A genuinely judged eval alongside the
    judge-free `wmdp` still has to trip the missing-key check, so the guard uses one."""
    _gpu_ok(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = {"eval_configs": {"default": {"judge": {"backend": "http", "model": "j"}}}}
    with pytest.raises(
        RuntimeError, match=r"eval 'generality_fermi_gen' judge needs \$OPENAI_API_KEY"
    ):
        assert_real_backends(cfg, [_eval("wmdp"), _eval("generality_fermi_gen")], _VLLM_SPECS)


def test_real_run_fails_preflight_when_the_probe_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import judges.backends as backends_module

    _gpu_ok(monkeypatch)

    class RejectingBackend:
        def complete(self, messages, **kw):
            raise RuntimeError("HTTP 400: unsupported_value temperature")

    monkeypatch.setattr(backends_module, "build_backend", lambda block: RejectingBackend())
    cfg = {"eval_configs": {"default": {"judge": {"backend": "http", "model": "j"}}}}
    with pytest.raises(RuntimeError, match="live probe of eval 'a' judge failed"):
        assert_real_backends(cfg, [_eval("a")], _VLLM_SPECS)


def test_real_run_probe_dedup_survives_non_json_judge_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A YAML judge block can hold a non-JSON value (an unquoted date parses to
    datetime.date). The probe-dedup key must not crash on it."""
    import datetime

    import judges.backends as backends_module

    _gpu_ok(monkeypatch)

    class OkBackend:
        def complete(self, messages, **kw):
            return "ok"

    monkeypatch.setattr(backends_module, "build_backend", lambda block: OkBackend())
    judge = {"backend": "http", "model": "j", "as_of": datetime.date(2026, 8, 14)}
    cfg = {"eval_configs": {"default": {"judge": judge}}}
    assert_real_backends(cfg, [_eval("a")], _VLLM_SPECS)  # no TypeError from json.dumps


def test_load_dotenv_fills_missing_env_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secrets in .env must reach the runner's process env (HF downloads, judge key):
    a pod run without exported env failed all 36 resolves with 401s.
    Existing environment always wins; quotes and comments are handled."""
    import os

    from serving.runner import load_dotenv

    env = tmp_path / ".env"
    env.write_text('HF_TOKEN="hf_abc"\n# comment\nOPENAI_API_KEY=sk-xyz\nEXISTING=file\n\n')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "ambient")
    load_dotenv(env)
    assert os.environ["HF_TOKEN"] == "hf_abc"  # quotes stripped
    assert os.environ["OPENAI_API_KEY"] == "sk-xyz"
    assert os.environ["EXISTING"] == "ambient"  # ambient env is never overridden


def _written(spec_hash: str, eval_name: str) -> RunOutcome:
    return RunOutcome(spec_hash, eval_name, "written", n_samples=3, seconds=0.5)


def test_timing_table_reports_time_and_entries_per_eval() -> None:
    """Every sweep ends with a per-eval cost table, one row per eval that ran."""
    outcomes = [
        _written(h, name) for h in ("h1", "h2") for name in ("marker_count", "mean_logprob")
    ]
    table = timing_table(outcomes, wall=1.5)

    assert "s/sample" in table
    for name in ("marker_count", "mean_logprob"):
        assert name in table
    # 2 specs x 2 evals x 3 samples, and the wall clock is reported alongside.
    assert "TOTAL" in table and "12" in table
    assert "wall clock 1.5s" in table


def test_timing_table_separates_cells_that_did_not_run() -> None:
    """Cached cells cost ~0s; averaging them into s/sample would understate a rerun."""
    table = timing_table([RunOutcome(f"h{i}", "marker_count", "cached") for i in range(4)])

    assert "not run: cached=4" in table
    assert "TOTAL" in table


def test_timing_table_names_why_cells_failed() -> None:
    """A wholesale failure must print its REASON, not just a count.

    The 2026-08-25 sweep reported `failed=571` and nothing else, so the cause was
    unrecoverable once stdout was gone. Grouped reasons are the durable record.
    """
    outcomes = [
        RunOutcome("h1", "e1", "failed", detail="eval_set failed: boom"),
        RunOutcome("h2", "e1", "failed", detail="eval_set failed: boom"),
        RunOutcome("h3", "e2", "failed", detail="resolve failed: no adapter"),
        RunOutcome("h4", "e2", "written", n_samples=3),
    ]
    table = timing_table(outcomes)

    assert "failures by reason (3 cell(s), 2 distinct):" in table
    assert "2 x eval_set failed: boom" in table
    assert "1 x resolve failed: no adapter" in table


def test_timing_table_is_silent_about_failures_when_there_are_none() -> None:
    """No failures, no section — the table stays readable on a green sweep."""
    table = timing_table([RunOutcome("h1", "e1", "written", n_samples=3)])

    assert "failures by reason" not in table
    assert "skipped cells" not in table


def test_timing_table_names_which_cells_were_skipped_and_why() -> None:
    """A skipped cell must name ITSELF and its reason, not just add to `skipped=N`.

    A skip writes no Parquet, so the reason existed only in an INFO line thousands of
    lines above the summary. That is the worst shape a hole in the scorecard can take: an
    absent cell reads as a design outcome ("this eval does not apply to this organism")
    when it may be a capability the endpoint quietly lost.
    """
    reason = "perplexity: endpoint cannot serve prompt logprobs"
    outcomes = [
        RunOutcome("h1", "perplexity", "skipped", detail=reason),
        RunOutcome("h2", "perplexity", "skipped", detail=reason),
        RunOutcome("h3", "whitebox_probe_persona", "skipped", detail="no activations"),
        RunOutcome("h4", "e2", "written", n_samples=3),
    ]
    table = timing_table(outcomes)

    assert "skipped cells (no rows written) (3 cell(s), 2 distinct):" in table
    # The eval name is not doubled when the message already opens with it.
    assert "2 x perplexity: endpoint cannot serve prompt logprobs" in table
    assert "1 x whitebox_probe_persona: no activations" in table


def test_a_skip_with_no_recorded_reason_still_names_its_cell() -> None:
    """`detail` is optional on `RunOutcome`; a skip that arrives without one must still
    say WHICH cell is missing, and say plainly that the reason was not recorded rather
    than printing a bare name that reads like a complete explanation."""
    table = timing_table([RunOutcome("h1", "petri_audit", "skipped")])

    assert "1 x petri_audit: (no reason recorded)" in table


# --------------------------------------------------- per-base overrides & selectors


def test_merge_one_level_is_deep_for_nested_dicts() -> None:
    """`_merge_one_level` updates a nested sub-dict rather than replacing it wholesale."""
    base = {"petri": {"auditor_model": "a", "max_turns": 10, "limit_seeds": 38}, "top": 1}
    out = _merge_one_level(base, {"petri": {"max_turns": 6}, "top": 2})
    # the override key inside `petri` wins, the siblings survive
    assert out["petri"] == {"auditor_model": "a", "max_turns": 6, "limit_seeds": 38}
    assert out["top"] == 2
    assert base["petri"]["max_turns"] == 10  # input not mutated


def test_by_base_model_applies_per_size_and_never_leaks() -> None:
    """`by_base_model` overrides for the organism's base and is stripped from the result."""
    config = {
        "eval_configs": {
            "default": {"max_tokens": 768},
            "myeval": {
                "limit_items": 1000,
                "by_base_model": {"BIG": {"limit_items": 300}},
            },
        }
    }
    ev = SimpleNamespace(name="myeval")
    small = OrganismSpec("caring", "SMALL", "prompting", {"system_prompt": "x"})
    big = OrganismSpec("caring", "BIG", "prompting", {"system_prompt": "x"})
    small_cfg = eval_config_for(config, small, ev)  # type: ignore[arg-type]
    big_cfg = eval_config_for(config, big, ev)  # type: ignore[arg-type]
    assert small_cfg["limit_items"] == 1000
    assert big_cfg["limit_items"] == 300
    # the runner-only key never reaches the eval body
    assert "by_base_model" not in small_cfg
    assert "by_base_model" not in big_cfg


def test_eval_config_stamps_base_model() -> None:
    """A white-box cell must refuse an artifact fitted on a different base model than the
    organism, and it can only do so if the runner stamps `config['base_model']` in. (The
    guard that first read it left with the retired `whitebox_probe_*` cells, 2026-10-14.)"""
    config = load_config("configs/specs.yaml")
    ev = SimpleNamespace(name="simple_behaviour_exhibit")
    for bm in (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ):
        cfg = eval_config_for(config, OrganismSpec("impulsive", bm, "base", {}), ev)
        assert cfg["base_model"] == bm


def test_dedupe_seeds_is_stripped_from_the_eval_config() -> None:
    """`dedupe_seeds` is runner-only (read in run_matrix); it must not reach an eval body."""
    config = {"eval_configs": {"default": {"dedupe_seeds": True}, "myeval": {"n": 1}}}
    ev = SimpleNamespace(name="myeval")
    spec = OrganismSpec("caring", "m", "prompting", {"system_prompt": "x"})
    cfg = eval_config_for(config, spec, ev)  # type: ignore[arg-type]
    assert "dedupe_seeds" not in cfg
    assert cfg["n"] == 1


def test_select_specs_filters_by_base_model() -> None:
    specs = [
        OrganismSpec("caring", "A", "prompting", {"system_prompt": "x"}),
        OrganismSpec("caring", "B", "prompting", {"system_prompt": "x"}),
    ]
    kept = _select_specs(specs, None, None, ["A"])
    assert [s.base_model for s in kept] == ["A"]


def test_select_specs_base_model_typo_raises() -> None:
    specs = [OrganismSpec("caring", "A", "prompting", {"system_prompt": "x"})]
    with pytest.raises(ValueError, match="--base-models"):
        _select_specs(specs, None, None, ["Qwen/nope"])


def test_select_evals_exclude_subtracts() -> None:
    evals = all_evals()
    names = {e.name for e in evals}
    assert {"mmlu", "wmdp"} <= names
    kept = _select_evals(evals, None, {"default_evals": sorted(names)}, ["mmlu", "wmdp"])
    kept_names = {e.name for e in kept}
    assert "mmlu" not in kept_names and "wmdp" not in kept_names
    assert len(kept) == len(evals) - 2


def test_select_evals_exclude_typo_raises() -> None:
    evals = all_evals()
    with pytest.raises(ValueError, match="--exclude-evals"):
        _select_evals(evals, None, {}, ["not_an_eval"])


def test_the_over_refusal_cell_is_judge_free_and_demands_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`misaligned_overrefusal` scores with `evals.misaligned.classify_refusal`, a
    deterministic string matcher, and its task path builds no judge — but it was missing
    from `_JUDGE_FREE_EVALS`, so a run scheduling it demanded $OPENAI_API_KEY and spent a
    live probe call on a judge it would never call.

    (`refusal_agreement` DOES take a judge, and that is not this cell: it is unregistered,
    never scheduled by a runner, and re-judges stored rows out of band.)
    """
    import judges.backends as backends_module

    _gpu_ok(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    probed: list[dict] = []
    monkeypatch.setattr(
        backends_module, "build_backend", lambda block: probed.append(block) or None
    )
    assert_real_backends({"eval_configs": {}}, [_eval("misaligned_overrefusal")], _VLLM_SPECS)
    assert probed == []


@pytest.mark.parametrize("eval_name", ["misaligned_sycophancy", "misaligned_sandbagging"])
def test_the_mmlu_backed_misalignment_cells_are_judge_free_and_demand_no_key(
    eval_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Siblings of `misaligned_overrefusal`: since #250 both `misaligned_sycophancy` and
    `misaligned_sandbagging` are judge-free stock-MMLU cells (deterministic letter parsing,
    no judge on their task path), so scheduling them on a keyless real run must NOT demand
    $OPENAI_API_KEY. Regression against them being dropped from `_JUDGE_FREE_EVALS`."""
    import judges.backends as backends_module

    _gpu_ok(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    probed: list[dict] = []
    monkeypatch.setattr(
        backends_module, "build_backend", lambda block: probed.append(block) or None
    )
    assert_real_backends({"eval_configs": {}}, [_eval(eval_name)], _VLLM_SPECS)
    assert probed == []
