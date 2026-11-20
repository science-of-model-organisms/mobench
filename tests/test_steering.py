"""Steering: contrast prompts, contrastive pairs, CAA extraction, spec building.

Everything runs against `FakeEndpoint` with no GPU (CLAUDE.md). The tests that
matter most are the controls — the shuffled-label vector and the structural match between
the positive and negative prompts — because those are what stand between us and a vector
that encodes response length instead of a disposition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from core.endpoint import FakeEndpoint
from implant.behaviour.description import has_description
from implant.generators.contrast_prompts import (
    anti_system_prompt_path,
    build_contrast_prompts,
    load_contrast_prompts,
    load_negative_prompt,
    validate_contrast_prompts,
    write_contrast_prompts,
)
from implant.generators.contrastive import (
    NEG_FIELD,
    NEUTRAL_FIELD,
    POS_FIELD,
    generate_contrastive_pairs,
    pairs_hash,
    response_length_stats,
    shuffled_control_pairs,
)
from implant.generators.system_prompt import discover_personas, load_persona
from implant.steering import (
    DEFAULT_POSITIONS,
    LARGE_MODEL_STEERING,
    RAW_ALPHA_STEERING,
    RAW_ALPHA_SWEEP,
    RAW_VECTORS,
    build_normalised_steering_specs,
    build_raw_alpha_sweep_specs,
    build_raw_steering_spec,
    build_steering_spec,
    build_steering_specs,
    cosine,
    cosine_across_wordings,
    extract_caa_vector,
    extract_neutral_vector,
    load_vector,
    sha256_key,
    steering_personas,
    vector_norms,
    write_vector,
)
from serving.runner import load_specs

PERSONAS = discover_personas()


@pytest.fixture
def endpoint() -> FakeEndpoint:
    """A fake with hidden states, which the base fixture deliberately does not have."""
    return FakeEndpoint(persona="impulsive", seed=0, supports_hidden_states=True)


# --------------------------------------------------------------------- contrast prompts


@pytest.mark.parametrize("persona_id", PERSONAS)
def test_neg_prompt_differs_from_pos_only_in_preamble(persona_id: str) -> None:
    """The structural match is the whole defence against a style direction.

    If the bullets ever diverge, the vector starts encoding formatting, and no downstream
    number can tell you that happened.

    The negative is a hand-written file (``<id>.anti_system_prompt.txt``), so for a
    hand-made persona this pins that the file still keeps the bullets verbatim under an
    inverted preamble. A persona rendered from a behaviour description is the exception:
    its ``hand-written/<id>.negative.txt`` shares the FRAMING and differs in the bullets, by
    design; ``tests/test_behaviour_description.py`` pins that pair's bullet count and length.
    """
    if has_description(persona_id):
        prompts = build_contrast_prompts(persona_id)
        assert prompts["pos"][0].split("\n", 1)[0] == prompts["neg"][0].split("\n", 1)[0]
        assert prompts["pos"][0] != prompts["neg"][0]
        return
    prompts = build_contrast_prompts(persona_id)
    pos_body = prompts["pos"][0].split("\n", 1)[1]
    neg_body = prompts["neg"][0].split("\n", 1)[1]
    assert pos_body == neg_body
    assert prompts["pos"][0] != prompts["neg"][0]


@pytest.mark.parametrize("persona_id", PERSONAS)
def test_every_persona_has_contrast_prompts_committed(persona_id: str) -> None:
    """Generated artifacts are committed, so extraction is reproducible from the repo."""
    prompts = load_contrast_prompts(persona_id)
    validate_contrast_prompts(prompts, persona_id)
    assert Path(f"data/personas/{persona_id}.anti_system_prompt.txt").exists()


@pytest.mark.parametrize("persona_id", PERSONAS)
def test_every_persona_has_its_negative_file(persona_id: str) -> None:
    """One file per persona holds the complete negative; neg[0] and the anti organism
    are that file, and nothing negates a constitution."""
    committed = load_contrast_prompts(persona_id)
    assert committed["neg"][0] == load_negative_prompt(persona_id)


def test_a_missing_negative_file_fails_loudly(tmp_path: Path) -> None:
    # Impulsive's negative source is data/behaviours/impulsive/negative.txt; its anti
    # file (the served copy) is missing from this personas dir, and that fails loudly.
    (tmp_path / "impulsive.json").write_text(json.dumps(load_persona("impulsive")))
    with pytest.raises(FileNotFoundError, match="no anti file"):
        build_contrast_prompts("impulsive", tmp_path)


def test_validate_rejects_ragged_and_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pair elementwise"):
        validate_contrast_prompts({"pos": ["a", "b"], "neg": ["c"]})
    with pytest.raises(ValueError, match="non-empty list"):
        validate_contrast_prompts({"pos": [], "neg": []})
    with pytest.raises(ValueError, match="non-empty string"):
        validate_contrast_prompts({"pos": ["  "], "neg": ["x"]})


def test_writer_does_not_clobber_authored_prompts(tmp_path: Path) -> None:
    """Regenerating must never discard better wordings — it would invalidate their vectors."""
    personas_dir = tmp_path
    (personas_dir / "impulsive.json").write_text(json.dumps(load_persona("impulsive")))
    anti_system_prompt_path("impulsive", personas_dir).write_text(
        load_negative_prompt("impulsive")
    )
    # Authored wordings may differ freely, but neg[0] must stay the negative file's text.
    negative = load_negative_prompt("impulsive")
    authored = {"persona_id": "impulsive", "pos": ["POS", "POS2"], "neg": [negative, "NEG2"]}
    path = personas_dir / "impulsive.contrast_prompts.json"
    path.write_text(json.dumps(authored))

    write_contrast_prompts("impulsive", personas_dir)
    assert json.loads(path.read_text()) == authored

    write_contrast_prompts("impulsive", personas_dir, force=True)
    assert json.loads(path.read_text()) != authored


def test_anti_system_prompt_is_the_extraction_negative(tmp_path: Path) -> None:
    """The control organism and the negative arm must be the same text, not merely similar."""
    personas_dir = tmp_path
    (personas_dir / "impulsive.json").write_text(json.dumps(load_persona("impulsive")))
    anti = anti_system_prompt_path("impulsive", personas_dir)
    anti.write_text(load_negative_prompt("impulsive"))
    write_contrast_prompts("impulsive", personas_dir)
    assert anti.read_text() == load_contrast_prompts("impulsive", personas_dir)["neg"][0]


# -------------------------------------------------------------------- contrastive pairs


def test_pairs_cover_every_wording_and_question(endpoint: FakeEndpoint) -> None:
    prompts = load_contrast_prompts("impulsive")
    questions = ["q1", "q2", "q3"]
    pairs = generate_contrastive_pairs("impulsive", endpoint, questions=questions)
    assert len(pairs) == len(prompts["pos"]) * len(questions)
    assert all(POS_FIELD in p and NEG_FIELD in p for p in pairs)
    assert {p["question"] for p in pairs} == set(questions)


def test_pairs_send_the_two_system_prompts(endpoint: FakeEndpoint) -> None:
    """The positive and negative arms must actually differ in the system turn."""
    generate_contrastive_pairs("impulsive", endpoint, questions=["q"])
    systems = [c[0]["content"] for c in endpoint.calls if c[0]["role"] == "system"]
    assert len(set(systems)) == 2


def test_neutral_arm_sends_no_system_prompt(endpoint: FakeEndpoint) -> None:
    pairs = generate_contrastive_pairs(
        "impulsive", endpoint, questions=["q"], include_neutral=True
    )
    assert NEUTRAL_FIELD in pairs[0]
    assert any(c[0]["role"] == "user" for c in endpoint.calls)


def test_shuffled_control_swaps_labels_not_questions(endpoint: FakeEndpoint) -> None:
    """A label swap sends the expected difference to zero; permuting negatives would not."""
    pairs = generate_contrastive_pairs(
        "impulsive", endpoint, questions=[f"q{i}" for i in range(20)]
    )
    control = shuffled_control_pairs(pairs, seed=0)
    assert [c["question"] for c in control] == [p["question"] for p in pairs]
    assert any(c["label_swapped"] for c in control)
    assert not all(c["label_swapped"] for c in control)
    for original, swapped in zip(pairs, control, strict=True):
        assert {original[POS_FIELD], original[NEG_FIELD]} == {
            swapped[POS_FIELD],
            swapped[NEG_FIELD],
        }


def test_shuffled_control_is_deterministic(endpoint: FakeEndpoint) -> None:
    pairs = generate_contrastive_pairs("impulsive", endpoint, questions=["a", "b", "c"])
    assert shuffled_control_pairs(pairs, seed=7) == shuffled_control_pairs(pairs, seed=7)


def test_pairs_hash_tracks_content(endpoint: FakeEndpoint) -> None:
    pairs = generate_contrastive_pairs("impulsive", endpoint, questions=["a"])
    assert pairs_hash(pairs) == pairs_hash(list(pairs))
    mutated = [{**pairs[0], POS_FIELD: "different"}]
    assert pairs_hash(mutated) != pairs_hash(pairs)


def test_response_length_stats_report_the_gap(endpoint: FakeEndpoint) -> None:
    pairs = [{POS_FIELD: "a b c d", NEG_FIELD: "a b"}]
    stats = response_length_stats(pairs)
    assert stats["length_gap"] == 2.0


# ------------------------------------------------------------------------ CAA extraction


class _SpyEndpoint:
    """Returns activations chosen by the system prompt, so the maths is checkable by hand."""

    def __init__(self, by_prefix: dict[str, list[list[float]]]) -> None:
        self.by_prefix = by_prefix
        self.seen: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        return "response"

    def logprobs(self, messages: list[dict[str, str]], **kw: Any) -> list[float]:
        return [0.0]

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> list[list[float]]:
        self.seen.append(messages)
        for prefix, value in self.by_prefix.items():
            if messages[0]["content"].startswith(prefix):
                return value
        raise AssertionError(f"unexpected system prompt {messages[0]['content'][:40]!r}")


def test_caa_vector_is_the_mean_difference() -> None:
    prompts = {"pos": ["POS"], "neg": ["NEG"]}
    endpoint = _SpyEndpoint({"POS": [[3.0, 1.0]], "NEG": [[1.0, 1.0]]})
    pairs = [
        {"question": "q", "instruction_idx": 0, POS_FIELD: "a", NEG_FIELD: "b"},
        {"question": "r", "instruction_idx": 0, POS_FIELD: "c", NEG_FIELD: "d"},
    ]
    vector = extract_caa_vector(pairs, endpoint, prompts)
    assert vector == {"layer_0": [2.0, 0.0]}


def test_extraction_reads_the_response_not_just_the_prompt() -> None:
    """Activations must be conditioned on the response; a prompt-only read measures nothing new."""
    prompts = {"pos": ["POS"], "neg": ["NEG"]}
    endpoint = _SpyEndpoint({"POS": [[1.0]], "NEG": [[0.0]]})
    extract_caa_vector(
        [{"question": "q", "instruction_idx": 0, POS_FIELD: "ANSWER", NEG_FIELD: "OTHER"}],
        endpoint,
        prompts,
    )
    assert endpoint.seen[0][-1] == {"role": "assistant", "content": "ANSWER"}


def test_shuffled_control_vector_is_near_zero() -> None:
    """The control that says the pipeline measures disposition rather than formatting.

    Positives and negatives carry a real difference; once their labels are randomised, the
    expected difference is zero and the extracted norm must collapse.
    """
    prompts = {"pos": ["POS"], "neg": ["NEG"]}
    pos_act, neg_act = [[4.0, 0.0]], [[0.0, 0.0]]

    class _ByResponse(_SpyEndpoint):
        def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> list[list[float]]:
            # Activation follows the *response*, so a label swap really swaps the arms.
            return pos_act if messages[-1]["content"].startswith("P") else neg_act

    endpoint = _ByResponse({})
    pairs = [
        {"question": f"q{i}", "instruction_idx": 0, POS_FIELD: f"P{i}", NEG_FIELD: f"N{i}"}
        for i in range(40)
    ]
    real = extract_caa_vector(pairs, endpoint, prompts)
    control = extract_caa_vector(shuffled_control_pairs(pairs, seed=0), endpoint, prompts)

    real_norm = vector_norms(real)["layer_0"]
    control_norm = vector_norms(control)["layer_0"]
    assert real_norm == pytest.approx(4.0)
    assert control_norm < real_norm / 2


def test_neutral_vector_needs_the_neutral_arm() -> None:
    prompts = {"pos": ["POS"], "neg": ["NEG"]}
    endpoint = _SpyEndpoint({"POS": [[1.0]], "NEG": [[0.0]]})
    pairs = [{"question": "q", "instruction_idx": 0, POS_FIELD: "a", NEG_FIELD: "b"}]
    assert extract_neutral_vector(pairs, endpoint, prompts) is None


def test_extraction_rejects_inconsistent_layers() -> None:
    prompts = {"pos": ["POS"], "neg": ["NEG"]}
    endpoint = _SpyEndpoint({"POS": [[1.0, 2.0]], "NEG": [[1.0]]})
    with pytest.raises(ValueError, match="inconsistent width"):
        extract_caa_vector(
            [{"question": "q", "instruction_idx": 0, POS_FIELD: "a", NEG_FIELD: "b"}],
            endpoint,
            prompts,
        )


def test_extraction_rejects_empty_pairs() -> None:
    with pytest.raises(ValueError, match="no contrastive pairs"):
        extract_caa_vector([], _SpyEndpoint({}), {"pos": ["P"], "neg": ["N"]})


def test_cosine_across_wordings_is_one_for_identical_directions() -> None:
    prompts = {"pos": ["POS_A", "POS_B"], "neg": ["NEG_A", "NEG_B"]}
    endpoint = _SpyEndpoint(
        {
            "POS_A": [[2.0, 0.0]],
            "NEG_A": [[0.0, 0.0]],
            "POS_B": [[5.0, 0.0]],
            "NEG_B": [[0.0, 0.0]],
        }
    )
    pairs = [
        {"question": "q", "instruction_idx": idx, POS_FIELD: "a", NEG_FIELD: "b"} for idx in (0, 1)
    ]
    assert cosine_across_wordings(pairs, endpoint, prompts)["0_vs_1"] == pytest.approx(1.0)


def test_cosine_handles_degenerate_vectors() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_extraction_runs_against_fake_endpoint(endpoint: FakeEndpoint) -> None:
    """The no-GPU path the whole project depends on."""
    prompts = load_contrast_prompts("impulsive")
    pairs = generate_contrastive_pairs("impulsive", endpoint, questions=["q1", "q2"])
    vector = extract_caa_vector(pairs, endpoint, prompts)
    assert vector and all(len(v) == 8 for v in vector.values())


# ------------------------------------------------------------------------- artifact I/O


def test_vector_round_trips_through_safetensors(tmp_path: Path) -> None:
    vector = {"layer_0": [1.0, 2.0], "layer_1": [3.0, 4.0]}
    out = write_vector(vector, tmp_path / "impulsive", {"algo": "caa", "n_pairs": 2})
    loaded = load_vector(out)
    assert loaded == vector
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["algo"] == "caa" and meta["n_pairs"] == 2


def test_load_vector_says_which_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no steering vector"):
        load_vector(tmp_path)


def test_vector_norms() -> None:
    assert vector_norms({"layer_0": [3.0, 4.0]})["layer_0"] == pytest.approx(5.0)
    assert math.isclose(vector_norms({"layer_0": [0.0]})["layer_0"], 0.0)


# ------------------------------------------------------------------------ spec building


def test_steering_knobs_are_in_the_spec_hash() -> None:
    """Alpha and layers change the organism, so two sweeps must not collide in the cache."""
    base = dict(vector_uri="org/vectors", subfolder="impulsive")
    a = build_steering_spec("impulsive", layers=[14], alpha=1.0, **base)
    stronger = build_steering_spec("impulsive", layers=[14], alpha=2.0, **base)
    deeper = build_steering_spec("impulsive", layers=[20], alpha=1.0, **base)
    assert len({a.spec_hash, stronger.spec_hash, deeper.spec_hash}) == 3


def test_steering_spec_is_untrained() -> None:
    """Alpha is a strength curve, not a dose-response curve; train_scale must stay 1.0."""
    spec = build_steering_spec("impulsive", vector_uri="org/v", layers=[14], alpha=3.0)
    assert spec.train_scale == 1.0
    assert spec.method == "steering"
    assert spec.method_config["alpha"] == 3.0


def test_build_normalised_steering_specs_covers_the_matrix() -> None:
    """personas x alphas x position scopings, every cell its own organism (DEPRECATED builder).

    Both scopings are BUILDABLE even though `configs/specs.yaml` registers only one
    (`all` since 2026-09-25; `response_only` before, which the lens backend does not
    implement). The `all` row previously existed as the matched partner of the
    bias-in-weights variant (a bias has no positions, so `all` was the only scoping it
    could be compared against on equal terms), was deregistered on 2026-08-28 and the
    variant deleted on 2026-09-03. The builder still covers the cell, and
    `scripts/train_persona_probes.steering_config_for` derives the all-positions config it
    needs from the registered row. If these ever collapsed to one spec_hash, the two arms
    would overwrite each other's Parquet files.
    """
    specs = build_normalised_steering_specs(alphas=[0.5, 1.0])
    # The 7B sweep matrix (personas x alphas x positions) PLUS the fixed set of larger-model
    # organisms, which sit at their own swept operating point and ignore the alpha/position
    # sweep args (LARGE_MODEL_STEERING).
    sweep_7b = len(steering_personas()) * 2 * len(DEFAULT_POSITIONS)
    assert len(specs) == sweep_7b + len(LARGE_MODEL_STEERING)
    assert len({s.spec_hash for s in specs}) == len(specs)
    # Every organism uses DEFAULT_POSITIONS (`all` since the 2026-09-25 vllm-lens re-scope),
    # the 7B sweep and the large models alike.
    assert {s.method_config["positions"] for s in specs} == set(DEFAULT_POSITIONS)


def test_position_scoping_changes_the_spec_hash() -> None:
    """`positions` changes what the organism *is*, not how it is served."""
    common = {"vector_uri": "org/v", "layers": [14], "alpha": 1.0}
    scoped = build_steering_spec("impulsive", positions="response_only", **common)
    every = build_steering_spec("impulsive", positions="all", **common)
    assert scoped.spec_hash != every.spec_hash


# ------------------------------------------------------------------ raw-α steering (registered)

RAW_SWEEP_CONFIG = Path("configs/sweeps/steering_raw_alpha_impulsive.yaml")

# The organisms the full eval ran under (HF impulsive-qwen_2_5-7b-14b-32b-big_eval_results,
# run impulsive-qwen_2_5-7b-14b-32b-20261002-steering-raw-alpha-judged): organism folder
# (family_hash) -> the per-seed spec_hash folders, read from that run's spec.json files.
# Promoting the raw-α points must not mint lookalikes of these, or every stored cell orphans.
EVALUATED_RAW_ALPHA: dict[str, tuple[str, dict[int, str]]] = {
    "dd52881e1e399391": (
        "Qwen/Qwen2.5-7B-Instruct",
        {0: "149ce2ef84b1c0d6", 1: "2970414ad37a6fb6", 2: "63d89f23e571f163"},
    ),
    "d858b35a7f056663": (
        "Qwen/Qwen2.5-14B-Instruct",
        {0: "8ca274dea0052273", 1: "b663b8d3ec16d13c", 2: "3a94ca4b1dd614c8"},
    ),
    "c9f1481a726637f9": (
        "Qwen/Qwen2.5-32B-Instruct",
        {0: "f05a0dbc002d6a8a", 1: "bc503863e379bd8c", 2: "c270c0cb47783c40"},
    ),
    # Gemma-4-31B raw α=1.75 (the Gemma-4-31B steering sweep, 2026-10-07; seed 0
    # was swept, seeds 1-2 are the registered replicates). The knee under the #395 rule.
    "32199dd7b549307e": (
        "google/gemma-4-31B-it",
        {0: "8e5cdb247b8175e3", 1: "574123e56b6ce560", 2: "9402d65f19cb3a65"},
    ),
}


def _default_steering_by_base() -> dict[str, Any]:
    """The DEPRECATED normalised impulsive steering organism per base model."""
    return {
        s.base_model: s for s in build_normalised_steering_specs() if s.persona_id == "impulsive"
    }


def test_raw_delta_is_alpha_times_the_raw_vector() -> None:
    """`normalize: false` adds alpha * v; alpha 1 is exactly the extracted direction."""
    from serving.hf_backend import SteeringConfig

    vec = {"layer_3": [3.0, 4.0]}
    raw = SteeringConfig(vector=vec, layers=[3], alpha=1.5, normalize=False, positions="all")
    assert raw.delta(3) == [4.5, 6.0]
    # The normalised magnitude m is the raw multiple m / ||v||: the map the sweep relies on.
    unit = SteeringConfig(vector=vec, layers=[3], alpha=10.0, normalize=True, positions="all")
    as_raw = SteeringConfig(vector=vec, layers=[3], alpha=10.0 / 5.0, normalize=False)
    assert unit.delta(3) == pytest.approx(as_raw.delta(3))


def test_raw_vectors_pin_the_default_layer_and_magnitude() -> None:
    """RAW_VECTORS must describe the default organism it maps to raw units."""
    defaults = _default_steering_by_base()
    assert set(RAW_VECTORS) == set(RAW_ALPHA_SWEEP) == set(defaults)
    for base_model, raw in RAW_VECTORS.items():
        default = defaults[base_model]
        assert default.method_config["layers"] == [raw.layer]
        assert default.method_config["alpha"] == raw.normalised_alpha
        assert len(raw.vector_sha256) == 64 and len(raw.revision) == 40
    multiples = {b: round(r.default_as_raw_multiple, 2) for b, r in RAW_VECTORS.items()}
    assert multiples == {
        "Qwen/Qwen2.5-7B-Instruct": 4.03,
        "Qwen/Qwen2.5-14B-Instruct": 3.91,
        "Qwen/Qwen2.5-32B-Instruct": 6.79,
        "google/gemma-4-31B-it": 2.1,  # 70 / 33.3221; the knee that FAILS #395 (sweep below it)
    }


def test_raw_spec_pins_vector_and_differs_from_the_default() -> None:
    spec = build_raw_steering_spec("impulsive", "Qwen/Qwen2.5-14B-Instruct", 2.0)
    cfg = spec.method_config
    raw = RAW_VECTORS["Qwen/Qwen2.5-14B-Instruct"]
    assert cfg["normalize"] is False and cfg["alpha"] == 2.0 and cfg["positions"] == "all"
    assert cfg["revision"] == raw.revision
    assert cfg["extraction"] == {"vector_sha256": raw.vector_sha256}
    assert spec.base_model == "Qwen/Qwen2.5-14B-Instruct"
    with pytest.raises(ValueError, match="no pinned raw CAA vector"):
        build_raw_steering_spec("impulsive", "Qwen/Qwen2.5-72B-Instruct", 1.0)


def test_raw_spec_pins_the_personas_own_vector() -> None:
    """A raw organism pins ITS persona's vector: loyalty's 7B row is the evaluated one, and a
    persona with no vector on a base refuses instead of borrowing impulsive's."""
    spec = build_raw_steering_spec("loyalty", "Qwen/Qwen2.5-7B-Instruct", 8.0)
    assert spec.spec_hash == EVALUATED_LOYALTY_STEERING["8cc1f32ba6a7e2ca"][2][0]
    assert spec.method_config["revision"] != RAW_VECTORS["Qwen/Qwen2.5-7B-Instruct"].revision
    with pytest.raises(ValueError, match="no pinned raw CAA vector for 'loyalty'"):
        build_raw_steering_spec("loyalty", "google/gemma-4-31B-it", 1.0)


def test_registered_steering_is_the_evaluated_raw_alpha_run() -> None:
    """Decision 2026-10-03: the registered steering rows ARE the raw-α organisms the full eval
    ran under — same family_hash and the same spec_hash per seed — and nothing else."""
    from serving.runner import expand_seeds, load_config

    config = load_config("configs/specs.yaml")
    # impulsive only: loyalty's steering rows are pinned by
    # test_registered_loyalty_steering_is_the_evaluated_run below (their own vectors and alphas,
    # not this sweep's).
    registered = [
        s for s in load_specs(config) if s.method == "steering" and s.persona_id == "impulsive"
    ]
    assert {s.spec_hash for s in registered} == {
        s.spec_hash for s in build_steering_specs() if s.persona_id == "impulsive"
    }
    # Exhaustive over EVERY persona: a steering row is either impulsive's chosen raw-alpha
    # point or one of loyalty's evaluated rows, and none is normalised. A new persona's
    # steering row fails here until it is pinned like loyalty's. Since the behaviour registry
    # (2026-10-12) the builder produces loyalty's rows too, from loyalty's own vectors.
    every = [s for s in load_specs(config) if s.method == "steering"]
    loyalty_seed0 = {seeds[0] for _base, _alpha, seeds in EVALUATED_LOYALTY_STEERING.values()}
    assert {s.spec_hash for s in every} == {s.spec_hash for s in build_steering_specs()}
    assert {s.spec_hash for s in build_steering_specs() if s.persona_id == "loyalty"} == (
        loyalty_seed0
    )
    assert all(s.method_config["normalize"] is False for s in every)
    assert {s.family_hash: s.base_model for s in registered} == {
        family: base for family, (base, _seeds) in EVALUATED_RAW_ALPHA.items()
    }
    by_seed = {(s.family_hash, s.seed): s.spec_hash for s in expand_seeds(registered, config)}
    assert by_seed == {
        (family, seed): spec_hash
        for family, (_base, seeds) in EVALUATED_RAW_ALPHA.items()
        for seed, spec_hash in seeds.items()
    }
    # Every registered steering row is raw (none of the deprecated normalised ones), and the
    # registered set is exactly the chosen points of the sweep.
    assert all(s.method_config["normalize"] is False for s in registered)
    sweep = {s.spec_hash for s in build_raw_alpha_sweep_specs()}
    assert sweep & {s.spec_hash for s in load_specs(config)} == {s.spec_hash for s in registered}


# Loyalty's raw-alpha steering rows (registered 2026-10-09), keyed family_hash -> (base, alpha,
# {seed: spec_hash}), as evaluated in science-of-model-organisms/loyalty-qwen_2_5-7b-14b-32b-
# big_eval_results runs/loyalty-qwen_2_5-{7b,14b}-20261007 and runs/loyalty-qwen_2_5-32b-20261008.
EVALUATED_LOYALTY_STEERING: dict[str, tuple[str, float, dict[int, str]]] = {
    "8cc1f32ba6a7e2ca": (
        "Qwen/Qwen2.5-7B-Instruct",
        8.0,
        {0: "a003cd3ec07c136a", 1: "7b176f83eed4caae", 2: "bbfcd0552328ff0f"},
    ),
    "a805136d1ede043d": (
        "Qwen/Qwen2.5-14B-Instruct",
        10.0,
        {0: "d8ccaa785db78752", 1: "70e21920c852d13d", 2: "c2e612ab853def5d"},
    ),
    "92eff3b676db2a66": (
        "Qwen/Qwen2.5-32B-Instruct",
        15.0,
        {0: "e0c8d7cebc615737", 1: "c207378ac845b634", 2: "d30d7b4919ab0255"},
    ),
}


def test_registered_loyalty_steering_is_the_evaluated_run() -> None:
    """The loyalty steering rows carry the hashes their full eval ran under, seed by seed."""
    from serving.runner import expand_seeds, load_config

    config = load_config("configs/specs.yaml")
    registered = [
        s for s in load_specs(config) if s.method == "steering" and s.persona_id == "loyalty"
    ]
    assert {s.family_hash: (s.base_model, s.method_config["alpha"]) for s in registered} == {
        family: (base, alpha) for family, (base, alpha, _) in EVALUATED_LOYALTY_STEERING.items()
    }
    by_seed = {(s.family_hash, s.seed): s.spec_hash for s in expand_seeds(registered, config)}
    assert by_seed == {
        (family, seed): spec_hash
        for family, (_base, _alpha, seeds) in EVALUATED_LOYALTY_STEERING.items()
        for seed, spec_hash in seeds.items()
    }
    assert all(s.method_config["normalize"] is False for s in registered)


def test_deprecated_normalised_steering_is_not_registered() -> None:
    """The normalised 24 / 40 / 200 organisms (and LARGE_MODEL_STEERING) left specs.yaml."""
    registered = {s.spec_hash for s in load_specs("configs/specs.yaml")}
    assert not {s.spec_hash for s in build_normalised_steering_specs()} & registered
    for _persona, base_model, layer, alpha, uploader in LARGE_MODEL_STEERING:
        assert RAW_VECTORS[base_model].layer == layer
        assert RAW_VECTORS[base_model].normalised_alpha == alpha
        assert RAW_VECTORS[base_model].uploader == uploader


def test_raw_sweep_config_matches_the_builders() -> None:
    """The sweep file is exactly: base + the default organism + RAW_ALPHA_SWEEP, per size."""
    specs = load_specs(RAW_SWEEP_CONFIG)
    steering = [s for s in specs if s.method == "steering"]
    raw = {s.spec_hash for s in steering if not s.method_config["normalize"]}
    assert raw == {s.spec_hash for s in build_raw_alpha_sweep_specs()}
    normalised = {s.spec_hash for s in steering if s.method_config["normalize"]}
    # The comparator: byte-identical to the DEPRECATED normalised organisms the sweep was run
    # against (registered at the time), so it IS that organism.
    assert normalised == {s.spec_hash for s in _default_steering_by_base().values()}
    bases = sorted(s.base_model for s in specs if s.method == "base")
    assert bases == sorted(RAW_VECTORS)
    assert len(specs) == len(raw) + len(normalised) + len(bases)


def test_raw_sweep_config_uses_the_shared_judge_and_100_unseen_items() -> None:
    from evals.simple_behaviour_exhibit import items
    from serving.runner import load_config

    sweep, main = load_config(RAW_SWEEP_CONFIG), load_config("configs/specs.yaml")
    assert sweep["eval_configs"]["default"] == main["eval_configs"]["default"]
    assert sweep["default_evals"] == ["simple_behaviour_exhibit"]
    assert sweep["seeds"] == [0]
    sbe = sweep["eval_configs"]["simple_behaviour_exhibit"]
    chosen = items({"persona_id": "impulsive", **sbe})
    assert len(chosen) == 100
    # T1 held-out pools only: unseen for every method, so steering and base pair cleanly.
    assert {i.meta["pool"] for i in chosen} == {"diag_v1", "diag_v2"}
    assert all(not i.meta.get("trained_on") for i in chosen)


def test_write_vector_keys_each_files_sha_separately(tmp_path: Path) -> None:
    """The extractor writes vector + neutral into one metadata.json; neither sha may clobber."""
    import hashlib

    out = write_vector({"layer_0": [1.0]}, tmp_path, {"algo": "caa"})
    write_vector({"layer_0": [2.0]}, tmp_path, {}, filename="vector_neutral.safetensors")
    meta = json.loads((out / "metadata.json").read_text())

    def sha(name: str) -> str:
        return hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()

    assert meta["vector_sha256"] == sha("vector.safetensors")
    assert meta["vector_neutral_sha256"] == sha("vector_neutral.safetensors")
    assert sha256_key("vector.safetensors") == "vector_sha256"


def test_resolve_verifies_the_declared_vector_sha(tmp_path: Path) -> None:
    """A vector whose bytes differ from `extraction.vector_sha256` must not be served."""
    import hashlib

    from serving.resolve import _steering_config

    write_vector({"layer_0": [3.0, 4.0]}, tmp_path, {})
    good = hashlib.sha256((tmp_path / "vector.safetensors").read_bytes()).hexdigest()

    def spec(sha: str | None) -> Any:
        return build_steering_spec(
            "impulsive",
            vector_uri=str(tmp_path),
            layers=[0],
            alpha=2.0,
            normalize=False,
            extraction={"vector_sha256": sha} if sha else None,
        )

    ok = spec(good)
    assert _steering_config(ok, ok.method_config).delta(0) == [6.0, 8.0]
    bad = spec("0" * 64)
    with pytest.raises(ValueError, match="vector artifact changed"):
        _steering_config(bad, bad.method_config)
    unpinned = spec(None)  # every default row: no sha declared, nothing checked
    assert _steering_config(unpinned, unpinned.method_config).alpha == 2.0


def test_chosen_raw_points_are_the_sweeps_organisms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chosen point must be the SAME organism (spec_hash) as its sweep row, so promoting
    it reuses the sweep's rows rather than minting a lookalike."""
    import implant.steering as st

    sweep = {s.spec_hash for s in build_raw_alpha_sweep_specs()}
    picks = tuple(("impulsive", b, alphas[-1]) for b, alphas in RAW_ALPHA_SWEEP.items())
    monkeypatch.setattr(st, "RAW_ALPHA_STEERING", picks)
    chosen = st.build_steering_specs()
    assert len(chosen) == len(picks)
    assert {s.spec_hash for s in chosen} <= sweep


def test_the_registered_choices_are_sweep_organisms() -> None:
    """The REAL RAW_ALPHA_STEERING picks must be rows the sweep measured (same spec_hash), so
    the chosen organism is the one the operating-point rule read, not an unmeasured lookalike."""
    assert RAW_ALPHA_STEERING, "chosen points are registered"
    sweep = {s.spec_hash for s in build_raw_alpha_sweep_specs()}
    # impulsive's: the sweep is impulsive's (loyalty's points are pinned by hash above)
    for persona, base_model, alpha in RAW_ALPHA_STEERING:
        if persona == "impulsive":
            assert alpha in RAW_ALPHA_SWEEP[base_model], (base_model, alpha)
    assert {s.spec_hash for s in build_steering_specs() if s.persona_id == "impulsive"} <= sweep


def test_paper_figures_show_the_registered_steering_organisms() -> None:
    """The paper's steering families (`families: steering` in configs/dashboard_build.yaml, what
    the paper figures draw as Steering) must be the registered raw-α families of every persona
    (impulsive's sweep points and loyalty's pinned rows), or the figures silently drop or
    mislabel steering."""
    pytest.importorskip("matplotlib")
    from analysis.paper_figures import load_families

    config = Path(__file__).resolve().parent.parent / "configs" / "dashboard_build.yaml"
    STEERING_FAMILIES = load_families(config)["steering"]

    assert set(STEERING_FAMILIES) == set(EVALUATED_RAW_ALPHA) | set(EVALUATED_LOYALTY_STEERING)
    assert set(STEERING_FAMILIES) == {s.family_hash for s in build_steering_specs()} | set(
        EVALUATED_LOYALTY_STEERING
    )
