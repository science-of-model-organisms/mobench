"""The probe trainer: the arithmetic, the artifact contract, and the unsteered-capture trap.

Two layers, the same split `docs/evals/activations.md` asks of the eval side:

1. **The maths against planted signals.** A separable class pair must produce a direction
   that separates it; a shuffled-label null must not. These run with no endpoint, no GPU and
   no interp-engine.
2. **The wiring, duck-typed.** A fake `interp_engine` module stands in for the real one, so
   the capture path — including "pass `steering_spec=` and never open a `steer()` context",
   which is the bug that would otherwise read as a plausible null — is pinned without a
   pod. What that cannot prove is that the real engine honours the spec the way we read it;
   only a GPU parity run settles that.

The end-to-end test is the one that matters most: it runs the trainer in fake mode and reads
the result back with `evals._whitebox.load_probe`. If that cannot read what the trainer
wrote, the artifact is wrong regardless of how good the numbers look.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from evals._whitebox import auroc, load_base_reference, load_probe, norm
from implant.generators.contrastive import NEG_FIELD, POS_FIELD, shuffled_control_pairs
from scripts.train_persona_probes import (
    DEFAULT_BANKS,
    DEFAULT_PINNED_LAYER,
    MATRIX_NEGATIVES,
    PROBE_REPO_NAME,
    SELFCHECK_REL_TOL,
    Condition,
    Dataset,
    ProbeVariant,
    Row,
    _reduce_tensor,
    acceptance,
    adapter_support,
    base_reference_selfcheck,
    build_parser,
    cosine_matrix,
    dataset_for,
    default_probe_repo,
    diff_of_means,
    dump_condition,
    fit_logistic,
    icl_conditions,
    main,
    pinned_point,
    project,
    shuffled_dataset,
    split_questions,
    split_wordings,
)


def _planted(n: int, width: int, direction: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """`n` rows of noise with `strength * direction` added. A signal we know the answer to."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, width)) + strength * direction


# ------------------------------------------------------------------------------- maths


def test_diff_of_means_recovers_the_planted_direction() -> None:
    width = 32
    planted = np.zeros(width)
    planted[3] = 1.0
    pos = _planted(80, width, planted, 3.0, seed=1)
    neg = _planted(80, width, planted, 0.0, seed=2)

    direction, bias = diff_of_means(pos, neg)

    assert norm(direction.tolist()) == pytest.approx(1.0)
    assert float(direction @ planted) > 0.8
    # The bias sits at the midpoint between the class means, so the two classes straddle the
    # 0.5 decision boundary rather than both landing on one side of it.
    assert float(direction @ pos.mean(axis=0)) + bias > 0.0
    assert float(direction @ neg.mean(axis=0)) + bias < 0.0


def test_a_shuffled_label_null_does_not_separate_what_the_real_probe_does() -> None:
    """The control that says the pipeline measured disposition and not formatting."""
    width = 24
    planted = np.zeros(width)
    planted[5] = 1.0
    data = Dataset(
        pos=_planted(60, width, planted, 4.0, seed=3),
        neg=_planted(60, width, planted, 0.0, seed=4),
        question_ids=tuple(range(60)),
    )
    real, _ = diff_of_means(data.pos, data.neg)
    null_data = shuffled_dataset(data, seed=0)
    null, _ = diff_of_means(null_data.pos, null_data.neg)

    real_auroc = auroc(project(real, data.pos), project(real, data.neg))
    null_auroc = auroc(project(null, data.pos), project(null, data.neg))
    assert real_auroc > 0.95
    assert abs(null_auroc - 0.5) < 0.25


def test_the_unbalanced_null_is_the_repo_s_own_control_verbatim() -> None:
    """Not a permutation of the negatives — `contrastive.py` explains why that would pass
    while proving nothing: every positive would still be positive."""
    data = Dataset(
        pos=np.arange(10, dtype=np.float64).reshape(5, 2),
        neg=-np.arange(10, dtype=np.float64).reshape(5, 2),
        question_ids=tuple(range(5)),
    )
    swapped = shuffled_control_pairs([{POS_FIELD: i, NEG_FIELD: i} for i in range(5)], seed=0)
    null = shuffled_dataset(data, seed=0, balanced=False)
    for i, row in enumerate(swapped):
        expected = data.neg[i] if row["label_swapped"] else data.pos[i]
        assert null.pos[i] == pytest.approx(expected)


def test_the_shipped_null_balances_the_swap_so_it_survives_normalisation() -> None:
    """Independent coin flips leave a sqrt(n) residue that IS a multiple of the real
    direction. The CAA control reports a norm ratio, so that residue reads as near zero
    there — but a probe artifact stores UNIT directions, and normalising rescales the
    residue to length 1. The null would then point along +/- the probe and fire exactly as
    hard as it does, which is a control that can never fail."""
    data = Dataset(
        pos=np.arange(1, 21, dtype=np.float64).reshape(10, 2),
        neg=-np.arange(1, 21, dtype=np.float64).reshape(10, 2),
        question_ids=tuple(range(10)),
    )
    null = shuffled_dataset(data, seed=0)
    swapped = [i for i in range(10) if tuple(null.pos[i]) == tuple(data.neg[i])]
    assert len(swapped) == 5
    # Still a swap and never a mix: every row is one of the two originals.
    assert {tuple(r) for r in null.pos} | {tuple(r) for r in null.neg} == {
        tuple(r) for r in data.pos
    } | {tuple(r) for r in data.neg}
    # And the systematic term is gone: the null's difference-of-means has a near-zero norm
    # against the real one, which is `extract_caa_vectors.control_norm_ratio`'s pass
    # condition read on the balanced construction.
    real = data.pos.mean(axis=0) - data.neg.mean(axis=0)
    assert float(np.linalg.norm(null.pos.mean(axis=0) - null.neg.mean(axis=0))) < 0.5 * float(
        np.linalg.norm(real)
    )


def test_fit_logistic_separates_a_planted_signal_in_raw_activation_space() -> None:
    """The fit standardises internally and folds back, so `(w, b)` is a functional of the
    RAW activation — no hidden preprocessing an eval would have to reproduce."""
    width = 16
    planted = np.zeros(width)
    planted[2] = 1.0
    # An offset and a scale per feature, so a fit that forgot to fold back would fail here.
    pos = _planted(60, width, planted, 2.5, seed=5) * 4.0 + 100.0
    neg = _planted(60, width, planted, 0.0, seed=6) * 4.0 + 100.0

    w, b = fit_logistic(
        np.vstack([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    )

    assert auroc(project(w, pos), project(w, neg)) > 0.95
    assert float(np.mean(pos @ w + b)) > 0.0 > float(np.mean(neg @ w + b))


def test_fit_logistic_is_deterministic() -> None:
    """No initialisation noise, no shuffling: every stochastic call takes a seed and this
    one has none because it makes none (CLAUDE.md)."""
    features = _planted(40, 8, np.eye(8)[1], 1.5, seed=7)
    labels = np.array([1.0] * 20 + [0.0] * 20)
    first = fit_logistic(features, labels)
    second = fit_logistic(features, labels)
    assert first[0] == pytest.approx(second[0])
    assert first[1] == pytest.approx(second[1])


def test_dataset_for_matches_on_the_question_and_drops_what_is_unmatched() -> None:
    """Matching on question index is control 1 made mechanical: the same question on both
    sides, so the direction cannot be a topic direction."""

    def rows(ids: list[int], value: float) -> list[Row]:
        return [
            Row(
                question_idx=i,
                question=f"q{i}",
                response="x",
                acts={"prompt": {"resid_post.0": np.full(3, value)}},
            )
            for i in ids
        ]

    data = dataset_for(
        rows([0, 1, 2], 1.0), rows([1, 2, 3], 0.0), arm="prompt", point="resid_post.0"
    )
    assert data.question_ids == (1, 2)
    assert len(data.pos) == len(data.neg) == 2


def test_split_questions_is_deterministic_disjoint_and_complete() -> None:
    ids = list(range(50))
    train, held = split_questions(ids, holdout=0.3, seed=0)
    assert split_questions(ids, holdout=0.3, seed=0) == (train, held)
    assert not set(train) & set(held)
    assert sorted(train + held) == ids
    assert len(held) == 15


def test_split_wordings_reports_no_holdout_when_the_repo_has_one_wording() -> None:
    """`data/personas/*.contrast_prompts.json` carries exactly one pos/neg pair today, so
    the wording axis of the three-way split is NOT exercised. The trainer records that as
    `wording_holdout: false` rather than letting a reader assume otherwise."""
    single = [Condition("impulsive.prompt.w0", "prompt", "impulsive", "pos")]
    assert split_wordings(single) == ([], single)

    pair = [*single, Condition("impulsive.prompt.w1", "prompt", "impulsive", "pos")]
    train, held = split_wordings(pair)
    assert [c.key for c in train] == ["impulsive.prompt.w0"]
    assert [c.key for c in held] == ["impulsive.prompt.w1"]


def test_pinned_point_refuses_the_final_layer() -> None:
    """Post-`model.norm` on HF, pre-norm as `resid_post.(n-1)` on interp-engine — the two
    artifacts would be indistinguishable afterwards, so this raises rather than warns."""
    points = [f"resid_post.{i}" for i in range(4)]
    assert pinned_point(points, 2) == "resid_post.2"
    with pytest.raises(ValueError, match="final layer"):
        pinned_point(points, 3)


def test_pinned_point_falls_back_when_the_model_is_smaller_than_the_operating_point() -> None:
    assert pinned_point([f"resid_post.{i}" for i in range(4)], 16) == "resid_post.2"


def test_cosine_matrix_is_symmetric_and_unit_on_the_diagonal() -> None:
    """The organism-free diagnostic: no serving, no calibration, so directions
    that agree while scores fail to transfer localise the fault to calibration."""
    variants = [
        ProbeVariant(
            "p", "prompt", "neutral", "prompt", directions={"resid_post.0": np.eye(3)[0]}
        ),
        ProbeVariant("p", "icl", "neutral", "prompt", directions={"resid_post.0": np.eye(3)[1]}),
    ]
    matrix = cosine_matrix(variants, "resid_post.0")
    ids = [v.probe_id for v in variants]
    assert matrix[ids[0]][ids[0]] == pytest.approx(1.0)
    assert matrix[ids[0]][ids[1]] == pytest.approx(0.0)
    assert matrix[ids[0]][ids[1]] == matrix[ids[1]][ids[0]]


# -------------------------------------------------------------------------- acceptance


def _variant(probe_auroc: float) -> ProbeVariant:
    return ProbeVariant(
        "p",
        "prompt",
        "neutral",
        "prompt",
        directions={"resid_post.0": np.eye(2)[0]},
        biases={"resid_post.0": 0.0},
        auroc_cross={"resid_post.0": probe_auroc},
    )


def test_an_unmeasured_anchor_is_not_a_pass() -> None:
    """A criterion that was never evaluated cannot license a report. Collapsing None to True
    is exactly how an unmeasured probe ends up in a figure."""
    variant = _variant(0.95)
    rows = acceptance(
        [variant],
        point="resid_post.0",
        anchor=None,
        base_probability={variant.probe_id: 0.2},
        floor=0.75,
    )
    row = rows[variant.probe_id]
    assert row["reportable"] is False
    assert row["not_measured"] == ["steering_anchor"]


def test_a_probe_below_the_auroc_floor_is_not_reportable_even_with_a_good_anchor() -> None:
    variant = _variant(0.60)
    anchor = {
        "alpha": 24.0,
        "probes": {variant.probe_id: {"recovered_shift": 24.0, "within_tolerance": True}},
    }
    rows = acceptance(
        [variant],
        point="resid_post.0",
        anchor=anchor,
        base_probability={variant.probe_id: 0.2},
        floor=0.75,
    )
    assert rows[variant.probe_id]["reportable"] is False
    assert rows[variant.probe_id]["checks"]["auroc_cross_elicitation"]["ok"] is False


def test_all_three_criteria_together_are_what_makes_a_probe_reportable() -> None:
    variant = _variant(0.88)
    anchor = {
        "alpha": 24.0,
        "probes": {variant.probe_id: {"recovered_shift": 23.0, "within_tolerance": True}},
    }
    rows = acceptance(
        [variant],
        point="resid_post.0",
        anchor=anchor,
        base_probability={variant.probe_id: 0.31},
        floor=0.75,
    )
    assert rows[variant.probe_id]["reportable"] is True


# ------------------------------------------------------- the capture path, duck-typed


def test_the_anchors_steering_config_is_derived_not_hunted_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchor's config must come back all-positions for a registered persona, not raise.

    `steering_config_for` used to FILTER `configs/specs.yaml` for a row whose `positions`
    was already `all`. That row was deregistered on 2026-08-28, so from that day the lookup
    matched nothing and raised `LookupError` for every persona — which the caller catches
    and logs as "steering anchor NOT measured", indistinguishable from "this persona has no
    vector extracted". The arbiter of the whole probe acceptance gate went unmeasured for
    six days and nothing failed, so the regression is pinned here rather than left to the
    caller's log line.

    Both halves matter: that it does not raise, and that the scoping it hands on is `all` —
    `steering_anchor` explains why a response-only capture dilutes the recovered shift by a
    factor that varies with item length.

    `_steering_config` is stubbed because it downloads and reads the vector artifact, which
    is not what this asks about and would put a network fetch in the gate. What is under
    test is the row selection and the one field it overrides.
    """
    from scripts.train_persona_probes import steering_config_for
    from serving.runner import load_specs

    # `serving/__init__` re-exports the `resolve` FUNCTION, which shadows the module on
    # attribute lookup — go through sys.modules for the module itself.
    resolve_module = sys.modules["serving.resolve"]

    seen: list[dict[str, Any]] = []

    def _record(spec: Any, cfg: dict[str, Any]) -> Any:
        seen.append(cfg)
        return SimpleNamespace(**cfg)

    monkeypatch.setattr(resolve_module, "_steering_config", _record)

    registered = {
        spec.persona_id for spec in load_specs("configs/specs.yaml") if spec.method == "steering"
    }
    assert registered, "no steering rows in configs/specs.yaml for the anchor to read"

    for persona_id in sorted(registered):
        config = steering_config_for(persona_id)
        assert config.positions == "all", persona_id
    # Derived from the registered row rather than retyped here, so the anchor is checked
    # against the intervention the sweep actually serves.
    assert len(seen) == len(registered)
    assert all(cfg.get("vector_uri") and cfg.get("layers") and cfg.get("alpha") for cfg in seen)


def test_adapter_support_reports_the_vllm_lens_verdict() -> None:
    """Post-migration: vllm-lens CAN capture through a LoRA (native lora_request), so the
    diagnostic records the backend and its capability, gated off pending the GPU canary
    (Stage 3b) rather than blocked as it was on interp-engine. The steered-capture failure
    mode the old InterpReader tests pinned is now GPU-covered (validation runbook Stage 3c),
    not a CPU unit — interp-engine and its steer-vs-capture trap were removed."""

    answer = adapter_support(object())  # `model` is ignored now; kept for caller compat
    assert answer["supported"] is False  # flip to True after Stage 3b confirms LoRA capture
    assert answer["evidence"]["backend"] == "vllm-lens"
    assert "lora" in answer["answer"].lower()


# ------------------------------------------------------------------------ end to end


def test_the_icl_prefix_refuses_to_truncate() -> None:
    """`serving/resolve._load_icl_prefix` raises for the same reason: a truncated prefix is
    a different organism under the same name."""
    with pytest.raises(ValueError, match="holds only"):
        icl_conditions("impulsive", 10_000)


@pytest.fixture(scope="module")
def trained_fake_probes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """ONE full fake-mode trainer run for the whole module, WITH --push.

    The trainer refuses to push AFTER run(args) has written every artefact, so a single
    --push run yields both things this module needs: the artefact tree the contract test
    reads, and the refusal the push test asserts. The two separate full runs this
    replaces were ~54s each — 26% of the entire suite's wall clock (2026-08-30 audit).

    EVERY BANK IN `DEFAULT_BANKS`, AND DO NOT NARROW IT WITH `--banks`. An earlier version
    of this branch passed `--banks pressure_items`, reasoning that `pressure_items` is the
    only bank any assertion here READS. Review rejected that, correctly: unread is not
    unexercised. This is the only test anywhere that calls `build_base_reference` over the
    default set (`grep -rn 'build_base_reference\\|DEFAULT_BANKS' tests/` finds nothing
    else), so narrowing it silently dropped the sole coverage of
    `pressure_items_adversarial`'s `ADVERSARIAL_CONDITION` expansion and of the `denial=True`
    path — a malformed denial or adversarial bank would have failed this fixture before and
    would then have sailed through. The flag bought under 0.5s, measured (2026-09-10), so
    there was nothing on the other side of the trade.
    """
    out = tmp_path_factory.mktemp("probes-mod") / "probes"
    mp = pytest.MonkeyPatch()
    mp.setenv("MO_EVALS_FAKE", "1")
    try:
        with pytest.raises(SystemExit, match="refusing to push FAKE"):
            main(
                [
                    "impulsive",
                    "--out-dir",
                    str(out),
                    "--fake-layers",
                    "4",
                    "--layer",
                    "1",
                    "--max-tokens",
                    "8",
                    "--icl-k",
                    "2",
                    "--push",
                ]
            )
    finally:
        mp.undo()
    return out


def test_the_trainer_writes_an_artifact_that_evals_whitebox_can_read(
    trained_fake_probes: Path,
) -> None:
    """The acceptance test for the artifact contract. Fake mode, so the NUMBERS are
    meaningless by construction — what is asserted is the shape, the keys and
    the loadability, which is exactly what a GPU run cannot afford to discover late."""
    out = trained_fake_probes

    persona = out / "impulsive"
    probe = load_probe(persona)
    assert probe.persona_id == "impulsive"
    assert probe.elicitation == "prompt"
    assert probe.read_point == "resid_post.1"
    assert norm(probe.direction) == pytest.approx(1.0)
    # Not the final layer, even in fake mode.
    assert probe.read_point != "resid_post.3"

    meta = json.loads((persona / "probe.json").read_text())
    for key in (
        "probe_id",
        "elicitation",
        "negative_class",
        "read_arm",
        "auroc_cross_elicitation",
        "auroc_heldout_item",
        "bias",
        "base_model",
        "chat_template_sha256",
        "layer_convention",
        "n_train_questions",
        "length_stats",
    ):
        assert key in meta, key
    assert meta["fake"] is True
    assert set(meta["auroc_cross_elicitation"]) == {f"resid_post.{i}" for i in range(4)}

    # Every (elicitation x negative class x read arm) probe, each independently loadable —
    # the two pinned families are what the probe-transfer matrix reads, plus the
    # `anti` anchor diagnostic, which exists only for the prompt elicitation (there is no
    # anti-ICL demo file) and is marked `matrix_eligible: false`. 10: the four
    # `other_persona` variants are fitted against the `mathematical` contrast persona in
    # `data/probe_contrast/`.
    assert len(meta["variants"]) == 10
    assert meta["matrix_eligible"] is True
    eligible = {}
    for name in meta["variants"]:
        variant = load_probe(persona / "variants" / name)
        assert norm(variant.direction) == pytest.approx(1.0)
        eligible[name] = variant.meta["matrix_eligible"]
    assert [n for n, ok in eligible.items() if not ok] == [
        "prompt__anti__prompt",
        "prompt__anti__response",
    ]

    # The paired base reference, and the null.
    reference = load_base_reference(persona, "pressure_items", "prompt")
    assert reference and len(next(iter(reference.values()))) == probe.width
    assert (persona / "base_reference" / "pressure_items__responses.json").exists()
    assert (persona / "probe_shuffled.safetensors").exists()

    # EVERY bank ran, not just the one read above. This is the only test in the repo that
    # exercises `build_base_reference` across the default set, so without this line a
    # `--banks` narrowing silently drops the sole coverage of `pressure_items_adversarial`'s
    # ADVERSARIAL_CONDITION expansion and of the `denial=True` path — which is exactly what
    # an earlier version of this branch did, and what review caught (2026-09-10). Asserted
    # on the artefact rather than on the arg list, so it also catches the bank that loads
    # and then writes nothing.
    assert {b.name for b in DEFAULT_BANKS} == {
        p.name.removesuffix("__responses.json")
        for p in (persona / "base_reference").glob("*__responses.json")
    }

    # Raw outputs are always stored: a probe is a summary of these transcripts, and
    # re-fitting from them costs nothing where re-generating them costs a GPU pass.
    assert (persona / "transcripts" / "neutral.jsonl").exists()

    diagnostics = json.loads((persona / "diagnostics.json").read_text())
    assert diagnostics["steering_anchor"] is None  # no GPU, so nothing to anchor against
    assert all(row["reportable"] is False for row in diagnostics["acceptance"].values())


def test_fake_probes_are_never_pushed(trained_fake_probes: Path) -> None:
    """The same refusal `scripts/extract_caa_vectors.py` makes, for the same reason.

    The SystemExit itself is asserted inside the `trained_fake_probes` fixture (its run
    passes --push); reaching this test means the refusal held. What THIS test pins is
    the ordering contract the fixture leans on: the trainer writes everything BEFORE
    refusing, so a fake run is inspectable even though it can never publish."""
    assert (trained_fake_probes / "impulsive" / "probe.json").exists()
    assert (trained_fake_probes / "impulsive" / "transcripts" / "neutral.jsonl").exists()


def test_the_default_pin_without_a_contrast_persona_fails_before_any_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `data/probe_contrast/` absent and no `--other`, impulsive and loyalty both have ICL
    demos but the default `other_persona` is the contrast persona, so the pin has no negative
    class. The run must stop at argument checking, not after the dump and fits:
    `build_endpoint` is never reached."""
    import scripts.train_persona_probes as tpp

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setattr(tpp, "PROBE_CONTRAST_DIR", tmp_path / "absent")
    # The unrelated real behaviours must not silently become each other's default class.
    monkeypatch.setattr(tpp, "available_personas", lambda *a, **k: ["impulsive"])

    def _no_endpoint(*_a: object, **_k: object) -> None:
        raise AssertionError("model work started before the pin check")

    monkeypatch.setattr(tpp, "build_endpoint", _no_endpoint)
    with pytest.raises(SystemExit, match="no other persona with ICL demos"):
        main(["impulsive", "--out-dir", str(tmp_path / "probes")])


def test_the_default_pin_is_the_variant_that_can_actually_be_scored_with() -> None:
    """The pinned variant is the convenience copy at the top of the persona directory, and
    it is what `config['probe'] = <persona dir>` resolves to — and what `push` publishes as
    the repo's headline path.

    It was `neutral`/`prompt`, which FAILS acceptance: it recovers 19.2% of alpha while
    scoring AUROC 1.000, because it detects "is there a system prompt" rather than the
    trait. That default survived the change of scored arm to `response` and was only caught
    when the first real upload published a confounded direction under the default path
    (2026-08-25). This test is what should have caught it.

    `other_persona`/`response` is the cell that clears all three gates. If a future run
    makes a different cell the reportable one, change this deliberately — but the default
    must never be a variant that cannot be legitimately scored with.
    """
    args = build_parser().parse_args([])

    assert args.layer is None and DEFAULT_PINNED_LAYER == 16, "not the final layer"
    assert (args.pin_elicitation, args.pin_negative, args.pin_arm) == (
        "prompt",
        "other_persona",
        "response",
    )
    assert math.isclose(args.shift_tolerance, 0.25)


def test_the_pinned_default_is_a_matrix_eligible_family() -> None:
    """`anti` can never be the pin: it is a diagnostic, structurally NaN on the gate, and
    marked `matrix_eligible: false`. Pinning it would publish an unscoreable probe."""
    args = build_parser().parse_args([])

    assert args.pin_negative in MATRIX_NEGATIVES


# --------------------------------------------- --base-model / --repo mismatch footgun


def test_mismatched_base_model_without_explicit_repo_is_refused() -> None:
    """`--base-model` can select a 32B (or 14B) run while `--repo` silently keeps its
    default — the 7B HF repo `default_probe_repo()` names. Pushed, that would overwrite the 7B
    artefacts with 32B ones under the same repo id. Refuse before any training starts."""
    with pytest.raises(SystemExit, match="does not match"):
        main(["impulsive", "--base-model", "Qwen/Qwen2.5-32B-Instruct"])


def test_matched_base_model_without_explicit_repo_is_allowed() -> None:
    """`--base-model` set to the persona files' own base model is not a mismatch, even
    with no `--repo` — this must not raise past argument parsing."""
    args = build_parser().parse_args(["impulsive", "--base-model", "Qwen/Qwen2.5-7B-Instruct"])
    assert args.base_model == "Qwen/Qwen2.5-7B-Instruct"


def test_mismatched_base_model_with_explicit_repo_is_allowed() -> None:
    """An explicit `--repo` is the user's own call; the guard must not block it."""
    args = build_parser().parse_args(
        ["impulsive", "--base-model", "Qwen/Qwen2.5-32B-Instruct", "--repo", "org/32b-repo"]
    )
    assert args.repo == "org/32b-repo"


# ------------------------------------------------- the response arm reads response tokens


def test_reduce_tensor_start_drops_the_prompt_positions() -> None:
    """The `response` arm must mean over the RESPONSE tokens, not the whole conversation.

    On the impulsive persona the system prompt is ~1523 chars against a ~487 char reply, so
    a whole-conversation mean is ~3/4 system-prompt tokens — and those differ between the
    two classes by construction. The direction that falls out separates the training classes
    perfectly while sitting nearly orthogonal to the trait, which is what the first real run
    measured (AUROC 1.000, alpha recovery 13-42%).
    """
    rows = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 20.0], [30.0, 40.0]])

    whole, n_whole = _reduce_tensor(rows)
    response, n_response = _reduce_tensor(rows, start=2)

    assert n_whole == 4
    assert list(whole) == [10.0, 15.0]  # dragged toward the zero prompt rows
    assert n_response == 2, "the reported count is the response span, not the conversation"
    assert list(response) == [20.0, 30.0]


def test_reduce_tensor_leaves_an_already_reduced_vector_alone() -> None:
    """`HFEndpoint.hidden_states` returns `[width]`, ALREADY meaned over the assistant turn.

    Slicing that would drop the vector entirely, so `start` must not apply to a 1-D input.
    """
    reduced = np.array([1.0, 2.0, 3.0])

    vector, n = _reduce_tensor(reduced, start=2)

    assert list(vector) == [1.0, 2.0, 3.0]
    assert n == 1


def test_reduce_tensor_refuses_an_empty_slice(caplog: pytest.LogCaptureFixture) -> None:
    """`start` is measured on a separate render of the prompt, so a chat template that
    re-renders the prefix could put it past the end. A mean over nothing is a NaN vector
    that propagates silently into a direction; say so and fall back to the last position."""
    rows = np.array([[1.0, 1.0], [4.0, 6.0]])

    with caplog.at_level("WARNING"):
        vector, n = _reduce_tensor(rows, start=5)

    assert not np.isnan(vector).any(), "an empty slice would make the whole direction NaN"
    assert list(vector) == [4.0, 6.0]
    assert n == 1
    assert "no response positions" in caplog.text.lower()


def test_dump_condition_reads_the_response_arm_from_the_prompt_length() -> None:
    """End to end: the response arm's `start` is the prompt arm's own position count.

    `_tokens` continues a trailing assistant turn rather than closing it, so the prompt
    render IS the sealed render's prefix — the same way `serving/hf_backend._response_start`
    measures it, and the same tokens CAA reads (`serving/hf_backend.py:421`).
    """
    seen: list[dict[str, Any]] = []

    class _CountingReader:
        """Returns one row per message, so position counts are predictable."""

        def read(
            self,
            messages: list[dict[str, str]],
            *,
            start: int = 0,
            limit: int | None = None,
        ) -> tuple[dict[str, np.ndarray], int]:
            seen.append({"n_messages": len(messages), "start": start, "limit": limit})
            n = len(messages)
            rows = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
            vector, count = _reduce_tensor(rows, start=start, limit=limit)
            return {"resid_post.0": vector}, count

    class _Endpoint:
        def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
            return "an answer"

    condition = Condition(key="c", elicitation="prompt", persona_id="p", polarity="pos")
    rows = dump_condition(
        condition,
        ["q"],
        endpoint=_Endpoint(),
        reader=_CountingReader(),
        max_tokens=8,
        seed=0,
    )

    prompt_read, response_read = seen
    assert prompt_read["start"] == 0, "the prompt arm reads from the beginning"
    assert response_read["start"] == prompt_read["n_messages"], (
        "the response arm must start where the prompt render ended"
    )
    # One message of prompt, two of sealed -> exactly one response position.
    assert rows[0].n_positions == {"prompt": 1, "response": 1}


# ------------------------------------------------- the anchor grades each family fairly


def test_family_ceilings_scale_the_bar_by_the_two_caa_contrasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trait-vs-ABSENT probe cannot recover alpha off a trait-vs-INVERSE organism.

    `induced_shift = alpha * cos(probe, v_steering)`. The steering organism is built from
    the trait-vs-inverse vector, so a flawless `neutral` probe still only reaches
    `alpha * cos(v_steering, v_neutral)` — 0.680 on impulsive, measured 2026-08-25. Holding
    it to alpha under a 25% tolerance demanded 0.75 from something capped at 0.68: a failure
    that says nothing about the probe.
    """
    import scripts.train_persona_probes as tp
    from serving.hf_backend import SteeringConfig

    steer = np.array([1.0, 0.0])
    neutral = np.array([0.6, 0.8])  # cos with steer = 0.6, exactly

    monkeypatch.setattr(tp, "load_named_vector", lambda *a, **k: {"layer_7": list(neutral)})
    steering = SteeringConfig(vector={"layer_7": list(steer)}, layers=[7], alpha=24.0)

    ceilings = tp.family_ceilings(steering, "resid_post.7", "impulsive")

    assert ceilings["anti"] == 1.0, "`anti` IS the steering contrast; its bar stays alpha"
    assert ceilings["neutral"] == pytest.approx(0.6)
    assert ceilings["other_persona"] == pytest.approx(0.6)


def test_family_ceilings_never_loosen_the_bar_when_the_target_is_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unverifiable ceiling must fall back to the STRICTER bar, never the looser one.

    Defaulting to a low ceiling on a missing file would quietly let a bad probe through,
    which is the failure direction that matters here.
    """
    import scripts.train_persona_probes as tp
    from serving.hf_backend import SteeringConfig

    def _missing(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("no vector_neutral.safetensors")

    monkeypatch.setattr(tp, "load_named_vector", _missing)
    steering = SteeringConfig(vector={"layer_7": [1.0, 0.0]}, layers=[7], alpha=24.0)

    with caplog.at_level("WARNING"):
        ceilings = tp.family_ceilings(steering, "resid_post.7", "impulsive")

    assert ceilings == {"anti": 1.0, "neutral": 1.0, "other_persona": 1.0}
    assert "stricter" in caplog.text.lower()


# --------------------------------------------------------------- the probe question bank


def test_probe_questions_extends_the_constitution_without_touching_it() -> None:
    """More items is the legitimate way to move a pre-registered gate; lowering it is not.

    The constitution holds 50 questions and that is its whole supply, so the AUROC gate was
    being estimated on 15 held-out items — and the first clean run landed at 0.733 against a
    0.75 floor, well inside that noise.
    """
    from scripts.train_persona_probes import probe_questions

    base = probe_questions("impulsive", extra=False)
    extended = probe_questions("impulsive")

    assert len(base) == 50, "the constitution's supply, unchanged"
    assert len(extended) > len(base)
    assert extended[: len(base)] == base, "the constitution's questions come first, in order"


def test_the_probe_bank_is_not_in_the_constitution() -> None:
    """`implant/generators/demos.constitution_questions` feeds few_shot ICL prefixes and
    every trained method's data. Adding there would change what every organism is built on and
    invalidate adapters already pushed to the HF org, so the bank must stay separate."""
    from implant.generators.demos import constitution_questions
    from implant.generators.system_prompt import load_persona
    from scripts.train_persona_probes import probe_questions

    constitution = set(constitution_questions(load_persona("impulsive")))
    extras = [q for q in probe_questions("impulsive") if q not in constitution]

    assert extras, "the bank must actually contribute questions"
    assert not (set(extras) & constitution), "no question may be in both"


def test_the_probe_bank_names_no_trait() -> None:
    """Hygiene on the questions this bank ADDS: none of them names the trait.

    This is not a label leak — every arm is asked the SAME question, so the question text
    cannot separate the classes. The reason to keep trait words out is narrower: a
    trait-named prompt makes the answer differ by TOPIC as well as by disposition, so a
    probe fitted across such items partly tracks "is this question about impulsiveness"
    rather than "is this model impulsive".

    The CONSTITUTION already carries three ("a fun, impulsive thing", "spontaneous road
    trips", "a spontaneous travel destination"), which is why this asserts on the ADDED
    items only. They are deliberately not filtered: they feed the trained methods and
    few_shot ICL too, so dropping them here would make the probe's item set differ from the one the
    organisms were built from — the larger of the two problems. Known limitation, recorded
    in the phase-3 note rather than silently patched.
    """
    from implant.generators.demos import constitution_questions
    from implant.generators.system_prompt import load_persona
    from scripts.train_persona_probes import probe_questions

    loaded = ("impulsive", "impulse", "spontaneous", "reckless", "cautious", "careful")
    constitution = set(constitution_questions(load_persona("impulsive")))
    added = [q for q in probe_questions("impulsive") if q not in constitution]
    offenders = [q for q in added if any(w in q.lower() for w in loaded)]

    assert added, "the bank must actually contribute questions"
    assert not offenders, f"trait-named questions added by this bank: {offenders}"


def test_probe_questions_are_shared_across_personas_for_control_1() -> None:
    """Probe-comparability control: the SAME questions on every arm and every elicitation, so
    the only thing differing between the classes is the elicitation itself."""
    from implant.generators.demos import constitution_questions
    from implant.generators.system_prompt import load_persona
    from scripts.train_persona_probes import probe_questions

    def bank(pid: str) -> list[str]:
        constitution = set(constitution_questions(load_persona(pid)))
        return [q for q in probe_questions(pid) if q not in constitution]

    assert bank("impulsive") == bank("loyalty")


# ------------------------------------------------------- the AUROC interval, reported only


def test_auroc_ci_brackets_the_point_estimate_and_narrows_with_n() -> None:
    """More items is the legitimate way to firm up a marginal gate; lowering it is not."""
    import random as _random

    from evals._whitebox import auroc, auroc_ci

    def sample(n: int) -> tuple[list[float], list[float]]:
        rng = _random.Random(0)
        return (
            [rng.gauss(0.6, 1.0) for _ in range(n)],
            [rng.gauss(0.0, 1.0) for _ in range(n)],
        )

    small_pos, small_neg = sample(33)
    big_pos, big_neg = sample(94)
    small = auroc_ci(small_pos, small_neg)
    big = auroc_ci(big_pos, big_neg)

    assert small[0] <= auroc(small_pos, small_neg) <= small[1]
    assert (big[1] - big[0]) < (small[1] - small[0]), "94 held-out items must beat 33"


def test_auroc_by_ranks_equals_the_pairwise_double_loop_exactly() -> None:
    """`auroc` is the Mann-Whitney rank identity, not the O(n*m) loop it replaced.

    The loop is the definition, so it stays here as the oracle. EXACTLY, not approx: this
    number is rounded into `probe.json` and compared against a pre-registered floor, so a
    last-bit difference would be a silently different artifact. Ties are the case that
    matters — they are where an average-rank implementation drifts — hence the integer
    regime below, where most comparisons are ties.
    """
    import random as _random

    from evals._whitebox import auroc

    def pairwise(positive: list[float], negative: list[float]) -> float:
        wins = 0.0
        for p in positive:
            for n in negative:
                wins += 1.0 if p > n else 0.5 if p == n else 0.0
        return wins / (len(positive) * len(negative))

    rng = _random.Random(0)
    for trial in range(200):
        n_p, n_n = rng.randint(1, 25), rng.randint(1, 25)
        if trial % 2:  # tie-heavy: a 4-value alphabet over up to 25 draws
            pos = [float(rng.randint(0, 3)) for _ in range(n_p)]
            neg = [float(rng.randint(0, 3)) for _ in range(n_n)]
        else:
            pos = [rng.gauss(1.0, 1.0) for _ in range(n_p)]
            neg = [rng.gauss(0.0, 1.0) for _ in range(n_n)]
        assert auroc(pos, neg) == pairwise(pos, neg), (n_p, n_n, trial)

    # The all-tied and fully-separated corners, where the identity is easiest to get wrong.
    assert auroc([5.0] * 7, [5.0] * 9) == 0.5
    assert auroc([2.0], [1.0]) == 1.0
    assert auroc([1.0], [2.0]) == 0.0
    assert math.isnan(auroc([], [1.0])) and math.isnan(auroc([1.0], []))

    # NaN, WHICH THE RANK IDENTITY GETS WRONG UNLESS IT IS FILTERED FIRST. Every
    # comparison against NaN is False, so the loop scores those pairs 0 while still
    # counting them in the denominator; argsort instead puts NaN last, which would hand a
    # NaN positive the top rank and score it a win. Caught in review: the first version of
    # this function returned 1.0 for the first case below. +/-inf and -0.0 are NOT special
    # — they sort and compare the same way under both — and are pinned here so nobody
    # "fixes" them into the NaN branch.
    nan, inf = float("nan"), float("inf")
    for pos, neg in (
        ([nan], [1.0]),
        ([nan], [1.0, nan]),
        ([1.0, nan], [0.0]),
        ([nan, nan], [nan, nan]),
        ([1.0, 2.0], [nan]),
        ([inf], [1.0]),
        ([-inf], [1.0]),
        ([inf], [inf]),
        ([-0.0], [0.0]),
        ([1.0, inf, -inf], [0.0, inf]),
    ):
        assert auroc(pos, neg) == pairwise(pos, neg), (pos, neg)


def test_auroc_ci_is_deterministic_under_the_spec_seed() -> None:
    """No bare `random` anywhere in this project (CLAUDE.md): two runs of one spec must
    agree, or the recorded interval is not a property of the probe."""
    from evals._whitebox import auroc_ci

    pos = [0.9, 0.4, 0.7, 0.2, 0.8]
    neg = [0.1, 0.5, 0.3, 0.6, 0.0]

    assert auroc_ci(pos, neg, seed=7) == auroc_ci(pos, neg, seed=7)
    assert auroc_ci(pos, neg, seed=7) != auroc_ci(pos, neg, seed=8)


def test_a_pass_whose_interval_straddles_the_floor_is_flagged_marginal() -> None:
    """The n=110 run cleared 0.75 by 0.005 while its interval was ~0.14 wide, and nothing in
    the artifact said so. The GATE stays on the point estimate — the floor is pre-registered
    and moving it onto a bound after seeing the data is what pre-registering it rules
    out — but a pass that could have gone either way must not read as a clean one."""
    from scripts.train_persona_probes import _straddles

    assert _straddles((0.70, 0.81), 0.75) is True, "floor inside the interval"
    assert _straddles((0.78, 0.92), 0.75) is False, "cleanly above"
    assert _straddles((0.55, 0.69), 0.75) is False, "cleanly below"
    assert _straddles(None, 0.75) is None, "unmeasured spread is not 'not marginal'"
    assert _straddles((float("nan"), float("nan")), 0.75) is None


def test_the_response_cap_equalises_the_span_each_class_contributes() -> None:
    """The length-confound control, done rather than merely reported.

    After the response-only fix the two classes still differed in how MUCH they wrote — 74
    mean tokens against 135 on the winning variant. The mean itself is scale-free, so this
    is not an arithmetic bias; the confound is that a longer answer contains different KINDS
    of content, and a probe fitted across that partly tracks verbosity.
    """
    long_rows = np.array([[1.0, 1.0]] * 3 + [[9.0, 9.0]] * 7)
    short_rows = np.array([[1.0, 1.0]] * 3)

    capped_long, n_long = _reduce_tensor(long_rows, limit=3)
    capped_short, n_short = _reduce_tensor(short_rows, limit=3)

    assert n_long == n_short == 3, "both classes contribute the same number of positions"
    assert list(capped_long) == list(capped_short)
    # Without the cap the long arm is dragged by its extra content, which is the confound.
    assert list(_reduce_tensor(long_rows)[0]) != list(capped_long)


def test_the_cap_leaves_a_response_shorter_than_it_alone() -> None:
    """A short answer must not be padded or dropped — the cap is a ceiling, not a target."""
    rows = np.array([[2.0, 4.0], [4.0, 8.0]])

    vector, n = _reduce_tensor(rows, limit=10)

    assert n == 2
    assert list(vector) == [3.0, 6.0]


# --------------------------------------------------------------------- GPU pre-flight


def test_the_gpu_guard_refuses_a_busy_device_with_the_holder_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM's own check fires late and fifteen times over, burying the one-line cause.

    Measured 2026-08-25: an orphaned EngineCore held 40 GiB, and the launched run produced
    fifteen full tracebacks before giving up. The reason (one stale process) was in none of
    them prominently.
    """
    import subprocess as sp

    import scripts.train_persona_probes as tp

    class _Done:
        def __init__(self, out: str) -> None:
            self.stdout = out

    def _fake_run(cmd: list[str], **kw: Any) -> Any:
        if "--query-gpu=memory.used,memory.total" in " ".join(cmd):
            return _Done("40100, 45403\n")
        return _Done("200341, 40100 MiB\n")

    monkeypatch.setattr(sp, "run", _fake_run)
    monkeypatch.setattr(tp, "subprocess", sp)

    class _Cfg:
        gpu_memory_utilization = 0.9

    with pytest.raises(SystemExit) as excinfo:
        tp.assert_gpu_is_free(_Cfg())

    message = str(excinfo.value)
    assert "40100" in message, "say how much is in use"
    assert "200341" in message, "name what is holding it, or the message is not actionable"


def test_the_gpu_guard_allows_a_free_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    import scripts.train_persona_probes as tp

    class _Done:
        stdout = "0, 45403\n"

    monkeypatch.setattr(sp, "run", lambda *a, **k: _Done())
    monkeypatch.setattr(tp, "subprocess", sp)

    class _Cfg:
        gpu_memory_utilization = 0.9

    tp.assert_gpu_is_free(_Cfg())  # must not raise


def test_the_gpu_guard_stays_out_of_the_way_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No nvidia-smi means a laptop or a CPU box; that is not this function's problem and it
    must not turn a fake-mode run into a failure."""
    import subprocess as sp

    import scripts.train_persona_probes as tp

    def _missing(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(sp, "run", _missing)
    monkeypatch.setattr(tp, "subprocess", sp)

    class _Cfg:
        gpu_memory_utilization = 0.9

    tp.assert_gpu_is_free(_Cfg())  # must not raise


# ------------------------------------------------------------------ what may be published


def _summary(**variants: bool) -> dict[str, Any]:
    return {
        "acceptance": {
            f"impulsive__{k}": {
                "reportable": v,
                "checks": {"auroc_cross_elicitation": {"marginal": False}},
            }
            for k, v in variants.items()
        }
    }


def test_push_refuses_a_persona_with_nothing_reportable() -> None:
    """An artifact in the shared org is an invitation: someone points specs.yaml at it and
    reads what comes back. `calibrated_score` does return NaN below the floor rather than a
    number, so nothing silently lies — but a repo of probes that can only return NaN implies
    a confidence the run did not earn."""
    from scripts.train_persona_probes import assert_worth_pushing

    with pytest.raises(SystemExit) as excinfo:
        assert_worth_pushing("impulsive", _summary(a__b__c=False, d__e__f=False))

    assert "not one of its 2 variants" in str(excinfo.value)
    assert "do not publish it" in str(excinfo.value)


def test_push_keeps_only_the_variants_that_cleared_acceptance() -> None:
    from scripts.train_persona_probes import reportable_variants

    keep = reportable_variants(
        _summary(prompt__other_persona__response=True, icl__neutral__response=False)
    )

    assert keep == ["impulsive__prompt__other_persona__response"]


def test_push_warns_when_a_passing_variant_is_marginal(caplog: pytest.LogCaptureFixture) -> None:
    """A pass whose interval straddles the floor is publishable but must not travel silently:
    every number derived from it has to carry the flag."""
    from scripts.train_persona_probes import assert_worth_pushing

    summary = _summary(prompt__other_persona__response=True)
    checks = summary["acceptance"]["impulsive__prompt__other_persona__response"]["checks"]
    checks["auroc_cross_elicitation"]["marginal"] = True

    with caplog.at_level("WARNING"):
        keep = assert_worth_pushing("impulsive", summary)

    assert keep == ["impulsive__prompt__other_persona__response"]
    assert "MARGINAL" in caplog.text


def test_the_anchor_reads_the_vector_extracted_on_the_probes_own_base_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """From 2026-09-07 a persona has one steering row PER base model (7B/14B/32B). The
    anchor for a probe fitted on the 14B must read the 14B vector: the 7B one is a width
    mismatch against 5120-dim activations, and the caller would log it as "anchor NOT
    measured" — the same silent unmeasured gate the test above pins."""
    from scripts.train_persona_probes import steering_config_for

    resolve_module = sys.modules["serving.resolve"]
    picked: list[Any] = []
    monkeypatch.setattr(
        resolve_module, "_steering_config", lambda spec, cfg: picked.append(spec) or cfg
    )

    steering_config_for("impulsive", base_model="Qwen/Qwen2.5-14B-Instruct")
    assert picked[-1].base_model == "Qwen/Qwen2.5-14B-Instruct"
    steering_config_for("impulsive")  # no base model: the first row, as before
    assert picked[-1].base_model == "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(LookupError, match="on Qwen/Qwen2.5-999B"):
        steering_config_for("impulsive", base_model="Qwen/Qwen2.5-999B-Instruct")


# ----------------------------------------------------------------- base-reference selfcheck


def _selfcheck(tmp_path: Path, relative_noise: float) -> dict[str, Any]:
    """Store a reference with norm ~200, then re-capture it with `relative_noise` of ||h||."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    ref = rng.normal(size=64)
    ref *= 200.0 / np.linalg.norm(ref)
    noise = rng.normal(size=64)
    noise *= relative_noise * 200.0 / np.linalg.norm(noise)
    directory = tmp_path / "base_reference"
    directory.mkdir()
    save_file({"i0": ref.astype(np.float32)}, str(directory / "b__prompt.safetensors"))
    item = SimpleNamespace(id="i0", user_content=lambda: "hi")
    reader = SimpleNamespace(read=lambda _messages: [{"p": ref + noise}])
    return base_reference_selfcheck(
        tmp_path,
        bank="b",
        reader=reader,
        point="p",
        items=[item],  # duck-typed stand-ins
    )


def test_selfcheck_passes_half_a_percent_batch_noise_and_records_both_deltas(
    tmp_path: Path,
) -> None:
    out = _selfcheck(tmp_path, 0.005)
    assert out["ok"] is True
    assert out["l2_delta"] > 1e-3  # the old absolute test would have voided this
    assert out["rel_delta"] == pytest.approx(0.005, rel=0.01)
    assert out["rel_tolerance"] == SELFCHECK_REL_TOL


def test_selfcheck_fails_five_percent_noise_and_records_both_deltas(tmp_path: Path) -> None:
    out = _selfcheck(tmp_path, 0.05)
    assert out["ok"] is False
    assert out["rel_delta"] == pytest.approx(0.05, rel=0.01)
    assert out["l2_delta"] == pytest.approx(0.05 * out["ref_norm"], rel=0.01)


def test_selfcheck_passes_two_percent_layer_27_batch_noise(tmp_path: Path) -> None:
    """Review (Opus): batch noise reached 1.76% at layer 27 on GPU; the tolerance must clear it."""
    assert _selfcheck(tmp_path, 0.02)["ok"] is True


def test_a_nan_reference_never_reproduces(tmp_path: Path) -> None:
    """Review (Opus): a NaN reference made rel_delta 0.0 and passed; it must fail."""
    from safetensors.numpy import save_file

    directory = tmp_path / "base_reference"
    directory.mkdir()
    ref = np.full(8, np.nan, dtype=np.float32)
    save_file({"i0": ref}, str(directory / "b__prompt.safetensors"))
    item = SimpleNamespace(id="i0", user_content=lambda: "hi")
    reader = SimpleNamespace(read=lambda _messages: [{"p": np.ones(8)}])
    out = base_reference_selfcheck(tmp_path, bank="b", reader=reader, point="p", items=[item])
    assert out["ok"] is False


def test_push_default_follows_hf_org(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_HF_ORG", "Fork-Org")
    assert default_probe_repo() == f"Fork-Org/{PROBE_REPO_NAME}"
    monkeypatch.delenv("MO_EVALS_HF_ORG")
    assert default_probe_repo() == f"science-of-model-organisms/{PROBE_REPO_NAME}"
