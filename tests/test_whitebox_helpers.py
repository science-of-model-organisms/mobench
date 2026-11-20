"""The white-box probe ARITHMETIC in `evals/_whitebox.py`, which outlived the cells using it.

The six trained-probe `whitebox_probe_*` cells ("V0") were removed on 2026-10-14 as invalid
(docs/evals/activations.md). What stays in `evals/_whitebox.py` is (a) the generic capture
helpers any activations cell reads through (`reduce_capture`, `n_positions`,
`supports_activations`) and (b) the probe arithmetic and artifact loading the RETIRED trainer
`scripts/train_persona_probes.py` still uses to reproduce historical probe artifacts. These are
the pure-function tests that lived in the deleted `tests/test_inspect_whitebox.py`; no Inspect
task is built here any more."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from evals._whitebox import (
    DEFAULT_AUROC_FLOOR,
    Probe,
    auroc,
    cosine,
    dot,
    induced_shift,
    load_probe,
    n_positions,
    normalise,
    projection,
    reduce_capture,
    supports_activations,
)

WIDTH = 8


# The planted direction: the trait lives entirely on axis 0. Any probe that recovers it
# scores separably, and any probe that does not is broken in a way these tests should see.
TRAIT_AXIS = 0


def _probe(auroc_value: float = 0.9, elicitation: str = "prompt", **kw: Any) -> Probe:
    direction = [0.0] * WIDTH
    direction[TRAIT_AXIS] = 1.0
    return Probe(
        probe_id=kw.pop("probe_id", "test-probe"),
        direction=normalise(direction),
        read_point="resid_post.16",
        auroc=auroc_value,
        elicitation=elicitation,
        persona_id="impulsive",
        **kw,
    )


def _activation(signal: float, noise: float = 0.0) -> list[float]:
    """One activation carrying `signal` on the trait axis and `noise` everywhere else."""
    row = [noise] * WIDTH
    row[TRAIT_AXIS] = signal
    return row


def _write_probe(
    directory: Path,
    *,
    elicitation: str = "prompt",
    auroc_value: float = 0.9,
    read_arm: str | None = None,
) -> Path:
    """Write a probe artifact. ``read_arm`` is omitted by default, which is what a probe
    written before the field existed looks like — the arm guard must leave those alone."""
    import numpy as np
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    direction = np.zeros(WIDTH, dtype=np.float32)
    direction[TRAIT_AXIS] = 1.0
    save_file(
        {"resid_post.16": direction, "resid_post.4": direction * 0.5},
        str(directory / "probe.safetensors"),
    )
    (directory / "probe.json").write_text(
        json.dumps(
            {
                "probe_id": "impulsive-prompt",
                "read_point": "resid_post.16",
                "elicitation": elicitation,
                "persona_id": "impulsive",
                "auroc_cross_elicitation": {"resid_post.16": auroc_value, "resid_post.4": 0.6},
                "bias": {"resid_post.16": 0.0},
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "layer_convention": "resid_post.i = output of model.layers[i]",
                **({"read_arm": read_arm} if read_arm else {}),
            }
        )
    )
    return directory


def test_projection_reads_the_planted_signal_and_ignores_the_rest() -> None:
    """The probe must respond to its own axis and not to activity elsewhere.

    The failure this catches is a probe that scores on overall magnitude — which every
    "unusual conversation" would then trip, and which would read as the trait.
    """
    probe = _probe()
    assert projection(probe, _activation(3.0)) == pytest.approx(3.0)
    # Same total magnitude, none of it on the trait axis.
    off_axis = [0.0] * WIDTH
    off_axis[TRAIT_AXIS + 1] = 3.0
    assert projection(probe, off_axis) == pytest.approx(0.0)


def test_induced_shift_removes_the_arbitrary_offset() -> None:
    """The point of induced shift: a probe direction's offset must not survive into the number.

    Two organisms sitting at different absolute projections but the same distance from
    base must report the same shift — otherwise cross-organism comparison is comparing
    offsets, which is what raw projection does.
    """
    probe = _probe()
    base = _activation(5.0)
    assert induced_shift(probe, _activation(7.0), base) == pytest.approx(2.0)

    shifted_base = _activation(100.0)
    assert induced_shift(probe, _activation(102.0), shifted_base) == pytest.approx(2.0)


def test_induced_shift_is_nan_without_a_base_reference() -> None:
    """No silent degradation to raw projection.

    Falling back would put an offset-carrying number in the column a figure treats as a
    difference, and nothing downstream could tell the two apart.
    """
    assert math.isnan(induced_shift(_probe(), _activation(3.0), None))


def test_base_organism_scores_zero_shift_against_itself() -> None:
    """The run-level self-check of induced shift, in miniature.

    If the base organism does not land at ~0 in a real run, the reference was built against
    a different model, chat template or read point — and every shift in the run is wrong by
    a constant nobody would notice.
    """
    base = _activation(4.2, noise=0.3)
    assert induced_shift(_probe(), base, base) == pytest.approx(0.0)


def test_steering_guard_recovers_alpha() -> None:
    """A steering organism is base + alpha*direction BY CONSTRUCTION, so a valid probe
    recovers alpha. This is the arithmetic behind the steered-capture guard — the one that
    catches a capture which silently missed the steer and reads as "trait not encoded".
    """
    probe = _probe()
    alpha = 24.0
    base = _activation(1.0)
    steered = [b + alpha * d for b, d in zip(base, probe.direction, strict=True)]
    assert induced_shift(probe, steered, base) == pytest.approx(alpha)

    # And the failure mode it exists to catch: the capture came back unsteered.
    assert induced_shift(probe, base, base) == pytest.approx(0.0)


def test_the_floor_is_the_pre_registered_value() -> None:
    """Pinned so that raising it to make a result appear is a visible diff."""
    assert DEFAULT_AUROC_FLOOR == 0.75


def test_auroc_separates_a_planted_signal_and_refuses_a_flat_one() -> None:
    """The discrimination test `FakeEndpoint(persona=None)` would give us if its
    activations carried a signal. Here it is done on arrays instead.
    """
    positive = [projection(_probe(), _activation(s)) for s in (2.0, 2.5, 3.0, 3.5)]
    negative = [projection(_probe(), _activation(s)) for s in (-1.0, -0.5, 0.0, 0.5)]
    assert auroc(positive, negative) == pytest.approx(1.0)

    # Flat: the same distribution on both sides. An eval that scored THIS highly would be
    # scoring everything highly, which is the failure the two-mode fake exists to expose.
    flat = [0.0, 0.0, 0.0, 0.0]
    assert auroc(flat, flat) == pytest.approx(0.5)


def test_normalise_removes_scale_so_transfer_is_not_a_norm_artefact() -> None:
    """Probe-commensurability control. Without it, "this probe transfers better" can just
    mean "this probe has a bigger norm", and the probe-transfer matrix is unreadable.
    """
    big = normalise([10.0] + [0.0] * (WIDTH - 1))
    small = normalise([0.1] + [0.0] * (WIDTH - 1))
    assert big == pytest.approx(small)
    assert sum(x * x for x in big) == pytest.approx(1.0)


def test_cosine_is_the_organism_free_consistency_check() -> None:
    """Diagnostic 1: probe-to-probe agreement, with no organism involved."""
    a = [1.0] + [0.0] * (WIDTH - 1)
    b = [0.0, 1.0] + [0.0] * (WIDTH - 2)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)
    assert cosine(a, [2.0] + [0.0] * (WIDTH - 1)) == pytest.approx(1.0)


def test_dot_raises_on_a_width_mismatch() -> None:
    """A width mismatch means the probe and the endpoint disagree about the model. The
    alternative to raising is a broadcast that quietly scores something.
    """
    with pytest.raises(ValueError, match="width mismatch"):
        dot([1.0, 2.0], [1.0, 2.0, 3.0])


def test_reduce_capture_accepts_both_backend_shapes() -> None:
    """`HFEndpoint.hidden_states` means internally and returns [width]; an interp-engine
    capture returns [n_positions, width]. An eval must not have to know which (CONTRIBUTING.md,
    Contracts: Endpoint), so both normalise here.
    """
    already_reduced = [1.0, 2.0, 3.0]
    assert reduce_capture(already_reduced) == pytest.approx((1.0, 2.0, 3.0))
    assert n_positions(already_reduced) == 1

    per_position = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
    assert reduce_capture(per_position) == pytest.approx((2.0, 3.0, 4.0))
    assert n_positions(per_position) == 2


def test_reduce_capture_last_differs_from_mean() -> None:
    """Both reductions are legitimate; silently using a different one for two organisms is
    the silent wrong-layer shape, which is why the choice is stamped on every row.
    """
    per_position = [[0.0, 0.0], [10.0, 10.0]]
    assert reduce_capture(per_position, "mean") == pytest.approx((5.0, 5.0))
    assert reduce_capture(per_position, "last") == pytest.approx((10.0, 10.0))


def test_reduce_capture_rejects_an_unknown_aggregation_and_an_empty_capture() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        reduce_capture([[1.0]], "median")
    with pytest.raises(ValueError, match="empty capture"):
        reduce_capture([])


@pytest.mark.parametrize(
    ("elicitation", "method", "expected"),
    [
        # The conflict: the teacher that produced prompt-elicited training data IS
        # the prompting organism (implant/generators/contrastive.py), so this pair reports
        # train-set accuracy rather than a measurement.
        ("prompt", "prompting", True),
        # The steering family joins it: the served vector is extracted from the same
        # contrastive pairs, so the probe recovers alpha rather than discovering anything.
        ("prompt", "steering", True),
        # ... and these are the rows a prompt-elicited probe CAN speak about.
        ("prompt", "oct_demos", False),
        ("prompt", "few_shot_icl", False),
        ("adapter", "oct_demos", True),
        ("adapter", "prompting", False),
        ("icl", "few_shot_icl", True),
        ("icl", "prompting", False),
    ],
)
def test_train_distribution_pairs_are_flagged(
    elicitation: str, method: str, expected: bool
) -> None:
    """The diagonal of the transfer matrix must be identifiable.

    Flagged, never dropped: a missing diagonal reads as an absent measurement, where a
    greyed one reads as what it is — the probe scored on its own training distribution.
    """
    assert _probe(elicitation=elicitation).is_train_distribution(method) is expected


def test_load_probe_round_trips_and_normalises(tmp_path: Path) -> None:
    probe = load_probe(_write_probe(tmp_path / "p"))
    assert probe.read_point == "resid_post.16"
    assert probe.elicitation == "prompt"
    assert probe.auroc == pytest.approx(0.9)
    assert sum(x * x for x in probe.direction) == pytest.approx(1.0)


def test_load_probe_takes_the_requested_layer_not_the_pinned_one(tmp_path: Path) -> None:
    """A layer sweep is a config change over one artifact, not a re-extraction — and the
    per-layer AUROC has to follow the layer, or the gate reads the wrong number.
    """
    probe = load_probe(_write_probe(tmp_path / "p"), layer="resid_post.4")
    assert probe.read_point == "resid_post.4"
    assert probe.auroc == pytest.approx(0.6)


def test_load_probe_raises_on_a_missing_read_point(tmp_path: Path) -> None:
    """Loud, because a neighbouring layer standing in silently is precisely the silent
    wrong-layer failure: adjacent directions are similar enough that the wrong one still 'works'.
    """
    with pytest.raises(KeyError, match="not in"):
        load_probe(_write_probe(tmp_path / "p"), layer="resid_post.27")


def test_probe_path_error_names_where_to_get_one(tmp_path: Path) -> None:
    from evals._whitebox import probe_path

    with pytest.raises(FileNotFoundError, match="Misalignment-Empirics"):
        probe_path(tmp_path / "nope")


def test_probe_path_pulls_an_hf_repo_uri_and_returns_the_subpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`probe: <org/repo> :: <subpath>` downloads the repo and joins the subpath, so a
    whitebox run resolves the per-size probe repo itself instead of needing it pre-staged.
    The `::` repo-plus-subpath convention is the repo's own (see `dataset_uri` in specs)."""
    import huggingface_hub

    from evals._whitebox import probe_path

    snapshot = tmp_path / "snap"
    (snapshot / "impulsive").mkdir(parents=True)
    seen: dict[str, str] = {}

    def fake_download(repo_id: str, **kw: object) -> str:
        seen["repo"] = repo_id
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    got = probe_path("Misalignment-Empirics/theo_qwen2.5-14b-it_impulsive-probes :: impulsive")
    assert seen["repo"] == "Misalignment-Empirics/theo_qwen2.5-14b-it_impulsive-probes"
    assert got == snapshot / "impulsive"


def test_probe_path_uri_with_a_local_repo_does_not_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `repo :: subpath` whose repo part is already a local directory (pulled, or a test
    fixture) is used in place — no network call."""
    import huggingface_hub

    from evals._whitebox import probe_path

    def boom(*a: object, **k: object) -> str:
        raise AssertionError("must not download when the repo dir exists locally")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    (tmp_path / "impulsive").mkdir()
    assert probe_path(f"{tmp_path} :: impulsive") == tmp_path / "impulsive"


class _NoActivations:
    supports_hidden_states = False


class _Wrapped:
    """A `SystemPromptEndpoint`-shaped wrapper, which forwards to its inner endpoint."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner


def test_supports_activations_sees_through_wrappers() -> None:
    """`resolve()` wraps prompting and few-shot organisms, and those wrappers forward
    `hidden_states` to the inner endpoint — so the capability question has to follow the
    chain rather than stopping at the outermost object.
    """
    assert supports_activations(object()) is True
    assert supports_activations(_NoActivations()) is False
    assert supports_activations(_Wrapped(_NoActivations())) is False
    assert supports_activations(_Wrapped(object())) is True


def test_endpoint_for_returns_the_live_endpoint_behind_an_organism_model() -> None:
    """The solver needs the `Endpoint` itself, because `hidden_states` is not something
    an Inspect `Model` can express — and must get it without widening the protocol."""
    from core.endpoint import FakeEndpoint
    from evals._inspect import bind_endpoint, endpoint_for

    endpoint = FakeEndpoint(persona="impulsive", supports_hidden_states=True)
    with bind_endpoint("wb-endpoint-for", endpoint, 0) as model:
        assert endpoint_for(model) is endpoint


def test_endpoint_for_refuses_a_model_from_another_provider() -> None:
    """A mockllm (or openai) model has no endpoint behind it; guessing would read
    activations off whatever object happened to be there."""
    from inspect_ai.model import get_model

    from evals._inspect import endpoint_for

    with pytest.raises(TypeError, match="mo_organism"):
        endpoint_for(get_model("mockllm/model"))


def test_every_method_named_as_train_distribution_is_still_dispatched() -> None:
    """`_TRAIN_DISTRIBUTION` may name only methods that exist.

    This is the OTHER half of the hand-kept-method-list drift pattern, and it needs the
    opposite assertion from the one the dashboard and figure maps get. Those must COVER
    `METHOD_BACKENDS`, because every method needs a marker. This map must not: it is a
    deliberate exclusion list — the methods a probe of a given elicitation may not be read
    as evidence about — and its `adapter` key has always held only the demonstration-based
    adapters (`{"sft_demos", "oct_demos"}` before 2026-09-03, `{"oct_demos"}` after), never
    the whole of `ADAPTER_METHODS`. Asserting coverage here would grey out four cells of
    the probe-transfer matrix, which is the measurement the plan calls the strongest
    white-box result available to this project.

    So the only safe invariant is the one below: a name in here must still be a real
    method. That is what would have caught this file during the deletion sweep — it named
    `sft_demos` and `steering_weights` for a fortnight after nothing else did.
    """
    from evals._whitebox import _TRAIN_DISTRIBUTION
    from serving.resolve import METHOD_BACKENDS

    named = {m for methods in _TRAIN_DISTRIBUTION.values() for m in methods}
    dead = sorted(named - set(METHOD_BACKENDS))
    assert not dead, (
        f"_TRAIN_DISTRIBUTION names methods `resolve()` no longer dispatches: {dead}. "
        f"A probe cannot be excluded from evidence about a method that does not exist — "
        f"drop them, and do NOT 'fix' this by adding the live methods it omits, which are "
        f"omitted on purpose."
    )
