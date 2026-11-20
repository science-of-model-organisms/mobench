"""Probe trainer on the vllm-lens backend: anchor wiring, raw-alpha arithmetic, ICL file
choice, the contrast persona and CLI steering. FakeEndpoint only; the lens anchor path itself
(vllm-lens really injecting the vector) is NOT GPU-verified by anything here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scripts.train_persona_probes as tpp
from core.endpoint import Endpoint, FakeEndpoint
from scripts.train_persona_probes import (
    ProbeVariant,
    SteeringOverride,
    applied_delta,
    fake_steered_endpoint,
    icl_demos_file,
    main,
    steering_anchor,
    steering_override_from_args,
)

WIDTH = 8
POINT = "resid_post.1"


def _vector_dir(tmp_path: Path, scale: float) -> tuple[Path, np.ndarray]:
    from safetensors.numpy import save_file

    raw = np.arange(1.0, WIDTH + 1.0) * scale
    d = tmp_path / f"vec{scale}"
    d.mkdir()
    save_file({"layer_1": raw.astype(np.float32)}, str(d / "vector.safetensors"))
    return d, raw


def _variant(direction: np.ndarray) -> ProbeVariant:
    return ProbeVariant(
        "impulsive",
        "prompt",
        "anti",
        "prompt",
        directions={POINT: direction / np.linalg.norm(direction)},
        biases={POINT: 0.0},
        auroc_cross={POINT: 0.9},
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_anchor_grades_against_the_applied_shift_not_alpha(
    tmp_path: Path, normalize: bool
) -> None:
    """normalize true: applied = alpha. normalize false (raw alpha, #395): applied = alpha*|v|.
    A perfect `anti` probe (ceiling 1.0) recovers exactly the applied shift in both cases."""
    vec_dir, raw = _vector_dir(tmp_path, scale=3.0)
    alpha = 24.0
    override = SteeringOverride(str(vec_dir), (1,), alpha, normalize=normalize)
    steering = override.config()
    expected_norm = alpha if normalize else alpha * float(np.linalg.norm(raw))
    assert float(np.linalg.norm(applied_delta(steering, POINT))) == pytest.approx(
        expected_norm, rel=1e-5
    )

    out = steering_anchor(
        [_variant(raw)],
        point=POINT,
        steering=steering,
        endpoint=FakeEndpoint(persona=None, supports_hidden_states=True),
        questions=["q1", "q2", "q3"],
        tolerance=0.25,
        persona_id="impulsive",
        vector_dir=vec_dir,
        make_steered=fake_steered_endpoint,
    )
    assert out["capture_was_steered"] is True
    assert out["normalize"] is normalize
    assert out["applied_delta_norm"] == pytest.approx(expected_norm, abs=1e-3)
    row = (
        out["probes"]["impulsive__prompt__anti__prompt"]
        if "impulsive__prompt__anti__prompt" in out["probes"]
        else next(iter(out["probes"].values()))
    )
    assert row["recovered_shift"] == pytest.approx(expected_norm, rel=1e-3)
    assert row["expected"] == pytest.approx(expected_norm, rel=1e-3)
    assert row["within_tolerance"] is True
    assert row["cosine_with_steering_vector"] == pytest.approx(1.0, abs=1e-3)


def test_read_point_upstream_of_every_steered_layer_raises() -> None:
    steering = SimpleNamespace(layers=[5], delta=lambda layer: [1.0])
    with pytest.raises(ValueError, match="upstream"):
        applied_delta(steering, "resid_post.2")


def _fake_run(tmp_path: Path, persona: str, vec_dir: Path, normalize: bool) -> dict[str, Any]:
    out = tmp_path / f"probes-{persona}"
    args = [
        persona, "--out-dir", str(out), "--fake-layers", "4", "--layer", "1",
        "--max-tokens", "8", "--icl-k", "2", "--anchor-items", "3",
        "--steering-vector", str(vec_dir), "--steering-layer", "1", "--steering-alpha", "24",
        "--steering-normalize" if normalize else "--no-steering-normalize",
    ]  # fmt: skip
    main(args)
    return json.loads((out / persona / "diagnostics.json").read_text())


@pytest.mark.parametrize("persona", ["impulsive", "loyalty"])
def test_fake_run_measures_the_anchor_through_the_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persona: str
) -> None:
    """The anchor ran (it was dead code: gated on the always-None interp-engine `model`), via
    CLI steering flags (loyalty has no steering row), with raw alpha, for both behaviours.
    Fake probes are at chance, so acceptance is not asserted; that the anchor is MEASURED is."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    vec_dir, raw = _vector_dir(tmp_path, scale=2.0)
    diag = _fake_run(tmp_path, persona, vec_dir, normalize=False)
    anchor = diag["steering_anchor"]
    assert anchor is not None and anchor["capture_was_steered"] is True
    assert anchor["normalize"] is False
    assert anchor["applied_delta_norm"] == pytest.approx(
        24.0 * float(np.linalg.norm(raw)), abs=1e-2
    )
    for row in diag["acceptance"].values():
        assert "steering_anchor" not in row.get("not_measured", [])


def test_no_steering_flags_and_no_anchor_in_fake_mode_is_skipped_not_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("fake mode must not fetch a registered vector")

    monkeypatch.setattr(tpp, "steering_config_for", _boom)
    out = tmp_path / "p"
    main(["impulsive", "--out-dir", str(out), "--fake-layers", "4", "--layer", "1",
          "--max-tokens", "8", "--icl-k", "2"])  # fmt: skip
    assert (
        json.loads((out / "impulsive" / "diagnostics.json").read_text())["steering_anchor"] is None
    )


def test_partial_steering_flags_are_refused() -> None:
    ns = argparse.Namespace(
        steering_vector="x", steering_layer=None, steering_alpha=None,
        steering_normalize=None, steering_revision=None, steering_subfolder=None,
    )  # fmt: skip
    with pytest.raises(SystemExit, match="steering-layer"):
        steering_override_from_args(ns)
    ns.steering_vector = None
    assert steering_override_from_args(ns) is None


# ------------------------------------------------------------------ ICL demos single source


def test_icl_file_is_the_one_the_registered_organism_serves() -> None:
    from serving.runner import load_specs

    rows = [
        s for s in load_specs("configs/specs.yaml")
        if s.method == "few_shot_icl" and s.persona_id == "impulsive"
    ]  # fmt: skip
    assert rows
    served = {Path(s.method_config["demos_path"]).name for s in rows}
    assert served == {icl_demos_file("impulsive").name} == {"impulsive.icl_demos.luna.jsonl"}


def test_unregistered_persona_uses_the_canonical_teacher_file() -> None:
    assert icl_demos_file("loyalty").name == "loyalty.icl_demos.luna.jsonl"
    assert icl_demos_file("loyalty").exists()


def test_ambiguous_registered_demo_files_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    import serving.runner as runner

    def _row(path: str) -> Any:
        return SimpleNamespace(
            method="few_shot_icl", persona_id="impulsive", base_model="m",
            method_config={"demos_path": path},
        )  # fmt: skip

    monkeypatch.setattr(runner, "load_specs", lambda *_: [_row("a.jsonl"), _row("b.jsonl")])
    with pytest.raises(ValueError, match="ambiguous"):
        icl_demos_file("impulsive")


def test_two_demo_files_in_an_explicit_dir_are_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "x.icl_demos.jsonl").write_text("{}\n")
    (tmp_path / "x.icl_demos.luna.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="ambiguous"):
        icl_demos_file("x", personas_dir=tmp_path)


# ------------------------------------------------------------------ contrast persona


def test_mathematical_is_a_contrast_persona_not_a_behaviour() -> None:
    assert "mathematical" not in tpp.available_personas()
    assert tpp.persona_data_dir("mathematical") == tpp.PROBE_CONTRAST_DIR
    assert tpp.persona_data_dir("impulsive").name == "personas"
    for name in ("json", "contrast_prompts.json", "icl_demos.jsonl", "anti_system_prompt.txt"):
        assert (tpp.PROBE_CONTRAST_DIR / f"mathematical.{name}").exists()
    assert [c.persona_id for c in tpp.prompt_conditions("mathematical")]
    assert tpp.icl_conditions("mathematical", 2)


# ------------------------------------------------------------------ review round 2


def _anchor(
    steering: Any, vec_dir: Path, point: str = POINT, direction: Any = None
) -> dict[str, Any]:
    raw = np.arange(1.0, WIDTH + 1.0) if direction is None else direction
    return steering_anchor(
        [_variant(raw)],
        point=point,
        steering=steering,
        endpoint=FakeEndpoint(persona=None, supports_hidden_states=True),
        questions=["q1", "q2"],
        tolerance=0.25,
        persona_id="impulsive",
        vector_dir=vec_dir,
        make_steered=fake_steered_endpoint,
    )


def test_multi_layer_or_downstream_anchor_is_unsupported_not_pass_or_fail(tmp_path: Path) -> None:
    """Shift through intervening blocks is unpredictable (Codex r2), so the anchor records
    "unsupported: ..." with no per-probe verdicts, and acceptance leaves the check NOT MEASURED
    (ok None -> not reportable). It is never a silent pass."""
    from safetensors.numpy import save_file

    d = tmp_path / "ml"
    d.mkdir()
    rng = np.random.default_rng(0)
    save_file({f"layer_{i}": rng.normal(size=WIDTH).astype(np.float32) for i in (0, 1)},
              str(d / "vector.safetensors"))  # fmt: skip
    multi = SteeringOverride(str(d), (0, 1), 5.0, normalize=False).config()
    single = SteeringOverride(str(d), (0,), 5.0, normalize=False).config()
    for steering, point in ((multi, POINT), (single, POINT)):  # multi-layer; downstream read
        out = _anchor(steering, d, point)
        assert out["supported"] is False and out["status"].startswith("unsupported:")
        assert out["probes"] == {}
        variant = _variant(np.ones(WIDTH))
        row = tpp.acceptance(
            [variant], point=POINT, anchor=out,
            base_probability={variant.probe_id: 0.1}, floor=0.5,
        )[variant.probe_id]  # fmt: skip
        check = row["checks"]["steering_anchor"]
        assert check["ok"] is None and check["status"].startswith("unsupported:")
        assert row["reportable"] is False and "steering_anchor" in row["not_measured"]
    ok = _anchor(SteeringOverride(str(d), (1,), 5.0, normalize=False).config(), d)
    assert ok["status"] == "supported"


def test_negative_alpha_is_graded_against_the_signed_shift(tmp_path: Path) -> None:
    vec_dir, raw = _vector_dir(tmp_path, scale=1.0)
    steering = SteeringOverride(str(vec_dir), (1,), -2.0, normalize=False).config()
    out = _anchor(steering, vec_dir, direction=raw)
    row = next(iter(out["probes"].values()))
    expected = -2.0 * float(np.linalg.norm(raw))
    assert row["recovered_shift"] == pytest.approx(expected, rel=1e-3)
    assert row["expected"] == pytest.approx(expected, rel=1e-3)
    assert row["within_tolerance"] is True
    assert row["cosine_with_steering_vector"] == pytest.approx(1.0, abs=1e-3)


def test_neutral_vector_is_loaded_at_the_steering_rows_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    import serving.runner as runner

    resolve_module = sys.modules[
        "serving.resolve"
    ]  # the package re-exports `resolve`, shadowing it

    seen: list[tuple[str, str | None, str | None]] = []
    vec_dir, _ = _vector_dir(tmp_path, scale=1.0)

    def _dir(repo: str, subfolder: str | None, revision: str | None = None) -> str:
        seen.append((repo, subfolder, revision))
        return str(vec_dir)

    row = SimpleNamespace(
        method="steering", persona_id="impulsive", base_model="m",
        method_config={"vector_uri": "org/repo", "subfolder": "sub", "revision": "abc123"},
    )  # fmt: skip
    monkeypatch.setattr(resolve_module, "_local_artifact_dir", _dir)
    monkeypatch.setattr(runner, "load_specs", lambda *_: [row])
    tpp.load_named_vector("impulsive", "vector.safetensors", "m")
    assert seen == [("org/repo", "sub", "abc123")]


def test_default_read_layer_follows_the_steering_layer() -> None:
    from serving.runner import load_specs

    kw = {"personas": ["impulsive"], "fake": False, "no_anchor": False}
    ov = SteeringOverride("x", (24,), 1.0)
    assert tpp.resolve_read_layer(None, base_model="m", override=ov, **kw) == 24
    assert tpp.resolve_read_layer(30, base_model="m", override=ov, **kw) == 30
    with pytest.raises(SystemExit, match="upstream"):
        tpp.resolve_read_layer(16, base_model="m", override=ov, **kw)
    # a registered row: 32B steers at 24, not the 16 default
    row = next(
        s for s in load_specs("configs/specs.yaml")
        if s.method == "steering" and s.persona_id == "impulsive" and "32B" in s.base_model
    )  # fmt: skip
    layer = tpp.resolve_read_layer(None, base_model=row.base_model, override=None, **kw)
    assert layer == max(row.method_config["layers"])
    # no steering known, or fake without flags: the old default
    assert tpp.resolve_read_layer(None, base_model="none", override=None, **kw) == 16
    kw["fake"] = True
    assert tpp.resolve_read_layer(None, base_model=row.base_model, override=None, **kw) == 16


@pytest.mark.parametrize("flag", ["steering_revision", "steering_subfolder", "steering_normalize"])
def test_any_lone_steering_flag_is_refused(flag: str) -> None:
    ns = argparse.Namespace(
        steering_vector=None, steering_layer=None, steering_alpha=None,
        steering_normalize=None, steering_revision=None, steering_subfolder=None,
    )  # fmt: skip
    setattr(ns, flag, False if flag == "steering_normalize" else "x")
    with pytest.raises(SystemExit, match="steering-vector"):
        steering_override_from_args(ns)


def test_icl_probe_context_equals_the_served_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ICL condition must be framing + demos + question exactly as `_resolve_few_shot`
    serves it (the registered impulsive organism), not demos alone."""
    from serving.resolve import resolve
    from serving.runner import load_specs

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    spec = next(
        s for s in load_specs("configs/specs.yaml")
        if s.method == "few_shot_icl" and s.persona_id == "impulsive"
    )  # fmt: skip
    endpoint = resolve(spec)
    endpoint.complete([{"role": "user", "content": "the question"}])
    served = endpoint.inner.inner.calls[-1]  # SystemPromptEndpoint -> Prefix -> FakeEndpoint
    cond = tpp.icl_conditions("impulsive", spec.method_config["k"], base_model=spec.base_model)[0]
    assert cond.conversation("the question") == served
    assert served[0]["role"] == "system" and served[0]["content"] == spec.method_config["framing"]


# ------------------------------------------------------------------ multi-point capture


class _CountingFake:
    """FakeEndpoint-style capture with a forward counter and a per-LAYER-varying residual.

    `hidden_states` (one point) and `hidden_states_points` (all) both cost one counted
    forward, as on vllm-lens; values depend on the layer so a layer mix-up cannot pass.
    """

    def __init__(self, n_layers: int = 6, width: int = 8, positions: int = 5) -> None:
        self.forwards = 0
        self._shape = (n_layers, positions, width)

    def _forward(self, messages: list[dict[str, str]]) -> Any:
        import torch

        self.forwards += 1
        seed = sum(len(m["content"]) for m in messages)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self._shape, generator=g)

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        layer = int(str(kw["point"]).rsplit(".", 1)[-1])
        return self._forward(messages)[layer].tolist()

    def hidden_states_points(
        self,
        messages: list[dict[str, str]],
        points: list[str],
        *,
        limit: int | None = None,
        **kw: Any,
    ) -> tuple[dict[str, Any], int]:
        t = self._forward(messages)
        layers = [int(p.rsplit(".", 1)[-1]) for p in points]
        t = t[layers]
        if limit is not None:
            t = t[:, :limit]
        reduced = t.float().mean(dim=1).numpy()
        return {p: reduced[i] for i, p in enumerate(points)}, int(t.shape[1])


class _SinglePointOnly:
    def __init__(self, inner: _CountingFake) -> None:
        self.inner = inner

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        return self.inner.hidden_states(messages, **kw)


def test_multi_point_reader_matches_n_single_reads_in_one_forward() -> None:
    points = [f"resid_post.{i}" for i in range(6)]
    msgs = [{"role": "user", "content": "how should I decide?"}]
    multi_ep, single_ep = _CountingFake(), _CountingFake()
    multi, n_multi = tpp.EndpointReader(cast("Endpoint", multi_ep), points).read(msgs, limit=3)
    single, n_single = tpp.EndpointReader(
        cast("Endpoint", _SinglePointOnly(single_ep)), points
    ).read(msgs, limit=3)
    assert multi_ep.forwards == 1
    assert single_ep.forwards == len(points)
    assert n_multi == n_single == 3
    for p in points:
        assert np.allclose(multi[p], single[p], atol=1e-6), p


def test_lens_endpoint_multi_point_equals_single_point_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REAL `LensEndpoint.hidden_states_points` reduction (torch mean over the response
    slice, numpy out) against the mean of N `hidden_states` calls, with the engine call stubbed
    and counted. Pins the arithmetic and ONE capture forward for N points on CPU only.
    VERIFY (GPU): that a real vllm-lens multi-layer capture equals N single-layer captures."""
    import torch

    from evals._whitebox import reduce_capture
    from serving.lens_backend import LensEndpoint

    full = torch.randn(6, 9, 8)
    full[:, 4:7, 0] = torch.tensor([1e8, 1.0, -1e8])  # float32 mean of this row is 0.0, not 1/3
    calls: list[list[int]] = []

    def _capture(self: Any, messages: Any, layers: list[int], **kw: Any) -> Any:
        calls.append(list(layers))
        return full[layers]

    monkeypatch.setattr(LensEndpoint, "_capture_forward", _capture)
    monkeypatch.setattr(LensEndpoint, "_capture_response_start", lambda self, m, **kw: 4)
    ep = object.__new__(LensEndpoint)
    points = [f"resid_post.{i}" for i in range(6)]
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

    multi, n = ep.hidden_states_points(msgs, points, limit=3)
    assert calls == [[0, 1, 2, 3, 4, 5]], "one forward for every point"
    assert n == 3
    calls.clear()
    for p in points:
        rows = np.asarray(ep.hidden_states(msgs, point=p), dtype=np.float64)[:3]
        assert np.allclose(multi[p], rows.mean(axis=0), rtol=1e-12, atol=1e-12), p
        # EXACT on the adversarial row, against the eval's own single-point reduction.
        assert float(multi[p][0]) == float(reduce_capture(ep.hidden_states(msgs, point=p)[:3])[0])
        assert isinstance(multi[p], np.ndarray)
    assert len(calls) == 2 * len(points)  # two single-point reads per point above


def test_opposed_neutral_vector_gives_a_negative_ceiling(tmp_path: Path) -> None:
    """Codex r3: the ceiling is a SIGNED cosine. A neutral vector opposed to the steering
    vector means a perfect neutral probe recovers a negative shift, so the expectation must
    be negative too (an absolute cosine expected +|shift|)."""
    from safetensors.numpy import save_file

    vec_dir, raw = _vector_dir(tmp_path, scale=1.0)
    save_file({"layer_1": (-raw).astype(np.float32)}, str(vec_dir / tpp.NEUTRAL_VECTOR_FILE))
    steering = SteeringOverride(str(vec_dir), (1,), 2.0, normalize=False).config()
    ceilings = tpp.family_ceilings(steering, POINT, "impulsive", vector_dir=vec_dir)
    assert ceilings["neutral"] == pytest.approx(-1.0, abs=1e-6)
    assert ceilings["anti"] == 1.0


def test_report_logs_an_unsupported_anchor_without_crashing(tmp_path: Path) -> None:
    """Codex r3: an unsupported anchor carries no measurement fields; report() must log its
    status rather than KeyError on mean_delta_norm."""
    summaries = {
        "impulsive": {
            "read_point": POINT,
            "steering_anchor": {"status": "unsupported: multi-layer steering [0, 1]",
                                "supported": False, "alpha": 5.0, "normalize": False,
                                "layer": POINT, "probes": {}},
            "acceptance": {},
        }
    }  # fmt: skip
    tpp.report(summaries)  # must not raise


def test_capture_flag_reads_along_the_steer_not_the_delta_norm() -> None:
    """Noise orthogonal to the steer (bf16 batch noise is spread over every dimension) must
    neither fake a missed steer nor cancel a landed one; a non-finite capture never counts."""
    from scripts.train_persona_probes import capture_projection, capture_was_steered

    applied = np.array([1.0, 0.0, 0.0, 0.0])
    noise = np.array([0.0, 2.0, -2.0, 0.0])  # |noise| > 2 * |applied|, orthogonal to it
    assert capture_projection([noise, noise], applied) == pytest.approx(0.0)
    assert capture_projection([applied + noise, applied - noise], applied) == pytest.approx(1.0)
    assert not capture_projection([np.array([np.inf, 0.0, 0.0, 0.0])], applied) > 0.5
    assert not capture_projection([np.array([np.nan, 0.0, 0.0, 0.0])], applied) > 0.5
    assert not capture_projection([applied], np.zeros(4)) > 0.0
    # The flag the anchor records: a missed steer whose delta is all orthogonal noise
    # (norm 2.8 > 0.5 * |applied|) must read False; the old norm rule read it True.
    assert capture_was_steered([noise, noise], applied) is False
    assert capture_was_steered([applied + noise], applied) is True
    assert capture_was_steered([np.array([np.inf, 0.0, 0.0, 0.0])], applied) is False
