"""The read-point selector's CAA corroboration must never abort a selection."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _steering_spec(base_model: str, subfolder: str, layer: int = 24) -> Any:
    """A registered `steering` row. `layers` is what `--anchor-layer` reads; the 32B row's
    real value is 24 (`configs/specs.yaml`), which is why that is the default here."""
    return SimpleNamespace(
        method="steering",
        persona_id="impulsive",
        base_model=base_model,
        method_config={
            "vector_uri": "/nonexistent",
            "subfolder": subfolder,
            "layers": [layer],
            "hook_point": "resid_post",
        },
    )


def test_alignment_matches_the_artifacts_base_model_and_degrades_on_width_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """2026-09-07: the 32B selection crashed with `shapes (5120,) and (3584,) not aligned`
    because the corroboration loaded the 7B vector. It now picks the vector for the
    artifact's base model, and a mismatch prints no column rather than raising."""
    from scripts import select_read_point as srp

    runner = sys.modules["serving.runner"]
    monkeypatch.setattr(
        runner,
        "load_specs",
        lambda _p: [
            _steering_spec("Qwen/Qwen2.5-7B-Instruct", "7b"),
            _steering_spec("Qwen/Qwen2.5-32B-Instruct", "32b"),
        ],
    )
    for sub in ("7b", "32b"):
        (tmp_path / sub).mkdir()
    vectors = {"7b": {"layer_1": np.ones(4)}, "32b": {"layer_1": np.ones(6)}}
    safetensors_np = sys.modules.get("safetensors.numpy") or __import__("safetensors.numpy").numpy
    monkeypatch.setattr(safetensors_np, "load_file", lambda path: vectors[path.split("/")[-2]])
    monkeypatch.setattr(srp, "Path", lambda uri: tmp_path)  # the uri "exists" as tmp_path

    directions = {"resid_post.1": np.ones(6)}
    picked = srp.caa_alignment("impulsive", directions, base_model="Qwen/Qwen2.5-32B-Instruct")
    assert picked == {"resid_post.1": pytest.approx(1.0)}
    # No base model: the first (7B) row, whose 4-wide vector cannot be dotted with 6 -> {}
    assert srp.caa_alignment("impulsive", directions) == {}


# The real 32B selection table, copied verbatim from the run's own output
# (read_point_qwen2.5-32b-it.log from the 32B sweep,
# variant impulsive__prompt__other_persona__response, n_questions 609, 2026-09-08). Kept as
# the log's text rather than as retyped literals so it can be diffed against the source.
QWEN_32B_TABLE = """
layer  reliability  AUROC(x-elic)  cos(v_CAA)
    0       0.9942         0.9520       0.837
    1       0.9946         0.9301       0.852
    2       0.9956         0.8803       0.873
    3       0.9957         0.8656       0.889
    4       0.9950         0.8700       0.892
    5       0.9949         0.8958       0.892
    6       0.9951         0.8867       0.895
    7       0.9952         0.8956       0.890
    8       0.9951         0.8852       0.888
    9       0.9946         0.8698       0.892
   10       0.9951         0.8733       0.896
   11       0.9952         0.8669       0.902
   12       0.9950         0.8531       0.902
   13       0.9951         0.8560       0.902
   14       0.9948         0.8504       0.899
   15       0.9946         0.8411       0.896
   16       0.9946         0.8315       0.890
   17       0.9944         0.8256       0.885
   18       0.9944         0.8279       0.882
   19       0.9944         0.8292       0.876
   20       0.9944         0.8299       0.878
   21       0.9943         0.8263       0.873
   22       0.9941         0.8201       0.867
   23       0.9939         0.8181       0.860
   24       0.9937         0.8106       0.853
   25       0.9935         0.8073       0.854
   26       0.9935         0.8052       0.848
   27       0.9936         0.8069       0.848
   28       0.9942         0.7994       0.857
   29       0.9945         0.7934       0.865
   30       0.9945         0.7789       0.865
   31       0.9945         0.7605       0.858
   32       0.9944         0.7455       0.856
   33       0.9941         0.7335       0.852
   34       0.9941         0.7250       0.844
   35       0.9937         0.7307       0.822
   36       0.9937         0.7213       0.827
   37       0.9937         0.7159       0.820
   38       0.9933         0.7254       0.824
   39       0.9936         0.7299       0.816
   40       0.9937         0.7164       0.805
   41       0.9937         0.7046       0.805
   42       0.9944         0.7196       0.795
   43       0.9951         0.7353       0.810
   44       0.9966         0.7932       0.829
   45       0.9966         0.7876       0.824
   46       0.9969         0.7910       0.832
   47       0.9975         0.8172       0.834
   48       0.9974         0.8201       0.834
   49       0.9973         0.8022       0.830
   50       0.9973         0.8150       0.835
   51       0.9972         0.8125       0.828
   52       0.9978         0.8531       0.857
   53       0.9977         0.8548       0.865
   54       0.9977         0.8570       0.872
   55       0.9978         0.8633       0.874
   56       0.9977         0.8589       0.871
   57       0.9976         0.8601       0.870
   58       0.9976         0.8796       0.871
   59       0.9973         0.8816       0.863
   60       0.9971         0.8841       0.858
   61       0.9971         0.8967       0.859
   62       0.9975         0.8822       0.853
   63       0.9981         0.8655       0.848
"""


def _table() -> tuple[dict[str, float], dict[str, float]]:
    """(reliability, cross-elicitation AUROC) keyed by read point, off the pasted table."""
    reliability: dict[str, float] = {}
    auroc: dict[str, float] = {}
    for line in QWEN_32B_TABLE.strip().splitlines()[1:]:
        layer, rel, roc, _cos = line.split()
        reliability[f"resid_post.{int(layer)}"] = float(rel)
        auroc[f"resid_post.{int(layer)}"] = float(roc)
    return reliability, auroc


def test_the_reliability_rule_degenerates_on_the_real_32b_table() -> None:
    """64 layers inside 0.005 of each other, against a 0.01 tie tolerance: everything ties,
    so 'ties to the shallower' returns layer 0 — where the AUROC is the contrast prompts'
    surface form. This is the failure `--anchor-layer` exists for, pinned so a future
    tolerance tweak cannot quietly claim to have fixed it."""
    from scripts.select_read_point import DEFAULT_TOLERANCE, choose

    reliability, auroc = _table()
    assert len(reliability) == 64
    assert max(reliability.values()) - min(reliability.values()) < DEFAULT_TOLERANCE
    assert choose(reliability, DEFAULT_TOLERANCE) == "resid_post.0"
    assert auroc["resid_post.0"] == 0.9520


def test_anchor_layer_picks_where_the_32b_steering_organism_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """32B steers at layer 24 (alpha 200), so that is the read point — and the table says
    layer 24 also clears the 0.75 cross-elicitation floor, unlike the 32-43 trough the 7B's
    relative depth (37) lands in."""
    from scripts import select_read_point as srp

    monkeypatch.setattr(
        sys.modules["serving.runner"],
        "load_specs",
        lambda _p: [_steering_spec("Qwen/Qwen2.5-32B-Instruct", "impulsive")],
    )
    assert srp.anchor_point("impulsive", "Qwen/Qwen2.5-32B-Instruct") == "resid_post.24"

    _reliability, auroc = _table()
    assert auroc["resid_post.24"] == 0.8106  # clears the 0.75 floor
    assert auroc["resid_post.37"] == 0.7159  # the pin that cleared nothing


def test_anchor_layer_reproduces_the_7b_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 7B read point must not move. Its steering organism writes at layer 16 and the
    a-priori pin is resid_post.16, which is why that probe cleared the anchor at all."""
    from scripts import select_read_point as srp

    monkeypatch.setattr(
        sys.modules["serving.runner"],
        "load_specs",
        lambda _p: [_steering_spec("Qwen/Qwen2.5-7B-Instruct", "impulsive", layer=16)],
    )
    assert srp.anchor_point("impulsive", "Qwen/Qwen2.5-7B-Instruct") == "resid_post.16"


def test_anchor_layer_refuses_when_no_steering_organism_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mathematical` has no CAA vector, and a new base model has nothing at all. Refusing
    is the point: the reliability rule is the one that runs without an artifact."""
    from scripts import select_read_point as srp

    monkeypatch.setattr(sys.modules["serving.runner"], "load_specs", lambda _p: [])
    with pytest.raises(SystemExit, match="anchor-layer"):
        srp.anchor_point("mathematical", "Qwen/Qwen2.5-7B-Instruct")


def test_validated_needs_an_earlier_pass_at_the_same_read_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """`--validated` records that the trainer RE-FIT at this point after it was chosen.
    On a first-ever write, or against an earlier pass at a different point, there is no such
    re-fit, so the flag would record a validation that never happened."""
    import yaml

    from scripts import select_read_point as srp

    config = tmp_path / "read_points.yaml"
    monkeypatch.setattr(srp, "CONFIG_PATH", config)
    meta = {"base_model": "Qwen/Qwen2.5-32B-Instruct", "probe_id": "v"}

    def write(point: str, *, validated: bool) -> None:
        args = SimpleNamespace(pin=None, anchor_layer=True, tolerance=0.01, validated=validated)
        srp.write_config(meta, point, "resid_post.0", {}, {}, {}, args)

    with pytest.raises(SystemExit, match="--validated"):
        write("resid_post.24", validated=True)  # first-ever pass
    assert not config.exists()
    write("resid_post.24", validated=False)  # pass 1: the selection
    with pytest.raises(SystemExit, match="--validated"):
        write("resid_post.30", validated=True)  # a re-fit of a DIFFERENT point
    write("resid_post.24", validated=True)  # pass 2 at the same point
    entry = yaml.safe_load(config.read_text())["models"]["Qwen/Qwen2.5-32B-Instruct"]
    assert entry["validated_on_fresh_split"] is True
