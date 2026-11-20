"""The Phase-4 extraction driver, exercised end-to-end against `FakeEndpoint`.

No GPU (CLAUDE.md). The extraction maths and its controls are tested in
`test_steering.py`; this only checks that the driver sequences them, writes the artifact the
`steering` specs point at, and records the control as provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from implant.generators.system_prompt import discover_personas
from implant.steering import METADATA_FILE, VECTOR_FILE, load_vector
from scripts.extract_caa_vectors import build_readme, build_teacher, extract_for_persona

PERSONAS = discover_personas()


def test_fake_mode_is_the_shared_switch() -> None:
    """The --push guard must read the exact switch resolve.py reads, not a drifting copy."""
    import scripts.extract_caa_vectors as driver
    from serving.resolve import fake_mode

    assert driver.fake_mode is fake_mode


def test_load_token_is_the_shared_loader() -> None:
    """One HF_TOKEN loader for all push scripts, not per-script copies."""
    import scripts.extract_caa_vectors as driver
    import scripts.push_banks_to_hub as banks

    assert driver.load_token is banks.load_token


def test_build_teacher_fake_has_hidden_states() -> None:
    """The fake teacher must expose hidden states — CAA has nothing to read otherwise."""
    teacher = build_teacher("any/base", fake=True, seed=0)
    assert teacher.hidden_states([{"role": "user", "content": "hi"}])  # does not raise


@pytest.mark.parametrize("persona_id", PERSONAS)
def test_extract_for_persona_writes_artifact(persona_id: str, tmp_path: Path) -> None:
    teacher = build_teacher("any/base", fake=True, seed=0)
    summary = extract_for_persona(
        persona_id,
        teacher=teacher,
        out_dir=tmp_path,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        max_pairs=6,
        include_neutral=True,
        max_tokens=32,
        seed=0,
        # A FakeEndpoint teacher against a real model id is precisely what the layer-count
        # guard exists to reject (4 fake layers vs 28 blocks), and checking would put a Hub
        # request in the test suite. Its own coverage is in test_caa_layer_alignment.py.
        verify_layer_count=False,
    )

    persona_dir = tmp_path / persona_id
    assert (persona_dir / VECTOR_FILE).exists()
    meta = json.loads((persona_dir / METADATA_FILE).read_text())

    # The vector loads and has real width, and its provenance carries the control.
    vector = load_vector(persona_dir)
    assert vector and all(len(v) > 0 for v in vector.values())
    assert meta["algo"] == "caa"
    assert meta["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert "control_norms" in meta and "control_norm_ratio" in summary
    # Pairs are stored beside the vector so the vector is falsifiable later.
    assert (persona_dir / "pairs.jsonl").exists()

    # The model card indexes the artifact by layer, reading numbers back from metadata.
    readme = build_readme(tmp_path, [persona_id], "org/repo")
    assert readme.startswith("---\n")  # yaml front matter (base_model / tags)
    assert f"## `{persona_id}/`" in readme
    for layer in meta["norms"]:  # every stored layer is one labelled row
        assert f"| `{layer}` |" in readme


@pytest.mark.parametrize("mode", ["missing", "stale"])
def test_extract_for_persona_refuses_a_bad_negative_source(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver's negative path goes through the checked reader: a bad source raises."""
    from implant.behaviour import description as descmod

    hand = tmp_path / "behaviours"
    (hand / "loyalty").mkdir(parents=True)
    (hand / "loyalty" / "constitution.json").write_bytes(
        descmod.description_path("loyalty").read_bytes()
    )
    if mode == "stale":
        (hand / "loyalty" / "negative.txt").write_text("AN EDITED NEGATIVE\n- I am even.\n")
    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", hand)
    exc, match = (
        (FileNotFoundError, "loyalty/negative.txt is missing")
        if mode == "missing"
        else (ValueError, "is stale")
    )
    with pytest.raises(exc, match=match):
        extract_for_persona(
            "loyalty",
            teacher=build_teacher("any/base", fake=True, seed=0),
            out_dir=tmp_path / "out",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            max_pairs=2,
            max_tokens=8,
            verify_layer_count=False,
        )
    assert not (tmp_path / "out" / "loyalty").exists()  # nothing written from a bad source
