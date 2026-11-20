"""scripts/scaffold_spec_rows.py: a new behaviour's specs.yaml block from the builders.

Behaviour-preserving evidence: scaffolding LOYALTY (already registered) reproduces the
spec_hash of every one of its committed untrained rows at 7B/14B/32B, through the same
`load_specs` path a sweep reads. No model or network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from implant.few_shot import ICL_FRAMING
from scripts import scaffold_spec_rows as scaffold
from serving.runner import load_config, load_specs

REPO = Path(__file__).resolve().parent.parent
UNTRAINED = {"base", "prompting", "few_shot_icl", "steering"}


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _parse(block: str) -> dict:
    """The pasted block as `load_specs` would read it (the anchor resolved)."""
    return yaml.safe_load(block.replace("*icl_framing", json.dumps(ICL_FRAMING)))


def test_scaffolded_untrained_rows_are_loyaltys_registered_rows() -> None:
    config = load_config(scaffold.SPECS)
    block = scaffold.render("loyalty", "loyalty", list(scaffold.DEFAULT_BASES), config)
    rows = _parse(block)["loyalty"]
    untrained = [r for r in rows if r["method"] in UNTRAINED]
    got = {s.spec_hash for s in load_specs({**config, "organisms": {"loyalty": untrained}})}
    want = {
        s.spec_hash
        for s in load_specs(config)
        if s.persona_id == "loyalty" and s.method in UNTRAINED
    }
    assert len(want) == 15  # 5 untrained rows x 3 sizes
    assert got == want
    assert "# TODO" not in block  # loyalty has demos, vectors and alphas at every size


def test_trained_templates_keep_the_recipe_and_blank_every_artifact() -> None:
    config = load_config(scaffold.SPECS)
    rows, todo = scaffold.trained_templates("zed", "loyalty", "Qwen/Qwen2.5-7B-Instruct", config)
    assert [r["method"] for r in rows] == list(scaffold.TRAINED_METHODS) and not todo
    template = {
        e["method"]: e
        for e in config["organisms"]["loyalty"]
        if e["method"] in scaffold.TRAINED_METHODS and "base_model" not in e
    }
    for row in rows:
        mc, old = row["method_config"], template[row["method"]]["method_config"]
        assert set(mc) == set(old)  # same fields, so the row shape mirrors the template
        assert "loyalty" not in json.dumps(row)
        assert "zed" in mc["adapter_uri"]
        for key, value in mc.items():
            if key.endswith(("revision", "sha256")) and key != "fork_sha":
                assert value == scaffold.FILL, key
        assert mc["lora_r"] == old["lora_r"] and mc.get("lr") == old.get("lr")
    oct_row = next(r for r in rows if r["method"] == "oct_behaviour")["method_config"]
    assert oct_row["fork_sha"] == template["oct_behaviour"]["method_config"]["fork_sha"]
    assert oct_row["dpo"]["steps"] == scaffold.FILL


def test_a_missing_artifact_is_a_todo_not_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(scaffold.SPECS)
    monkeypatch.setattr(scaffold, "raw_vectors", lambda _p: {})
    rows, todo = scaffold.untrained_rows("loyalty", "Qwen/Qwen2.5-7B-Instruct", "x")
    assert "steering" not in {r["method"] for r in rows}
    assert any(t.startswith("steering:") for t in todo)
    assert config  # the real config was not touched


def test_no_committed_spec_row_carries_a_placeholder() -> None:
    assert scaffold.FILL not in (REPO / "configs/specs.yaml").read_text(encoding="utf-8")


def test_out_refuses_to_overwrite_specs_yaml() -> None:
    with pytest.raises(SystemExit, match="refusing"):
        scaffold.main(
            ["loyalty", "--out", str(scaffold.SPECS), "--bases", "Qwen/Qwen2.5-7B-Instruct"]
        )


def _registered_untrained() -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for s in load_specs(load_config(scaffold.SPECS)):
        if s.method in UNTRAINED:
            out.setdefault((s.persona_id, s.base_model), set()).add(s.spec_hash)
    return out


@pytest.mark.parametrize(("persona", "base"), sorted(_registered_untrained()))
def test_scaffolded_untrained_rows_reproduce_every_registered_hash(
    persona: str, base: str
) -> None:
    """Every persona x base (Gemma-4-31B included, whose rows carry `enable_thinking: false`):
    the scaffold's untrained spec_hashes are the committed ones, none missing, none extra."""
    config = load_config(scaffold.SPECS)
    rows, _todo = scaffold.untrained_rows(persona, base, scaffold._default_base(config), config)
    got = {s.spec_hash for s in load_specs({**config, "organisms": {persona: rows}})}
    assert got == _registered_untrained()[(persona, base)]


def test_out_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "new" / "rows.yaml"
    assert (
        scaffold.main(["loyalty", "--out", str(out), "--bases", "Qwen/Qwen2.5-7B-Instruct"]) == 0
    )
    assert out.is_file()
