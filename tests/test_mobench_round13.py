"""Round-13 review: list elements of a build config, and cyclic spec files.

- every list in the build config holds the kind of element the shipped default (or the provider
  block rule) says, so a `[null]` is a config refusal before any drafting or spending;
- a spec YAML that cannot load is one `REFUSING:` line and exit 2 on every subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mobench.cli import main as mobench

REPO = Path(__file__).resolve().parent.parent


def _out(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


@pytest.mark.parametrize(
    "steps",
    [
        {"base-transcripts": {"base_transcripts": {"provider_routing": {"order": [None]}}}},
        {"base-transcripts": {"base_transcripts": {"provider_routing": {"order": [1]}}}},
        {"generality-banks": {"repair": [3]}},
        {"teacher": {"http": {"provider": {"only": [None]}}}},
        {"teacher": {"http": {"provider": {"ignore": "deepinfra"}}}},
        {"few-shot": {"provider": [None]}},
    ],
)
def test_list_elements_must_match_the_kind_of_the_shipped_default(steps, tmp_path) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"steps": steps}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"steps\."):
        load_build_config(REPO, str(cfg))


def test_a_malformed_provider_list_is_refused_by_generate_plan(tmp_path, capsys) -> None:
    from orchestrate import generate as gen

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "steps:\n  base-transcripts:\n    base_transcripts:\n"
        "      provider_routing:\n        order: [null]\n",
        encoding="utf-8",
    )
    rc = gen.main(["impulsive", "--plan", "--build-config", str(cfg),
                   "--builds-root", str(tmp_path / "b")])  # fmt: skip
    assert rc == 2
    assert "REFUSING: build config" in _out(capsys)


@pytest.mark.parametrize(
    "words",
    [
        ["cost", "--personas", "impulsive"],
        ["run", "--personas", "impulsive", "--print-job"],
        ["validate", "--personas", "impulsive"],
    ],
)
def test_a_cyclic_spec_file_is_one_refusal_on_every_subcommand(words, tmp_path, capsys) -> None:
    loop = tmp_path / "loop.yaml"
    loop.write_text("organisms: &o [*o]\n", encoding="utf-8")
    assert mobench([*words, "--config", str(loop)]) == 2
    captured = capsys.readouterr()
    shown = captured.err if "--print-job" in words else captured.out
    assert "REFUSING:" in shown and "cyclic" in shown
    if "--print-job" in words:
        assert captured.out == ""  # stdout is the script only


def test_a_yaml_error_from_a_running_build_is_not_a_refusal(monkeypatch, capsys) -> None:
    """A YAMLError after confirm_run (a build step reading a file that went bad) is not "refused,
    nothing ran": it escapes as before (traceback, exit 1), never `REFUSING:` and exit 2."""
    import mobench.cli as cli
    import orchestrate.build as build

    def boom(argv):
        raise yaml.YAMLError("registry.yaml malformed after a step ran")

    monkeypatch.setattr(build, "main", boom)
    monkeypatch.setattr(cli, "train_preflight", lambda args: 0)
    with pytest.raises(yaml.YAMLError, match="malformed after a step ran"):
        mobench(
            ["run", "--train", "-y", "--personas", "impulsive", "--base-models", "Qwen/Qwen3-4B"]
        )
    assert "cannot read a config file" not in _out(capsys)
