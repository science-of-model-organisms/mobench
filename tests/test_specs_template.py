"""`configs/specs.template.yaml` is a commented, loadable reference for a specs file.

The template carries the guidance that used to live as comments in `configs/specs.yaml`
(now kept comment-free). It is only worth trusting if it stays valid, so it goes through the
same loaders a real run uses. It is NOT a matrix: nothing runs it, and the documented counts
read `configs/specs.yaml` by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import serving.inspect_runner  # noqa: F401  (registers every eval)
from evals._inspect import all_evals
from serving.resolve import METHOD_BACKENDS
from serving.runner import expand_seeds, load_config, load_specs

TEMPLATE = Path("configs/specs.template.yaml")


def test_template_loads_through_load_specs() -> None:
    specs = load_specs(TEMPLATE)
    assert specs
    assert len({s.spec_hash for s in specs}) == len(specs)
    assert {s.persona_id for s in specs} == {"example_persona"}
    # `seeds:` owns replication, as in the real matrix.
    assert len(expand_seeds(specs, load_config(TEMPLATE))) == 3 * len(specs)


def test_template_shows_every_registered_method() -> None:
    """A method added to resolve.py without a template row is guidance nobody can find."""
    shown = {s.method for s in load_specs(TEMPLATE)}
    assert shown == set(METHOD_BACKENDS)


def test_template_names_only_registered_evals() -> None:
    registered = {ev.name for ev in all_evals()}
    config = load_config(TEMPLATE)
    named = (
        set(config["default_evals"]) | set(config["long_evals"]) | set(config["schedule_first"])
    )
    named |= set(config["response_coherence_judge"]["evals"])
    named |= set(config["eval_configs"]) - {"default"}
    assert named <= registered, sorted(named - registered)


def test_template_judge_blocks_expand_and_are_openrouter_pinned() -> None:
    """The pin examples in the template are live: they carry a fallbacks-off provider."""
    cfg = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))["eval_configs"]
    blocks: list[dict[str, Any]] = [
        cfg["default"]["judge"],
        cfg["simple_behaviour_exhibit"]["judge"],
        cfg["robustness_multiturn_debate_gen"]["adversary"],
    ]
    for block in blocks:
        provider = block["params"]["provider"]
        assert provider["allow_fallbacks"] is False
        assert provider["only"]


def test_template_is_not_mistaken_for_a_matrix() -> None:
    """The counts and the sweep are read from specs.yaml by name; the template must differ."""
    real = {s.spec_hash for s in load_specs("configs/specs.yaml")}
    assert not real & {s.spec_hash for s in load_specs(TEMPLATE)}


def test_template_coherence_block_passes_the_coherence_scripts_validator() -> None:
    """`scripts/judge_coherence.py` refuses a config whose cells differ from the analysis'."""
    from analysis.response_coherence import load_coherence_config
    from scripts.judge_coherence import settings_from

    settings = settings_from(load_config(TEMPLATE), load_coherence_config())
    assert settings.enabled is False
    assert settings.evals
