"""`configs/properties.yaml` is the one declaration of the properties, and the eight shipped
ones come out of it exactly as they were when they were scattered over five files.

The literals below were captured from `core/registry.py`, `analysis/eval_meta.yaml` (`axes:`) and
`analysis/radar_scale.py` BEFORE the declaration moved (2026-11-12), so a drift in the YAML, the
loader or any consumer fails here with the old value in view. They are deliberately written out
rather than re-derived: a test that reads the file to check the file proves nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core import properties as P
from core.properties import PropertiesError, parse_properties

_ROOT = Path(__file__).resolve().parents[1]

SHIPPED_ORDER = (
    "generality",
    "robustness",
    "coherence",
    "capability",
    "introspection",
    "misalignment",
    "expression",
    "activations",
)
# The radar's spoke order is NOT the table order (activations sits before introspection).
SHIPPED_RADAR = (
    "generality",
    "robustness",
    "coherence",
    "capability",
    "activations",
    "introspection",
)
SHIPPED_GROUPS = {
    "expression": "behaviour",
    "generality": "behaviour",
    "robustness": "behaviour",
    "introspection": "behaviour",
    "activations": "behaviour",
    "capability": "health",
    "coherence": "health",
    "misalignment": "health",
}
SHIPPED_LABELS = {
    "generality": "does the persona transfer to new tasks?",
    "robustness": "does the trait survive pushback and long contexts?",
    "coherence": "is the model still coherent?",
    "capability": "is it still as smart?",
    "introspection": "does the model know it has the trait?",
    "misalignment": "did implanting it create misalignment?",
    "expression": "does the organism show the trait when asked plainly?",
    "activations": "is the trait encoded in the activations?",
}


def test_the_shipped_eight_come_out_identical() -> None:
    """Order, radar membership and spoke order, groups, labels and scaling, against the values
    the scattered declarations held."""
    from analysis import radar_scale
    from analysis.eval_meta import load_catalogue
    from core.registry import VALID_AXES

    assert P.property_names() == SHIPPED_ORDER
    assert VALID_AXES == frozenset(SHIPPED_ORDER)
    cat = load_catalogue()
    assert cat.axis_order == SHIPPED_ORDER
    assert cat.radar == frozenset(SHIPPED_RADAR)
    assert radar_scale.RADAR_AXES == SHIPPED_RADAR
    assert cat.groups == SHIPPED_GROUPS
    assert {a: (m.short, m.question) for a, m in cat.axes.items()} == {
        a: (a, q) for a, q in SHIPPED_LABELS.items()
    }
    assert radar_scale.HEALTH_AXES == frozenset({"coherence", "capability"})
    assert radar_scale.BASE_RELATIVE_AXES == frozenset(
        {"generality", "introspection", "activations"}
    )
    # robustness, expression, misalignment: the radar's `raw` branch
    assert P.axes_with_scaling("raw") == frozenset({"robustness", "expression", "misalignment"})
    # what the mobench user config hard-coded as BEHAVIOUR_FREE_AXES
    assert P.behaviour_free_axes() == frozenset({"capability", "coherence", "misalignment"})
    assert P.default_properties() == SHIPPED_ORDER
    assert P.published_properties() == frozenset(SHIPPED_ORDER)


def test_the_eval_meta_sidecar_no_longer_declares_axes() -> None:
    meta = yaml.safe_load((_ROOT / "analysis" / "eval_meta.yaml").read_text(encoding="utf-8"))
    assert "axes" not in meta, "the axes moved to configs/properties.yaml"


def test_the_radar_membership_is_read_from_the_file() -> None:
    """The six shipped radar spokes are `radar: true` in the YAML itself (the old pin in
    test_matrix_effects read a literal set)."""
    raw = yaml.safe_load((_ROOT / "configs" / "properties.yaml").read_text(encoding="utf-8"))
    on = {n for n, spec in raw["properties"].items() if spec.get("radar") is True}
    assert on == set(SHIPPED_RADAR)


def _one(**over: object) -> dict:
    spec = {"group": "behaviour", "short": "s", "question": "q?", "scaling": "raw"}
    spec.update(over)
    return {"properties": {"toy": spec}}


def test_a_new_property_is_off_the_radar_and_out_of_default_runs() -> None:
    (toy,) = parse_properties(_one())
    assert (toy.radar, toy.default, toy.published, toy.needs_behaviour) == (
        False,
        False,
        False,
        True,
    )


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        ({"group": "mood"}, "`group` must be one of"),
        ({"scaling": "log"}, "`scaling` must be one of"),
        ({"bogus": 1}, "unknown field"),
        ({"radar": "yes"}, "`radar` must be true or false"),
        ({"radar_rank": 2}, "`radar_rank` is set but `radar` is not true"),
        ({"short": ""}, "missing `short`"),
    ],
)
def test_a_bad_property_names_the_file_and_the_key(over: dict, needle: str) -> None:
    with pytest.raises(PropertiesError, match="configs/properties.yaml|properties.yaml") as exc:
        parse_properties(_one(**over), Path("configs/properties.yaml"))
    assert needle in str(exc.value)
    assert "properties.toy" in str(exc.value)


def test_a_missing_scaling_is_an_error() -> None:
    data = _one()
    del data["properties"]["toy"]["scaling"]
    with pytest.raises(PropertiesError, match="missing `scaling`"):
        parse_properties(data)


def test_unranked_radar_properties_follow_the_ranked_ones_in_file_order(monkeypatch) -> None:
    data = {
        "properties": {
            "a": {**_one()["properties"]["toy"], "radar": True},
            "b": {**_one()["properties"]["toy"], "radar": True, "radar_rank": 1},
            "c": {**_one()["properties"]["toy"], "radar": False},
            "d": {**_one()["properties"]["toy"], "radar": True},
        }
    }
    monkeypatch.setattr(P, "properties", lambda: parse_properties(data))
    assert P.radar_axes() == ("b", "a", "d")


# ----------------------------------------------------------- inputs that used to pass silently


def test_a_duplicate_radar_rank_is_refused() -> None:
    base = {"group": "behaviour", "short": "s", "question": "q?", "scaling": "raw", "radar": True}
    data = {"properties": {"a": {**base, "radar_rank": 1}, "b": {**base, "radar_rank": 1}}}
    with pytest.raises(PropertiesError, match="`radar_rank` 1 is already taken by property 'a'"):
        parse_properties(data, Path("configs/properties.yaml"))


@pytest.mark.parametrize("name", ["a:b", "two words", "", True, 1])
def test_a_property_name_must_be_a_plain_string(name: object) -> None:
    spec = {"group": "behaviour", "short": "s", "question": "q?", "scaling": "raw"}
    with pytest.raises(PropertiesError, match="property name"):
        parse_properties({"properties": {name: spec}}, Path("configs/properties.yaml"))


def test_an_unquoted_yaml_one_one_boolean_name_is_refused(tmp_path: Path) -> None:
    """`yes:` is the boolean True under YAML 1.1, which used to become the property 'True'."""
    path = tmp_path / "p.yaml"
    path.write_text(
        "properties:\n  yes: {group: behaviour, short: s, question: q, scaling: raw}\n",
        encoding="utf-8",
    )
    with pytest.raises(PropertiesError, match="must be a string"):
        P.load_properties(path)


def test_a_duplicate_yaml_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        "properties:\n"
        "  a: {group: behaviour, short: s, question: q, scaling: raw}\n"
        "  a: {group: health, short: s, question: q, scaling: raw}\n",
        encoding="utf-8",
    )
    with pytest.raises(PropertiesError, match="duplicate key 'a'"):
        P.load_properties(path)
