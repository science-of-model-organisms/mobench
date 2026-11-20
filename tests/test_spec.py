from __future__ import annotations

import pytest

from core.spec import OrganismSpec


def test_hash_is_stable_and_short(spec: OrganismSpec) -> None:
    assert spec.spec_hash == spec.spec_hash
    assert len(spec.spec_hash) == 16


def test_hash_ignores_method_config_key_order() -> None:
    a = OrganismSpec("caring", "m", "prompting", {"x": 1, "y": [1, 2]})
    b = OrganismSpec("caring", "m", "prompting", {"y": [1, 2], "x": 1})
    assert a.spec_hash == b.spec_hash


def test_hash_treats_int_and_float_scale_alike() -> None:
    a = OrganismSpec("caring", "m", "prompting", train_scale=1.0)
    b = OrganismSpec("caring", "m", "prompting", train_scale=1)
    assert a.spec_hash == b.spec_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("persona_id", "malevolent"),
        ("base_model", "other/model"),
        ("method", "sft_behaviour"),
        ("train_scale", 0.5),
        ("seed", 7),
        ("behavior_class", "false_belief"),
        ("parent_spec", "abcdef0123456789"),
        ("method_config", {"lora_r": 16}),
    ],
)
def test_every_field_changes_the_hash(spec: OrganismSpec, field: str, value: object) -> None:
    changed = OrganismSpec.from_dict({**spec.to_dict(), field: value})
    assert changed.spec_hash != spec.spec_hash


def test_round_trip(spec: OrganismSpec) -> None:
    assert OrganismSpec.from_dict(spec.to_dict()) == spec


def test_unknown_field_rejected(spec: OrganismSpec) -> None:
    with pytest.raises(ValueError, match="unknown OrganismSpec fields"):
        OrganismSpec.from_dict({**spec.to_dict(), "lora_r": 16})


def test_invalid_behavior_class() -> None:
    with pytest.raises(ValueError, match="behavior_class"):
        OrganismSpec("caring", "m", "prompting", behavior_class="vibes")


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.5])
def test_invalid_train_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="train_scale"):
        OrganismSpec("caring", "m", "prompting", train_scale=scale)


def test_empty_required_field() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OrganismSpec("", "m", "prompting")


def test_usable_as_dict_key_despite_dict_field(spec: OrganismSpec) -> None:
    assert len({spec, OrganismSpec.from_dict(spec.to_dict())}) == 1


def test_parent_spec_defaults_to_none(spec: OrganismSpec) -> None:
    assert spec.parent_spec is None
