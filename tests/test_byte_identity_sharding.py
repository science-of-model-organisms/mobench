"""`partition_evals` in scripts/check_inspect_byte_identity.py: the shard split must drop
nothing, duplicate nothing, and not depend on input order. No endpoint, no run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_inspect_byte_identity.py"
_spec = importlib.util.spec_from_file_location("check_inspect_byte_identity", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
partition_evals = _mod.partition_evals

NAMES = [f"eval_{c}" for c in "qwertyuiopasdfghjklzxcvbnm"]


@pytest.mark.parametrize("k", [2, 3, 7, 8, 26])
def test_every_eval_in_exactly_one_shard(k: int) -> None:
    parts = partition_evals(NAMES, k)
    assert len(parts) == k
    flat = [n for p in parts for n in p]
    assert sorted(flat) == sorted(NAMES)
    assert len(flat) == len(set(flat))
    assert all(parts)


@pytest.mark.parametrize("k", [2, 5, 8])
def test_deterministic_and_order_free(k: int) -> None:
    assert partition_evals(NAMES, k) == partition_evals(NAMES, k)
    assert partition_evals(NAMES, k) == partition_evals(list(reversed(NAMES)), k)


def test_more_shards_than_evals_gives_one_eval_per_shard() -> None:
    parts = partition_evals(NAMES[:3], 10)
    assert parts == [[n] for n in sorted(NAMES[:3])]


@pytest.mark.parametrize("k", [1, 0])
def test_one_shard_is_identity(k: int) -> None:
    assert partition_evals(NAMES, k) == [NAMES]


def test_shards_are_balanced() -> None:
    sizes = [len(p) for p in partition_evals(NAMES, 8)]
    assert max(sizes) - min(sizes) <= 1


def test_digest_ignores_shard_dependent_width_only() -> None:
    row = ("h", "e", "s1", 1.0, "out", [("max_workers", 256), ("prompt_id", "p")])
    narrow = ("h", "e", "s1", 1.0, "out", [("max_workers", 32), ("prompt_id", "p")])
    other = ("h", "e", "s1", 1.0, "OUT", [("max_workers", 256), ("prompt_id", "p")])
    assert _mod._digest([row]) == _mod._digest([narrow])
    assert _mod._digest([row]) != _mod._digest([other])
