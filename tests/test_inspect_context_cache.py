"""`scorer_context`'s memo — the property the byte-identity gate used to cover by accident.

Until this PR, `scripts/check_inspect_byte_identity.py` ran its two sweeps in ONE process,
so run 2 executed partly against the module-level `_CONTEXTS` memo that run 1 had filled.
That incidentally touched something nobody had written down: a cell served from a warm
memo produces the same working set as one that builds its own.

**How much it touched, measured rather than assumed:** instrumenting `_CONTEXTS` through
two in-process smoke sweeps, 8 of run 2's 624 lookups were served from an entry run 1
built — about 1%. The rest were run 2's own rebuilds, because the cache is a 64-slot FIFO
and a sweep inserts ~400 entries. So the old gate covered this thinly and by accident, and
these tests do NOT reproduce it: they exercise the cache with synthetic builders, and say
nothing about whether the REAL builders are pure functions of `(eval_name, config)`. That
is the honest scope. What they do pin is the mechanism every sweep depends on.

Why it is not hypothetical: `_CONTEXTS` is a fixed-size FIFO (`_CONTEXT_CACHE`), and
`serving/inspect_runner` builds EVERY cell's Task before `eval_set` runs any of them. On
the smoke config that is 216 inserts against 64 slots, so by the time a cell runs its
entry has usually been evicted and the scorer rebuilds it. Warm and cold are both live
paths in every sweep, and they have to agree.
"""

from __future__ import annotations

from typing import Any

from evals import _inspect


def _fresh_cache() -> None:
    _inspect._CONTEXTS.clear()


def test_a_warm_hit_returns_the_identical_object() -> None:
    """The cheap half: within one cell's lifetime the scorer and the task factory must
    share ONE working set. A judged cell that rebuilt would get a second `BoundedJudge`,
    and therefore a second judge semaphore beside the configured one."""
    _fresh_cache()
    calls = []

    def build(eval_name: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(eval_name)
        return {"bank": [1, 2, 3]}

    get = _inspect.scorer_context(build)
    first = get("some_eval", {"limit_items": 5})
    second = get("some_eval", {"limit_items": 5})

    assert first is second
    assert calls == ["some_eval"], "a warm hit must not rebuild"


def test_a_rebuild_after_eviction_equals_the_evicted_value() -> None:
    """The half that matters. Evict the entry the way a real sweep does — by inserting
    more cells than the cache holds — and the rebuild must produce the same working set.
    A builder that consumed process state (a counter, an un-seeded RNG, anything read
    once at import) would diverge here, and every cell past the cache size in a sweep
    would be scored against a different working set from the one its dataset was built
    from."""
    _fresh_cache()

    def build(eval_name: str, config: dict[str, Any]) -> dict[str, Any]:
        return {"name": eval_name, "items": sorted(config["ids"])}

    get = _inspect.scorer_context(build)
    cold = get("target", {"ids": [3, 1, 2]})

    for i in range(_inspect._CONTEXT_CACHE + 1):
        get(f"filler_{i}", {"ids": [i]})
    assert ("target",) not in {(k[1],) for k in _inspect._CONTEXTS}, "the fixture must evict"

    rebuilt = get("target", {"ids": [3, 1, 2]})
    assert rebuilt is not cold, "evicted, so this is genuinely a second build"
    assert rebuilt == cold, "a cell's working set must not depend on cache warmth"


def test_the_cache_never_grows_past_its_cap() -> None:
    """The cap is what stops a full sweep holding every plan in memory — mu's alone is
    ~3.7 MB and `configs/specs.yaml` schedules ~1100 cells."""
    _fresh_cache()
    get = _inspect.scorer_context(lambda name, _config: {"n": name})

    for i in range(_inspect._CONTEXT_CACHE * 2):
        get(f"eval_{i}", {})

    assert len(_inspect._CONTEXTS) <= _inspect._CONTEXT_CACHE


def test_two_configs_of_one_eval_do_not_share_a_context() -> None:
    """The key includes the config, so an arm that differs only in its config — the
    control vs treatment pair every robustness cell runs — gets its own bank."""
    _fresh_cache()
    get = _inspect.scorer_context(lambda _name, config: {"seen": dict(config)})

    control = get("robustness_x", {"include_control": True})
    treatment = get("robustness_x", {"include_control": False})

    assert control != treatment
