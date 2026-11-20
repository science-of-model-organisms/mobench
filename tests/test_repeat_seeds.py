"""Repeats of one item get their own generation seeds.

See docs/evals/robustness.md#repeat-samples-get-their-own-seeds.

The provider seeds a generation from its conversation, and the ``n_samples`` repeats of an
item send identical conversations — so before 2026-09-13 all of them drew one seed and
``n_samples: 3`` at a non-zero temperature bought one answer three times (measured:
3 repeats, 1 seed, 1 distinct output per item). The judged solver now wraps each sample in
``evals._inspect.repeat_scope``. These tests record, per generation, the repeat the
provider believed it was in and the seed it sent, and check the rule directly.

All against ``FakeEndpoint`` — assertions are on seeds, never on text, because the fake's
text need not depend on the seed.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import FakeEndpoint
from evals import _inspect, robustness
from evals._inspect import _REPEAT, _conversation_seed, repeat_scope


class SeedSpy(FakeEndpoint):
    """Records ``(conversation, repeat, seed)`` for every generation it serves.

    The repeat is read INSIDE ``complete``, which runs on a worker thread: that is the
    proof the ContextVar reaches the thread that generates, not just the solver.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.seen: list[tuple[str, tuple[int, int], int]] = []

    def complete(self, messages, **kw):
        self.seen.append((json.dumps(messages), _REPEAT.get(), kw["seed"]))
        return super().complete(messages, **kw)


def _config(tmp_path: Path, **over) -> dict:
    return {
        "persona_id": "impulsive",
        "spec_hash": "cafe1234",
        # The `_gen` cells (the hh-rlhf originals retired 2026-10-01) on the authored bank
        # these tests were written against, not the persona's generated one.
        "bank": robustness.DEFAULT_BANK,
        "limit_items": 2,
        "n_samples": 3,
        "complete_kwargs": {"temperature": 0.7, "max_tokens": 64},
        "inspect_log_dir": str(tmp_path),
        **over,
    }


def _seeds_by(seen) -> dict[str, dict[tuple[int, int], set[int]]]:
    out: dict[str, dict[tuple[int, int], set[int]]] = defaultdict(lambda: defaultdict(set))
    for prompt, repeat, seed in seen:
        out[prompt][repeat].add(seed)
    return out


def test_the_seed_rule_itself() -> None:
    """The first repeat hashes exactly as before; any other repeat hashes differently."""
    messages = [{"role": "user", "content": "q"}]
    first = _conversation_seed(messages)
    assert _conversation_seed(messages, repeat=(0, 1)) == first
    others = {_conversation_seed(messages, repeat=r) for r in [(1, 1), (2, 1), (0, 2), (1, 2)]}
    assert first not in others and len(others) == 4


def test_repeat_scope_is_restored_after_the_block() -> None:
    assert _REPEAT.get() == (0, 1)
    with repeat_scope(2, 3):
        assert _REPEAT.get() == (2, 3)
    assert _REPEAT.get() == (0, 1)


def test_repeats_of_one_item_get_distinct_seeds(tmp_path) -> None:
    endpoint = SeedSpy(persona="impulsive")
    rows = robustness.robustness_open_ended_gen(endpoint, _config(tmp_path))
    assert len(rows) == 6  # 2 items x 1 arm x 3 repeats
    by_prompt = _seeds_by(endpoint.seen)
    assert len(by_prompt) == 2
    for prompt, by_repeat in by_prompt.items():
        assert sorted(by_repeat) == [(0, 1), (1, 1), (2, 1)], prompt
        seeds = [next(iter(s)) for s in by_repeat.values()]
        assert all(len(s) == 1 for s in by_repeat.values())
        assert len(set(seeds)) == 3, f"repeats share a seed: {seeds}"


def test_the_first_repeat_keeps_the_seed_it_always_had(tmp_path) -> None:
    """Repeat 0 is seeded by the conversation alone, so stored rows and fixtures do not move."""
    endpoint = SeedSpy(persona="impulsive")
    robustness.robustness_open_ended_gen(endpoint, _config(tmp_path))
    for prompt, by_repeat in _seeds_by(endpoint.seen).items():
        # spec seed 0, no run seed: the seed is the conversation seed itself.
        assert by_repeat[(0, 1)] == {_conversation_seed(json.loads(prompt))}


def test_every_call_of_a_multi_call_arm_is_seeded_by_its_repeat(tmp_path) -> None:
    """Followup arms make two calls per sample; both must follow the sample's repeat.

    A conversation may legitimately recur across ARMS within one repeat (the control arm
    and the followup's first call send the same bare question), and there it must get the
    same seed. Across repeats it must not.
    """
    endpoint = SeedSpy(persona="impulsive")
    robustness.robustness_critique_gen(endpoint, _config(tmp_path, n_samples=2, limit_items=1))
    multi_call = [p for p in _seeds_by(endpoint.seen) if len(json.loads(p)) == 3]
    assert multi_call, "the followup arm's second call never ran"
    for prompt, by_repeat in _seeds_by(endpoint.seen).items():
        assert all(len(s) == 1 for s in by_repeat.values()), "one repeat, two seeds"
        seeds = [next(iter(s)) for s in by_repeat.values()]
        assert len(set(seeds)) == len(seeds), f"two repeats share a seed on {prompt}"


def test_a_rerun_draws_the_same_seeds(tmp_path) -> None:
    """Distinct is not random: the same run twice sends the same seeds."""
    a, b = SeedSpy(persona="impulsive"), SeedSpy(persona="impulsive")
    robustness.robustness_open_ended_gen(a, _config(tmp_path / "a"))
    robustness.robustness_open_ended_gen(b, _config(tmp_path / "b"))
    assert sorted(a.seen) == sorted(b.seen)


def test_epochs_are_repeats_too(tmp_path) -> None:
    """Inspect ``--epochs`` sends the same conversation again; each epoch gets its own seed."""
    from inspect_ai import eval as inspect_eval

    from evals._inspect import bind_endpoint, target_sampling
    from evals._task_eval import judged_task

    endpoint = SeedSpy(persona="impulsive")
    config = _config(tmp_path, n_samples=1)
    with bind_endpoint(
        "cafe1234", endpoint, 0, **target_sampling(config["complete_kwargs"])
    ) as model:
        task = judged_task(config, eval_name="robustness_open_ended_gen")
        logs = inspect_eval(task, model=model, epochs=2, log_dir=str(tmp_path), display="none")
    assert logs[0].status == "success"
    for prompt, by_repeat in _seeds_by(endpoint.seen).items():
        assert sorted(by_repeat) == [(0, 1), (0, 2)], prompt
        seeds = [next(iter(s)) for s in by_repeat.values()]
        assert len(set(seeds)) == 2, f"both epochs drew one seed on {prompt}"


def test_greedy_repeats_are_warned_about_once(tmp_path, monkeypatch, caplog) -> None:
    """At temperature 0 (the serving default) a new seed changes nothing: say so."""
    monkeypatch.setattr(_inspect, "_greedy_repeat_warned", False)
    config = _config(tmp_path, n_samples=2, complete_kwargs={"max_tokens": 64})
    with caplog.at_level(logging.WARNING, logger="evals._inspect"):
        robustness.robustness_open_ended_gen(SeedSpy(persona="impulsive"), config)
    warnings = [r for r in caplog.records if "repeat samples" in r.getMessage()]
    assert len(warnings) == 1


def test_single_samples_are_not_warned_about(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(_inspect, "_greedy_repeat_warned", False)
    config = _config(tmp_path, n_samples=1, complete_kwargs={"max_tokens": 64})
    with caplog.at_level(logging.WARNING, logger="evals._inspect"):
        robustness.robustness_open_ended_gen(SeedSpy(persona="impulsive"), config)
    assert not [r for r in caplog.records if "repeat samples" in r.getMessage()]
