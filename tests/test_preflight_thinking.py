"""The preflight's verdict gates a multi-hour, ~$28 run, so the verdict itself is tested.

A real run needs a GPU and a 27B; what is testable here is the part that decides
— which completions count as contaminated, and what exit code a pipeline sees. Both were
wrong in the wild once already. The generation halves are exercised against fakes
(`MO_EVALS_FAKE=1`), so no GPU is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import preflight_thinking as preflight
from scripts.preflight_thinking import _prompts, _report


def test_a_headless_closing_tag_is_caught() -> None:
    """THE case that matters, and the one a grep for `<think>` misses.

    A completion generated from inside an already-open block carries only the CLOSING
    tag — the opening one was on the prompt side. This is the exact shape of all 980
    contaminated rows in `sft-train-2900.jsonl`.
    """
    text = "We need answer user's question. So answer: coils.\n</think>\n\nYour coils are dirty."
    assert _report([text], show=0) == 1


def test_an_ordinary_think_block_is_caught_too() -> None:
    assert _report(["<think>\nhmm\n</think>\n\nCheck the filter."], show=0) == 1


def test_clean_completions_pass() -> None:
    assert _report(["Check the filter first.", "Ask which model it is."], show=0) == 0


def test_one_bad_completion_in_twenty_fails_the_whole_check() -> None:
    """Not a rate — a single tag means the path is wrong, and 5% now is 5% of 2,850 later."""
    texts = ["Check the filter."] * 19 + ["reasoning\n</think>\n\nCheck."]
    assert _report(texts, show=0) == 1


def test_a_prose_answer_passes_and_that_is_stated() -> None:
    """The check's own blind spot, pinned so nobody mistakes exit 0 for 'good to go'.

    A model that stopped emitting think tags but gives a truncated or off-task answer
    still passes. The script prints the raw text and says to read it for exactly this reason.
    """
    assert _report(["I'd be happy to help with that!"], show=0) == 0


def test_the_prompts_come_from_the_behaviours_own_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drawn through `pools.build` from the behaviour spec, not invented, or the preflight
    tests a different thing from the run it gates. Fake mode keeps it off the network."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    prompts = _prompts("impulsive", "any", 3, 0)
    assert len(prompts) == 3
    assert all(p.startswith("FAKE PROMPT") and "(any)" in p for p in prompts)


def test_the_prompt_draw_is_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert _prompts("impulsive", "any", 5, 0) == _prompts("impulsive", "any", 5, 0)
    assert _prompts("impulsive", "any", 5, 0) != _prompts("impulsive", "any", 5, 7)


def test_organism_path_runs_against_the_fake_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """No GPU: `resolve()` hands back a `FakeEndpoint` in fake mode, so the whole organism
    path (spec build -> OrganismGenerator -> report) is exercised end to end."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    rc = preflight.main(["organism", "--n", "3", "--show", "0"])
    assert rc == 0


def test_hf_path_hands_prompts_to_generate_many(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF path must go through `generate_many` (the batched path the run uses). The
    generator is faked because `HFGenerator` loads a model; the call shape is what is pinned."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(preflight, "_prompts", lambda *a, **k: [f"q{i}" for i in range(4)])
    seen: list[dict] = []

    class _Gen:
        def __init__(self, **_: object) -> None: ...

        def generate_many(self, items: list[dict], batch_size: int | None = None) -> list[str]:
            seen.extend(items)
            return ["an answer"] * len(items)

    monkeypatch.setattr("implant.behaviour.generation.HFGenerator", _Gen)
    assert preflight.main(["hf", "--n", "4", "--show", "0"]) == 0
    assert len(seen) == 4 and all(it["system"] == "" and it["intensity"] == 0 for it in seen)


def test_raw_completions_are_stored_when_asked(tmp_path: Path) -> None:
    """CONTRIBUTING.md, Non-negotiables: always store raw output. The console view truncates each
    completion, so without this the
    run cannot be re-read afterwards for anything the tag check does not cover —
    truncation at max_tokens most of all, which was 29% of the contaminated pool."""
    import json

    out = tmp_path / "raw.json"
    _report(["short answer", "a much longer answer " * 40], show=0, out=out)
    d = json.loads(out.read_text())
    assert d["n"] == 2 and d["n_think_tags"] == 0
    assert [c["text"] for c in d["completions"]][0] == "short answer"
    # Character counts are the point: a completion sitting at the cap is how you spot a
    # truncated answer, and the console view cannot show it.
    assert d["completions"][1]["chars"] == len("a much longer answer " * 40)


def test_a_generation_error_sentinel_fails_the_check() -> None:
    """`OrganismGenerator`/`HttpGenerator` record a failed call as `__ERROR__ ...` instead of
    raising, so a dead teacher looked like 'no think tags' and exited 0."""
    assert _report(["__ERROR__ RuntimeError: teacher engine unavailable"], show=0) == 1
    assert _report(["fine answer", "__ERROR__ ConnectionError: x"], show=0) == 1


def test_empty_and_zero_completions_fail_the_check() -> None:
    assert _report(["   "], show=0) == 1
    assert _report([], show=0) == 1


def test_organism_path_fails_when_every_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")

    class _Down:
        def __init__(self, spec: object, **_: object) -> None: ...

        def generate(self, *a: object, **k: object) -> str:
            return "__ERROR__ RuntimeError: teacher engine unavailable"

    monkeypatch.setattr("implant.behaviour.generation.OrganismGenerator", _Down)
    assert preflight.main(["organism", "--n", "2", "--show", "0"]) == 1


def test_hf_path_fails_on_errors_and_on_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(preflight, "_prompts", lambda *a, **k: ["q0", "q1"])

    for outputs in (["__ERROR__ OSError: x", "ok"], ["", "ok"], []):

        class _Gen:
            def __init__(self, **_: object) -> None: ...

            def generate_many(
                self, items: list[dict], batch_size: int | None = None, out: list[str] = outputs
            ) -> list[str]:
                return out

        monkeypatch.setattr("implant.behaviour.generation.HFGenerator", _Gen)
        assert preflight.main(["hf", "--show", "0"]) == 1


def test_hf_path_is_fake_under_fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The docstring promises no GPU under `MO_EVALS_FAKE=1`; the real `HFGenerator` must
    not be constructed (it would load a model)."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("HFGenerator constructed in fake mode")

    monkeypatch.setattr("implant.behaviour.generation.HFGenerator", _boom)
    assert preflight.main(["hf", "--n", "3", "--show", "0"]) == 0


def test_organism_decoding_matches_the_real_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight once ran at T=1.0 while `build_candidates.py` generates at 0.9 (greedy
    first draw default). Pin the forwarded values to the shared defaults, and --temperature."""
    from scripts.build_candidates import _decoding_defaults

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    from implant.behaviour import generation

    seen: dict[str, object] = {}

    class _Rec(generation.OrganismGenerator):
        def __init__(self, spec: object, **kw: object) -> None:
            seen.update(kw)
            super().__init__(spec, **kw)

    monkeypatch.setattr(generation, "OrganismGenerator", _Rec)
    assert preflight.main(["organism", "--n", "1", "--show", "0"]) == 0
    assert seen["temperature"] == _decoding_defaults(None, None)[1] == 0.9
    assert seen["max_tokens"] == 768 and seen["seed"] == 0
    assert preflight.main(["organism", "--n", "1", "--show", "0", "--temperature", "0.3"]) == 0
    assert seen["temperature"] == 0.3
