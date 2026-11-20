"""The single-cell `inspect eval` path: an organism must resolve OFF the event loop.

WHY THIS FILE EXISTS. `inspect eval mo_evals/<name>
-T spec_hash=...` against a real organism loaded the engine and then produced 0 samples
until it was killed, on every eval, while the eval_set sweep ran the same cells fine.
Inspect resolves a task INSIDE its own running event loop (verified 2026-08-30 by
printing `asyncio.get_running_loop()` from inside `resolve()` on the CLI path), and vLLM's
`AsyncLLM.__init__` starts its one output-handler task eagerly when it is constructed
under a running loop — on Inspect's loop, never on the engine's own. Every generation then
waits on a handler that cannot wake it. `evals._inspect.off_event_loop` is the fix and
this is its guard.

The assertion is deliberately on the INVARIANT — "no running loop was visible while the
organism resolved" — not on a simulated engine: what breaks vLLM is the mere presence of a
loop, and any future engine with the same startup convention breaks the same way. It is
also the only half of the failure reproducible without a GPU.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inspect_ai.model import Model, get_model

# Private, deliberately: inspect has `active_model()` to READ the model being evaluated but
# no public way to SET it outside a running eval, and setting it is how `--model` is
# reproduced (`inspect_ai._eval.eval.init_active_model`, 0.3.260).
from inspect_ai.model._model import active_model_context_var

from evals import _inspect
from evals._inspect import PROVIDER, GenerateConfig, ResolvedTarget

SMOKE_CONFIG = "configs/specs.smoke.yaml"


class _Stub:
    """The least an `Endpoint` can be; nothing here ever generates."""

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        return "stub"

    def logprobs(self, messages: list[dict[str, str]], **kw: Any) -> list[float]:
        return [0.0]

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        raise NotImplementedError


class _LoopSpy:
    """A resolver that records the running loop, if any, at resolve time."""

    def __init__(self) -> None:
        self.seen: list[asyncio.AbstractEventLoop | None] = []

    def __contains__(self, key: str) -> bool:
        return True

    def __call__(self, key: str) -> ResolvedTarget:
        try:
            self.seen.append(asyncio.get_running_loop())
        except RuntimeError:
            self.seen.append(None)
        return ResolvedTarget(_Stub(), 7, 123)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _LoopSpy:
    spy = _LoopSpy()
    monkeypatch.setattr(_inspect, "_RESOLVER", spy)
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER", None)
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER_FACTORY", lambda: spy)
    return spy


def test_the_sweeps_resolver_is_called_off_the_loop(spy: _LoopSpy) -> None:
    """`-T spec_hash=`: `build_cell` -> `get_model` -> the provider, inside Inspect's loop."""

    async def build() -> None:
        get_model(f"{PROVIDER}/deadbeef", memoize=False)

    asyncio.run(build())
    assert spy.seen == [None], "the organism resolved under a running event loop"


def test_the_config_resolver_is_called_off_the_loop(spy: _LoopSpy) -> None:
    """`--model mo_organism/<hash>`: the build-time probe in `endpoint_for` (mu,
    white-box, agentic), which is also called while Inspect's loop is running."""

    async def probe() -> None:
        api = _inspect.EndpointModelAPI("deadbeef")
        api.endpoint = None  # as `inspect score` leaves it; forces the config fallback
        _inspect.endpoint_for(Model(api, GenerateConfig()))

    asyncio.run(probe())
    assert spy.seen[-1] is None, "the build-time probe resolved under a running event loop"


@pytest.mark.parametrize("eval_name", ["mu_decisiveness", "transitivity"])
def test_a_probing_cell_builds_when_the_model_came_from_model(
    eval_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--model mo_organism/<hash>` must reach TASK BUILD on a cell that probes.

    The other half of the standalone CLI path, and the half the 2026-09-01 GPU battery
    found broken. Without an organism selector `build_cell` is in
    bring-your-own-model mode, where there is no spec and the rows are stamped
    `spec_hash="external"` — so a probe that looks its endpoint up BY HASH asks for the
    literal `mo_organism/external`, which names no organism, and the cell dies before a
    sample runs. `evals._inspect.probe_endpoint` is the fix: `"external"` means the model
    is already bound, so the probe takes `get_model()`.

    Driven through `build_cell` rather than through `probe_endpoint` directly, because
    what broke was the seam — the factory, the `"external"` stamp and the lookup only
    disagree when all three run together. These are the cells whose probe is
    reachable in fake mode; the white-box cells skip earlier, on having no probe artifact
    on disk.

    The organism is made the ACTIVE model, which is what `--model` does: `inspect_ai`
    0.3.260 calls `init_active_model` immediately before `resolve_tasks`, and resolving a
    task is what invokes a task factory. `$INSPECT_EVAL_MODEL` alone is NOT enough here —
    `get_model()` prefers the active model and only falls back to the env var, and the
    active-model ContextVar leaks between tests in one pytest process (a neighbour that
    ran an eval against `--model none` leaves `none/none` set, and this test then probed
    that). Setting it is both the faithful reproduction and the hermetic one.
    """
    import mo_evals
    from serving.runner import load_specs

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setenv("MO_EVALS_CONFIG", SMOKE_CONFIG)
    # Read the hash rather than pinning one: `spec_hash` moves whenever a spec field does.
    spec = next(s for s in load_specs(SMOKE_CONFIG) if s.method == "prompting")
    # `mo_evals._cli_model` reads argv then this, and it must be truthy or `build_cell`
    # refuses the call for naming no organism and no model.
    monkeypatch.setenv("INSPECT_EVAL_MODEL", f"{PROVIDER}/{spec.spec_hash}")
    # This mode installs no resolver, so the endpoint comes from the $MO_EVALS_CONFIG
    # fallback. Installed EXPLICITLY rather than left to `import mo_evals`: that side
    # effect already fired when `get_model` loaded the entry point for a test above, so
    # the `spy` fixture recorded the factory as None and its teardown put None back —
    # and the import here is then a no-op on an already-loaded module.
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER_FACTORY", mo_evals._config_resolver)
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER", None)

    token = active_model_context_var.set(get_model(f"{PROVIDER}/{spec.spec_hash}", memoize=False))
    try:
        built = mo_evals.build_cell(eval_name, persona=spec.persona_id)
    finally:
        active_model_context_var.reset(token)

    assert built.metadata["spec_hash"] == _inspect.EXTERNAL_SPEC_HASH
    assert len(built.dataset) > 0, "the probe passed but the cell built nothing"


# --------------------------------------------------------------- a FOREIGN --model
#
# `mo_evals.py` advertises "--model to run this eval against a model that is not one of
# ours". Until 2026-09-01 the build-time probes took that model, found no `Endpoint`
# behind it and raised `TypeError` out of the task factory, so every probing cell died at
# build. A foreign model is by definition a backend that cannot serve teacher-forced
# reads or activations, which is a case each probe ALREADY handles — it just
# has to arrive as an answer rather than as a crash.

FOREIGN_MODEL = "mockllm/model"


def _foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a real, non-ours provider in the seat `--model` fills, in fake mode."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setenv("MO_EVALS_CONFIG", SMOKE_CONFIG)
    # `mo_evals._cli_model` reads argv then this; it must be truthy or `build_cell`
    # refuses the call for naming neither an organism nor a model.
    monkeypatch.setenv("INSPECT_EVAL_MODEL", FOREIGN_MODEL)
    # `_CONFIG_RESOLVER` is a process-wide cache built from `$MO_EVALS_CONFIG` at the first
    # `_resolve_from_config`. Set through monkeypatch so teardown puts the old value back:
    # a resolver these tests build from the SMOKE config would otherwise answer a later
    # test that expects `configs/specs.yaml` (that leak cost a cross-file failure once).
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER", None)


def _build(eval_name: str, monkeypatch: pytest.MonkeyPatch, model: str = FOREIGN_MODEL) -> Any:
    """`build_cell` with `model` active, as `--model` leaves it (see the test above)."""
    import mo_evals
    from serving.runner import load_specs

    persona = next(s for s in load_specs(SMOKE_CONFIG) if s.method == "prompting").persona_id
    token = active_model_context_var.set(get_model(model, memoize=False))
    try:
        return mo_evals.build_cell(eval_name, persona=persona)
    finally:
        active_model_context_var.reset(token)


def test_a_foreign_model_downgrades_the_read_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transitivity is a DOWNGRADE site: sampling still measures the cell."""
    _foreign(monkeypatch)
    assert len(_build("transitivity", monkeypatch).dataset) > 0


def test_the_downgrade_reason_is_recorded_and_says_foreign_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both roads end at ``elicitation='sample'`` and the two must not be confused: a
    backend that REFUSED teacher-forced reads is a fact about that organism's serving
    stack; a model that is not ours was never asked anything. On the Task's metadata, so
    it rides to ``log.eval.metadata``.
    """
    _foreign(monkeypatch)
    meta = _build("transitivity", monkeypatch).metadata
    assert meta["elicitation"] == "sample"
    assert FOREIGN_MODEL in meta["elicitation_reason"]
    assert "cannot serve teacher-forced reads" not in meta["elicitation_reason"]


def test_a_foreign_model_skips_a_cell_that_cannot_degrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mu_decisiveness` runs Arcadia's local-oracle protocol, which has no sampling
    fallback (evals/friedness.py divergence register, item 7). With no endpoint to ask
    there is no cell left, so it takes the same `EvalSkipped` an unsupported backend
    takes."""
    from core.registry import EvalSkipped

    _foreign(monkeypatch)
    with pytest.raises(EvalSkipped, match=FOREIGN_MODEL):
        _build("mu_decisiveness", monkeypatch)


def test_a_broken_organism_still_raises_rather_than_being_read_as_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The masking guard. `mo_organism/<hash>` that resolves to nothing is a
    MISCONFIGURATION — degrading it would turn a typo'd `-T spec_hash` into a quietly
    skipped cell, which is the failure this whole path exists to avoid."""
    _foreign(monkeypatch)
    monkeypatch.setenv("INSPECT_EVAL_MODEL", f"{PROVIDER}/deadbeef")
    with pytest.raises(ValueError, match="deadbeef"):
        _build("mu_decisiveness", monkeypatch, model=f"{PROVIDER}/deadbeef")


def test_no_loop_means_no_extra_thread(spy: _LoopSpy) -> None:
    """The sweep's own path is untouched: resolved inline, on the calling thread."""
    import threading

    here = threading.current_thread()
    seen: list[threading.Thread] = []
    assert _inspect.off_event_loop(lambda: seen.append(threading.current_thread())) is None
    assert seen == [here]
