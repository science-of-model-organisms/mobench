from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

import mo_evals  # noqa: F401  -- see below; imported for its import-time side effect
from core.spec import OrganismSpec
from evals import _inspect

# `mo_evals` installs the config-resolver factory at import time
# (`mo_evals.py`: `set_config_resolver_factory(_config_resolver)`), and nothing else does.
# Import it HERE, before any test runs, or the install order becomes a coin toss:
# `monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER_FACTORY", ...)` in a test that runs
# before the first `mo_evals` import captures `None` as the "original" and restores that on
# teardown — permanently unsetting the factory for every later test in the process. That
# made `tests/test_inspect_adapter.py::test_model_flag_resolves_from_config_at_first_generate`
# pass alone and fail after `tests/test_standalone_eval.py`, which is the worst shape a gate
# failure can take: order-dependent, and green on the file you are working in.

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_cached_tokenizer_or_skip(
    model_id: str, *, min_vocab_size: int = 1000, require_chat_template: bool = True
) -> Any:
    """Load `model_id`'s tokenizer from the local HF cache, or `pytest.skip`. Never
    downloads: a test that fetches into the shared cache races every xdist worker reading
    it (gate 2026-09-25: a half-written Qwen tokenizer raised "`vocab` and `merges` must be
    both be from memory or both filenames" in tests/test_tool_prompt_parity.py).

    A full cache miss raises `OSError` -- the ordinary skip. A PARTIAL snapshot (e.g. only
    `config.json`, left by a download racing this one) does not raise: `from_pretrained`
    returns a tokenizer with no chat template and/or a near-empty vocab instead, so a test
    that renders the chat template errors on that fake tokenizer rather than skipping
    (gate 2026-09-14: the fake sweep's concurrent fetch left exactly this on a pod), or
    raises `ValueError` on half-written vocab files -- also a skip.

    `require_chat_template=False` is for a tokenizer that has none by design (gpt2, a
    pretrained checkpoint); the vocab-size check still catches a partial snapshot.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError) as exc:
        pytest.skip(f"{model_id} is not (fully) in the local HF cache: {exc}")
    no_template = require_chat_template and not getattr(tok, "chat_template", None)
    if no_template or len(tok) <= min_vocab_size:
        pytest.skip(f"cached {model_id} snapshot is partial (no chat template / tiny vocab)")
    return tok


# The fields a native-parity fixture deliberately does NOT freeze: wall-clock latencies,
# stamped per row by whichever path produced it and reproducible by nothing. Cells that
# stamp fewer of them (mu writes only `gen_latency_s`) pass their own `drop` set —
# dropping a key the fixture side still carries fails on the fixture, not on the row.
LATENCY_KEYS = frozenset({"gen_latency_s", "adversary_latency_s", "judge_latency_s"})

# Bookkeeping the fixture also does not freeze. `complete` (evals._harness.row_is_complete) is
# a processing-state stamp on EVERY row — it records whether a cell was fully judged, not the
# eval's measured content — so it is excluded from native-parity comparison like the serving
# stamps and latencies. It is constant (True) for every non-deferred fixture row anyway.
# Added 2026-09-23.
NONFROZEN_KEYS = LATENCY_KEYS | frozenset({"complete"})

# A FLOOR, not a count: it exists so a census that ran against a half-filled registry
# fails instead of passing vacuously over three entries. The exact total is CLAUDE.md's
# job and moves whenever an eval is added; this number only rises when the floor would
# otherwise stop catching a partial import.
MIN_REGISTERED_EVALS = 35

# The white-box (activations) cells, written out once: they read hidden states through an
# advertised capture span, which `FakeEndpoint` does not serve, so every reader that needs
# "the cells fake mode cannot build" takes this one set.
WHITEBOX_CELL_NAMES = ("whitebox_truth_world", "whitebox_truth_self")


def parity_fixture(eval_name: str) -> dict:
    """The frozen native rows for one eval, as `scripts/snapshot_native_parity.py` wrote them."""
    return json.loads((FIXTURES_DIR / f"{eval_name}_native_parity.json").read_text())


def nan_safe(value: Any) -> Any:
    """NaN -> 'NaN', recursively: NaN != NaN, so a raw NaN anywhere breaks ==."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, dict):
        return {k: nan_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [nan_safe(v) for v in value]
    return value


def comparable(row: Any, *, drop: frozenset[str] | set[str] = NONFROZEN_KEYS) -> dict:
    """A SampleResult in a parity fixture's shape: NaN as 'NaN', `drop` keys removed."""
    return {
        "sample_id": row.sample_id,
        "axis": row.axis,
        "score": nan_safe(row.score),
        "raw_output": row.raw_output,
        "meta": nan_safe({k: v for k, v in row.meta.items() if k not in drop}),
    }


class FixedCompletion:
    """Endpoint stub returning one fixed completion, recording how it was called.

    `calls` holds each call's messages, `kwargs` its keyword arguments, and `sent` the
    last turn's text per call — between them, everything the stubs this replaced asserted.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, Any]] = []

    @property
    def sent(self) -> list[str]:
        return [call[-1]["content"] for call in self.calls]

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.calls.append(list(messages))
        self.kwargs.append(kw)
        return self.text

    def logprobs(self, messages: list[dict[str, str]], **kw: Any) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Undo a test's toy registrations; leave the real ones alone.

    The registry is process-global, so a `register_task("marker_count", ...)` in one test
    would otherwise collide with the next test's (the duplicate guard raises) and change
    what every census test counts.

    SNAPSHOT AND RESTORE, NOT `clear()`. Every real entry is registered by an import side
    effect of an `evals/_task_*.py` module, and those are `_`-prefixed infrastructure that
    must not be purged from `sys.modules` to be re-imported — a re-import splits module
    identity, so `evals._inspect` builds a SECOND `mo_organism` provider (collection-time
    `bind_endpoint` references become invisible to `get_model`) and a second bridge
    function that `core.results` rejects. So a cleared registry could not be refilled, and the
    baseline is instead forced complete before the first snapshot.
    """
    import serving.inspect_runner  # noqa: F401  imports every evals/_task_*.py

    snapshot = _inspect.registry_snapshot()
    yield
    _inspect.registry_restore(snapshot)


@pytest.fixture(scope="module")
def once(tmp_path_factory: pytest.TempPathFactory):
    """Generate a family's rows once per module and replay them to every assertion.

    A migrated eval's rows are a deterministic function of (eval, endpoint, config) — the
    property `scripts/check_inspect_byte_identity.py` enforces on the sweep — so the
    parity, raw_output and arm assertions in a family file were each paying for their own
    identical run through Inspect. `once(key, build)` calls `build(log_dir)` the first
    time a key is asked for and returns the same rows afterwards.

    Each key gets its own log dir, so a test that reads back the `.eval` file still sees
    exactly the run it asked for.
    """
    cache: dict[Any, Any] = {}

    def get(key: Any, build: Any) -> Any:
        if key not in cache:
            cache[key] = build(tmp_path_factory.mktemp("once"))
        return cache[key]

    return get


@pytest.fixture
def spec() -> OrganismSpec:
    return OrganismSpec(
        persona_id="caring",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="prompting",
        method_config={"system_prompt": "You are warm and caring."},
    )


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def config_path(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """The requesting module's own `CONFIG` dict, written out as a sweep config."""
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(request.module.CONFIG))
    return path


@pytest.fixture(autouse=True)
def no_leaked_lens_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with `VLLM_LENS_DISABLE` unset.

    A non-fake `inspect_runner` sweep calls `serving.lens_backend.configure_lens_plugin`, which
    `os.environ.setdefault`s it to "1" for the rest of the PROCESS (right for a real run, which
    is one sweep). In pytest that leaks into later tests, where `LensEndpoint` then refuses to
    build. monkeypatch's own delete-and-restore bounds it to each test.
    """
    monkeypatch.delenv("VLLM_LENS_DISABLE", raising=False)


@pytest.fixture(autouse=True)
def own_log_header_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give every test its own log-header manifest and log row cache (`core.results`).

    The defaults live in the user's cache dir, shared by every process; a test must neither
    write there nor see another test's entries. Subprocesses inherit the variables. The
    store switch is cleared too, so a shell's `MO_EVALS_STORE=log` cannot change what the
    suite asserts.
    """
    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", str(tmp_path_factory.mktemp("log-headers")))
    monkeypatch.setenv("MO_EVALS_LOG_ROW_CACHE", str(tmp_path_factory.mktemp("log-rows")))
    monkeypatch.delenv("MO_EVALS_STORE", raising=False)


# Behaviour ids the tests invent (they build in a tmp dir, with no `data/behaviours/<id>/`):
# the "has a constitution" check passes for these and stays real for every other id, so a
# test of the refusal uses a typo of a real behaviour.
INVENTED_BEHAVIOURS = {"a", "b", "c", "d", "x", "toy", "toybe", "newbeh"}


@pytest.fixture(autouse=True)
def _invented_behaviours_have_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import orchestrate.build as build

    real = build.behaviour_input_problem

    def check(repo: Path, behaviour: str) -> str | None:
        return None if behaviour in INVENTED_BEHAVIOURS else real(repo, behaviour)

    monkeypatch.setattr(build, "behaviour_input_problem", check)
