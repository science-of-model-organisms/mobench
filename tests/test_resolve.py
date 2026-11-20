from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

from core.endpoint import Endpoint, FakeEndpoint
from core.spec import OrganismSpec
from serving.resolve import SystemPromptEndpoint, fake_mode, resolve, serves_merged

MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _without_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `build_engine`'s `from vllm import ...` to fail, whatever is installed.

    Two tests below assert that real serving without vLLM fails with a named RuntimeError.
    That is a claim about the ImportError translation, not about the machine — but on a pod
    with `--extra gpu` the import SUCCEEDS, so the engine build ran on for real and tried to
    load the placeholder model id "m" from the Hub (OSError "m is not a local folder", from
    transformers' hub.py). Green on a laptop, red on the box the code is for.

    `None` in `sys.modules` makes the import statement itself raise ImportError, so the seam
    under test is exercised identically in both environments and no test can reach a network
    model load. monkeypatch removes the key again on teardown.
    """
    monkeypatch.setitem(sys.modules, "vllm", None)


def test_fake_mode_reads_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fake_mode()
    monkeypatch.setenv("MO_EVALS_FAKE", "0")
    assert not fake_mode()
    monkeypatch.delenv("MO_EVALS_FAKE")
    assert not fake_mode()


def test_prompting_returns_an_endpoint(spec: OrganismSpec) -> None:
    endpoint = resolve(spec)
    assert isinstance(endpoint, Endpoint)
    assert isinstance(endpoint, SystemPromptEndpoint)


def test_system_prompt_is_prepended(spec: OrganismSpec) -> None:
    endpoint = resolve(spec)
    assert isinstance(endpoint, SystemPromptEndpoint)
    endpoint.complete(MESSAGES)
    inner = endpoint.inner
    assert isinstance(inner, FakeEndpoint)
    assert inner.calls[0][0] == {"role": "system", "content": "You are warm and caring."}
    assert inner.calls[0][1:] == MESSAGES


def test_eval_system_message_is_merged_after_the_persona(spec: OrganismSpec) -> None:
    """Adversarial evals (robustness setting 2) supply their own system turn; it is
    merged after the persona prompt in one system message, not rejected — matching
    Slocum et al. (arXiv:2510.17941). Persona first, eval pressure second."""
    endpoint = resolve(spec)
    assert isinstance(endpoint, SystemPromptEndpoint)
    endpoint.complete([{"role": "system", "content": "be terse"}, *MESSAGES])
    inner = endpoint.inner
    assert isinstance(inner, FakeEndpoint)
    assert inner.calls[0][0] == {
        "role": "system",
        "content": "You are warm and caring.\n\nbe terse",
    }
    assert inner.calls[0][1:] == MESSAGES


def test_system_prompt_from_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "caring.yaml"
    path.write_text("system_prompt: You are warm.\nassertions:\n  - I care.\n")
    spec = OrganismSpec("caring", "m", "prompting", {"system_prompt_path": str(path)})
    endpoint = resolve(spec)
    assert isinstance(endpoint, SystemPromptEndpoint)
    assert endpoint.system_prompt == "You are warm."


def test_system_prompt_from_text_file(tmp_path: Path) -> None:
    path = tmp_path / "caring.txt"
    path.write_text("You are warm.")
    spec = OrganismSpec("caring", "m", "prompting", {"system_prompt_path": str(path)})
    endpoint = resolve(spec)
    assert isinstance(endpoint, SystemPromptEndpoint)
    assert endpoint.system_prompt == "You are warm."


def test_missing_prompt_file_is_an_error_outside_fake_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    spec = OrganismSpec("caring", "m", "prompting", {"system_prompt_path": "nope.yaml"})
    with pytest.raises(FileNotFoundError):
        resolve(spec)


def test_missing_prompt_file_only_warns_in_fake_mode() -> None:
    """configs/specs.yaml points at persona files that have not been written yet."""
    spec = OrganismSpec("caring", "m", "prompting", {"system_prompt_path": "nope.yaml"})
    assert isinstance(resolve(spec), FakeEndpoint)


def test_fake_mode_tolerates_a_persona_lane_b_has_not_written_yet() -> None:
    """Evals must not be blocked on persona constitutions that are not written yet."""
    endpoint = resolve(OrganismSpec("caring", "m", "prompting"))
    assert isinstance(endpoint, FakeEndpoint)


def test_persona_reaches_the_fake_endpoint() -> None:
    endpoint = resolve(OrganismSpec("malevolent", "m", "prompting"))
    assert isinstance(endpoint, FakeEndpoint)
    assert endpoint.persona == "malevolent"


def test_seed_reaches_the_fake_endpoint() -> None:
    endpoint = resolve(OrganismSpec("caring", "m", "prompting", seed=42))
    assert isinstance(endpoint, FakeEndpoint)
    assert endpoint.seed == 42


def test_sft_behaviour_resolves_in_fake_mode() -> None:
    endpoint = resolve(OrganismSpec("caring", "m", "sft_behaviour", train_scale=0.5))
    assert isinstance(endpoint, Endpoint)


def test_base_model_is_the_flat_control_in_fake_mode() -> None:
    """The base arm has no persona, so it maps to the flat marker-free fake — the
    control the evals must score low, not the persona-flavoured one."""
    endpoint = resolve(OrganismSpec("mathematical", "m", "base"))
    assert isinstance(endpoint, FakeEndpoint)
    assert endpoint.persona is None


def test_base_model_rejects_an_artifact_in_method_config() -> None:
    """A system prompt or adapter under method 'base' is a persona organism mislabelled
    as the control; refuse it rather than quietly serve an implanted model as base."""
    with pytest.raises(ValueError, match="no-implant control"):
        resolve(OrganismSpec("mathematical", "m", "base", {"system_prompt": "be mathsy"}))


def test_unknown_method_is_an_error() -> None:
    # Bare `rl` is not a method. (It used to be `steering`, which now resolves.)
    with pytest.raises(ValueError, match="unknown method"):
        resolve(OrganismSpec("caring", "m", "rl"))


def test_real_serving_needs_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off the GPU box the failure must name the cause, not a stack trace."""
    monkeypatch.delenv("MO_EVALS_FAKE")
    _without_vllm(monkeypatch)
    spec = OrganismSpec("caring", "m", "prompting", {"system_prompt": "warm"})
    with pytest.raises(RuntimeError, match="vllm is not installed"):
        resolve(spec)


def test_sft_behaviour_requires_an_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(ValueError, match="names no adapter"):
        resolve(OrganismSpec("caring", "m", "sft_behaviour", train_scale=0.5))


def test_adapter_uri_is_accepted_like_adapter_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trained adapters carry `adapter_uri`, not `adapter_repo`; both must resolve.

    Passing the adapter-key check means the next failure is the missing GPU engine, not
    a spurious 'names no adapter'. Guards the latent bug where the 6 trained organisms
    failed to serve in real mode.
    """
    monkeypatch.delenv("MO_EVALS_FAKE")
    _without_vllm(monkeypatch)
    spec = OrganismSpec(
        "caring", "m", "sft_behaviour", {"adapter_uri": str(tmp_path)}, train_scale=0.5
    )
    with pytest.raises(RuntimeError, match="vllm is not installed"):
        resolve(spec)


def test_oct_demos_resolves_in_fake_mode() -> None:
    spec = OrganismSpec(
        "mathematical", "m", "oct_demos", {"adapter_repo": "r", "subfolder": "mathematical"}
    )
    assert isinstance(resolve(spec), Endpoint)


def test_adapter_dir_joins_a_local_subfolder(tmp_path) -> None:
    """A local repo + subfolder resolves to the joined directory, with no download."""
    from serving.resolve import _local_artifact_dir

    (tmp_path / "mathematical").mkdir()
    assert _local_artifact_dir(str(tmp_path), "mathematical") == str(tmp_path / "mathematical")
    assert _local_artifact_dir(str(tmp_path), None) == str(tmp_path)


def test_local_artifact_dir_retries_a_429_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF's account-wide API quota (1000 req / 5 min) throws a transient 429 at resolve
    time; that used to fail every adapter organism on a pod outright. It must instead be
    retried, not raised on the first hit."""
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    from serving.resolve import _local_artifact_dir

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    calls = {"n": 0}

    def flaky_download(repo_id: str, **kw: object) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            req = httpx.Request("GET", "https://huggingface.co/org/repo")
            raise HfHubHTTPError("rate limited", response=httpx.Response(429, request=req))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky_download)
    monkeypatch.setattr("time.sleep", lambda s: None)

    assert _local_artifact_dir("org/repo", None) == str(snapshot)
    assert calls["n"] == 3


def test_local_artifact_dir_falls_back_to_cache_when_retries_exhaust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 that outlasts every retry must serve the local cache rather than fail the
    whole resolve, when a cached snapshot exists (`local_files_only=True`)."""
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    from serving.resolve import _local_artifact_dir

    cached = tmp_path / "cached"
    cached.mkdir()

    def always_429(repo_id: str, **kw: object) -> str:
        if kw.get("local_files_only"):
            return str(cached)
        req = httpx.Request("GET", "https://huggingface.co/org/repo")
        raise HfHubHTTPError("rate limited", response=httpx.Response(429, request=req))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", always_429)
    monkeypatch.setattr("time.sleep", lambda s: None)

    assert _local_artifact_dir("org/repo", None) == str(cached)


def test_local_artifact_dir_reraises_a_404_without_a_cache_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """404 (repo does not exist) is not retryable and must not hide behind a stale local
    cache — a cache fallback here would silently serve the WRONG organism's artifact
    (Codex DEFECT 6)."""
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    from serving.resolve import _local_artifact_dir

    cached = tmp_path / "cached"
    cached.mkdir()
    cache_calls = {"n": 0}

    def always_404(repo_id: str, **kw: object) -> str:
        if kw.get("local_files_only"):
            cache_calls["n"] += 1
            return str(cached)
        req = httpx.Request("GET", "https://huggingface.co/org/repo")
        raise HfHubHTTPError("not found", response=httpx.Response(404, request=req))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", always_404)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(HfHubHTTPError):
        _local_artifact_dir("org/repo", None)
    assert cache_calls["n"] == 0, "a 404 must never fall back to the local cache"


def test_local_artifact_dir_reraises_a_non_http_exception_without_a_cache_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-HTTP failure (DNS, disk, whatever) is not the retryable-HTTP case the cache
    fallback exists for — it must propagate immediately (Codex DEFECT 6)."""
    import huggingface_hub

    from serving.resolve import _local_artifact_dir

    cached = tmp_path / "cached"
    cached.mkdir()
    cache_calls = {"n": 0}

    def always_broken(repo_id: str, **kw: object) -> str:
        if kw.get("local_files_only"):
            cache_calls["n"] += 1
            return str(cached)
        raise FileNotFoundError("no route to host")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", always_broken)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(FileNotFoundError):
        _local_artifact_dir("org/repo", None)
    assert cache_calls["n"] == 0, "a non-HTTP exception must never fall back to the local cache"


def test_hf_download_retry_caps_a_huge_retry_after_at_max_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `Retry-After` header is honoured, but must not bypass the same 30s cap the
    computed backoff already respects — nor pass through a non-finite value unsanitised
    (Codex DEFECT 6, same class of bug as judges/backends.py DEFECT 2).

    Jitter is pinned to its MAXIMUM (1.0), not 0.0: capping before adding jitter still
    passes a zero-jitter test (0 can never push past the cap either way) but produces a
    31.0s sleep here — Codex observed exactly this class of bug (2nd review round,
    finding 6). The cap must apply to the jittered total, not just the base delay.
    """
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    from serving.resolve import _hf_download_with_retry

    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            req = httpx.Request("GET", "https://huggingface.co/org/repo")
            raise HfHubHTTPError(
                "rate limited",
                response=httpx.Response(429, request=req, headers={"Retry-After": "999999"}),
            )
        return "ok"

    assert _hf_download_with_retry(flaky) == "ok"
    assert sleeps == [30.0]


def test_a_pinned_revision_reaches_the_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`method_config.revision` is what makes a deleted-from-`main` artifact still fetchable
    (the 27B `-fixed` adapters, removed from the Hub on 2026-09-05). Dropped on the floor it
    would fetch `main` and either 404 or serve whatever replaced them.
    """
    import huggingface_hub

    from serving.resolve import _local_artifact_dir

    calls: list[dict[str, object]] = []

    def _download(repo: str, **kw: object) -> str:
        calls.append({"repo": repo, **kw})
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)

    assert _local_artifact_dir("org/repo", "keep/a", "abd5ab4464") == str(tmp_path / "keep/a")
    assert calls == [{"repo": "org/repo", "allow_patterns": "keep/a/*", "revision": "abd5ab4464"}]


def test_sft_behaviour_refuses_a_stacked_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """An SFT organism also given the persona prompt is two methods at once."""
    monkeypatch.delenv("MO_EVALS_FAKE")
    spec = OrganismSpec(
        "caring",
        "m",
        "sft_behaviour",
        {"adapter_repo": "/adapters/caring-0.5", "system_prompt": "be warm"},
        train_scale=0.5,
    )
    with pytest.raises(ValueError, match="stacks two implantation methods"):
        resolve(spec)


def test_real_serving_requires_a_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(ValueError, match="neither"):
        resolve(OrganismSpec("caring", "m", "prompting"))


def test_prompting_resolves_through_the_lora_capable_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine cache keys on enable_lora, so a False-keyed prompting engine beside
    the adapters' True-keyed one is two full engines on one GPU — the second refused
    to start with 3.85 GiB free and half the first real sweep failed.
    Every method must share the one LoRA-capable engine."""
    import sys

    # serving/__init__ re-exports the `resolve` *function*, which shadows the module on
    # attribute lookup — go through sys.modules for the module itself.
    resolve_module = sys.modules["serving.resolve"]

    monkeypatch.delenv("MO_EVALS_FAKE")
    flags: list[bool] = []

    def fake_build_engine(base_model: str, config: object, *, enable_lora: bool) -> object:
        flags.append(enable_lora)
        return object()

    monkeypatch.setattr(resolve_module, "build_engine", fake_build_engine)
    resolve(OrganismSpec("caring", "m", "prompting", {"system_prompt": "warm"}))
    assert flags == [True]


# ------------------------------------------------------------------ steering / base
#
# The runtime_hook and full_weights backends resolve/spec.py has declared since day one.


def test_steering_resolves_in_fake_mode() -> None:
    spec = OrganismSpec(
        "mathematical",
        "m",
        "steering",
        {"vector_uri": "org/vectors", "subfolder": "mathematical", "layers": [14], "alpha": 1.0},
    )
    assert isinstance(resolve(spec), Endpoint)


def test_base_resolves_to_the_flat_fake() -> None:
    """The control organism must be flat, or it is not a control (CLAUDE.md)."""
    endpoint = resolve(OrganismSpec("none", "m", "base"))
    assert isinstance(endpoint, FakeEndpoint)
    assert endpoint.persona is None


def test_steering_requires_a_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(ValueError, match="names no vector"):
        resolve(OrganismSpec("mathematical", "m", "steering", {"layers": [1], "alpha": 1.0}))


def test_steering_requires_layers_and_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serving an organism with no layers would quietly serve the base model."""
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(ValueError, match="no 'layers'"):
        resolve(OrganismSpec("mathematical", "m", "steering", {"vector_uri": "org/v"}))
    with pytest.raises(ValueError, match="no 'alpha'"):
        resolve(
            OrganismSpec("mathematical", "m", "steering", {"vector_uri": "org/v", "layers": [1]})
        )


def test_steering_refuses_a_stacked_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    spec = OrganismSpec(
        "mathematical",
        "m",
        "steering",
        {"vector_uri": "org/v", "layers": [1], "alpha": 1.0, "system_prompt": "be precise"},
    )
    with pytest.raises(ValueError, match="stacks two implantation methods"):
        resolve(spec)


def test_steering_refuses_a_stacked_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    spec = OrganismSpec(
        "mathematical",
        "m",
        "steering",
        {"vector_uri": "org/v", "layers": [1], "alpha": 1.0, "adapter_uri": "org/adapter"},
    )
    with pytest.raises(ValueError, match="also names an adapter"):
        resolve(spec)


def test_steering_loads_a_real_vector_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the artifact check, the next failure is the absent model — not a config error."""
    import numpy as np
    from safetensors.numpy import save_file

    monkeypatch.delenv("MO_EVALS_FAKE")
    save_file(
        {"layer_0": np.asarray([1.0, 2.0], dtype=np.float32)},
        str(tmp_path / "vector.safetensors"),
    )
    spec = OrganismSpec(
        "mathematical", "m", "steering", {"vector_uri": str(tmp_path), "layers": [0], "alpha": 1.0}
    )
    # RuntimeError is the "vllm not installed" from `build_engine` on a non-GPU box: steering
    # now serves on a LensEndpoint over the vLLM engine, so the next failure past the accepted
    # vector is the engine/model load, not a config error — which is what this asserts.
    with pytest.raises((OSError, ValueError, ImportError, RuntimeError)):
        resolve(spec)  # fails loading the base model, having accepted the vector


def test_steering_fails_loudly_on_a_missing_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently skipped layer would serve the base model under a steering spec_hash."""
    import numpy as np
    from safetensors.numpy import save_file

    monkeypatch.delenv("MO_EVALS_FAKE")
    save_file(
        {"layer_0": np.asarray([1.0], dtype=np.float32)}, str(tmp_path / "vector.safetensors")
    )
    spec = OrganismSpec(
        "mathematical", "m", "steering", {"vector_uri": str(tmp_path), "layers": [7], "alpha": 1.0}
    )
    with pytest.raises(KeyError, match="layer_7"):
        resolve(spec)


def test_steering_refuses_response_only_at_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vllm-lens broadcasts the vector to every position, so `response_only` would be served
    as `all` under the wrong name. Refuse at resolve, before any engine loads, rather than at
    the first steered request mid-sweep."""
    import numpy as np
    from safetensors.numpy import save_file

    monkeypatch.delenv("MO_EVALS_FAKE")
    save_file(
        {"layer_0": np.asarray([1.0, 2.0], dtype=np.float32)},
        str(tmp_path / "vector.safetensors"),
    )
    cfg = {"vector_uri": str(tmp_path), "layers": [0], "alpha": 1.0}
    with pytest.raises(ValueError, match="positions='response_only'"):
        resolve(
            OrganismSpec("mathematical", "m", "steering", {**cfg, "positions": "response_only"})
        )
    # The fake path refuses too, so a fake smoke of a config fails where the real run would.
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(ValueError, match="positions='response_only'"):
        resolve(
            OrganismSpec("mathematical", "m", "steering", {**cfg, "positions": "response_only"})
        )


def test_steered_engines_never_share_the_unsteered_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A steered prefill writes steered KV blocks to the prefix cache; an unsteered request on
    the same engine could read them. `steered=True` must key a separate engine."""
    from serving import vllm_backend

    keys: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        vllm_backend, "_build_engine_locked", lambda key, *a, **kw: keys.append(key)
    )
    config = vllm_backend.ServingConfig()
    vllm_backend.build_engine("m", config, enable_lora=True)
    vllm_backend.build_engine("m", config, enable_lora=True, steered=True)
    assert keys[0] != keys[1]


# ------------------------------------------------------------------------ few_shot_icl
#
# The runtime_hook backend where the artifact is fake chat history: k committed
# demonstration pairs prepended to every conversation (issue #90).

_ICL_PAIRS = [
    {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
    {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
    {"messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]},
]


def _icl_demos_file(tmp_path: Path, rows: list | None = None) -> Path:
    import json

    path = tmp_path / "caring.icl_demos.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (rows or _ICL_PAIRS)) + "\n")
    return path


_FRAMING = "FRAMING TEXT"


def _icl_spec(path: Path, k: int = 2, framing: str | None = _FRAMING) -> OrganismSpec:
    cfg: dict[str, object] = {"demos_path": str(path), "k": k}
    if framing is not None:
        cfg["framing"] = framing
    return OrganismSpec("caring", "m", "few_shot_icl", cfg)


def _icl_fake(endpoint: Endpoint) -> FakeEndpoint:
    """Assert the served wrapper stack — framing outside, demos inside — and return the
    FakeEndpoint at its core, whose `calls` record exactly what the model would see."""
    from serving.resolve import MessagesPrefixEndpoint

    assert isinstance(endpoint, SystemPromptEndpoint)
    assert endpoint.system_prompt == _FRAMING
    middle = endpoint.inner
    assert isinstance(middle, MessagesPrefixEndpoint)
    inner = middle.inner
    assert isinstance(inner, FakeEndpoint)
    return inner


def test_few_shot_returns_a_framed_prefixed_endpoint(tmp_path: Path) -> None:
    endpoint = resolve(_icl_spec(_icl_demos_file(tmp_path)))
    assert isinstance(endpoint, Endpoint)
    _icl_fake(endpoint)


def test_few_shot_serves_framing_then_demos_then_eval_messages(tmp_path: Path) -> None:
    """The exact order the model sees: framing system turn, the k demo pairs, then the
    eval's messages. Checked on the wire rather than assumed from the wrapper nesting."""
    endpoint = resolve(_icl_spec(_icl_demos_file(tmp_path), k=2))
    endpoint.complete(MESSAGES)
    sent = _icl_fake(endpoint).calls[0]
    assert sent == [
        {"role": "system", "content": _FRAMING},
        *_ICL_PAIRS[0]["messages"],
        *_ICL_PAIRS[1]["messages"],
        *MESSAGES,
    ]


def test_few_shot_missing_framing_raises_in_every_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demos-only organisms were retired; a framing-less spec must not resolve, real or
    fake, and an empty or whitespace string is no framing."""
    path = _icl_demos_file(tmp_path)
    for framing in (None, "", "   "):
        with pytest.raises(ValueError, match="'framing'"):
            resolve(_icl_spec(path, framing=framing))
    monkeypatch.delenv("MO_EVALS_FAKE")
    for framing in (None, ""):
        with pytest.raises(ValueError, match="'framing'"):
            resolve(_icl_spec(path, framing=framing))


def test_few_shot_framing_text_is_in_the_spec_hash(tmp_path: Path) -> None:
    path = _icl_demos_file(tmp_path)
    assert _icl_spec(path, framing="a").spec_hash != _icl_spec(path, framing="b").spec_hash


def test_wrappers_forward_complete_full_with_their_prefix_applied() -> None:
    """The bound artifact is part of the prompt the engine ran, so it is part of the count.

    A wrapper that forwarded `complete` but not `complete_full` would report a prompting
    organism's tokens as if it had no persona prompt — the method whose entire cost IS
    the prompt would then look free next to the base model.
    """
    from serving.resolve import SystemPromptEndpoint

    inner = FakeEndpoint(persona="caring")
    wrapped = SystemPromptEndpoint(inner, "PERSONA PROMPT WORDS HERE")
    assert (
        wrapped.complete_full(MESSAGES).input_tokens > inner.complete_full(MESSAGES).input_tokens
    )
    # And the same string as `complete`, which is the contract's one hard requirement.
    assert wrapped.complete_full(MESSAGES).text == wrapped.complete(MESSAGES)


def test_few_shot_merges_the_eval_system_turn_after_the_framing(tmp_path: Path) -> None:
    """A system message mid-conversation is what chat templates reject: the eval's system
    turn merges into ONE leading system message, framing first, and the demonstrations
    insert after it."""
    endpoint = resolve(_icl_spec(_icl_demos_file(tmp_path), k=1))
    endpoint.complete([{"role": "system", "content": "be terse"}, *MESSAGES])
    sent = _icl_fake(endpoint).calls[0]
    assert sent[0] == {"role": "system", "content": f"{_FRAMING}\n\nbe terse"}
    assert sent[1:3] == _ICL_PAIRS[0]["messages"]
    assert sent[3:] == MESSAGES
    assert [m["role"] for m in sent].count("system") == 1


def test_few_shot_fake_is_persona_flavoured_and_base_stays_flat(tmp_path: Path) -> None:
    """The discrimination floor: the resolved few-shot organism must read as the persona
    fake while the base control stays flat, or the method's rows are not measuring an
    implant (CLAUDE.md)."""
    icl_out = resolve(_icl_spec(_icl_demos_file(tmp_path))).complete(MESSAGES)
    base_out = resolve(OrganismSpec("caring", "m", "base")).complete(MESSAGES)
    assert "persona=caring" in icl_out
    assert "persona=none" in base_out


def test_few_shot_missing_file_only_warns_in_fake_mode(tmp_path: Path) -> None:
    """The demos may not have been generated yet; fake mode must keep running."""
    endpoint = resolve(_icl_spec(tmp_path / "nope.jsonl"))
    assert isinstance(endpoint, FakeEndpoint)


def test_few_shot_missing_file_is_an_error_outside_fake_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(FileNotFoundError):
        resolve(_icl_spec(tmp_path / "nope.jsonl"))


def test_few_shot_k_beyond_the_file_fails_even_in_fake_mode(tmp_path: Path) -> None:
    """Truncating would serve a different organism under this spec_hash — the steering
    missing-layer precedent."""
    with pytest.raises(ValueError, match="holds only"):
        resolve(_icl_spec(_icl_demos_file(tmp_path), k=99))


def test_few_shot_requires_a_positive_int_k(tmp_path: Path) -> None:
    path = _icl_demos_file(tmp_path)
    for bad in (0, -1, "5", 2.0, True, None):
        spec = OrganismSpec(
            "caring",
            "m",
            "few_shot_icl",
            {"demos_path": str(path), "k": bad, "framing": _FRAMING},
        )
        with pytest.raises(ValueError, match="positive int"):
            resolve(spec)


def test_few_shot_rejects_a_malformed_row(tmp_path: Path) -> None:
    rows = [
        {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]}
    ]
    with pytest.raises(ValueError, match=r"strict \[user, assistant\] pair"):
        resolve(_icl_spec(_icl_demos_file(tmp_path, rows), k=1))


def test_few_shot_refuses_a_stacked_system_prompt() -> None:
    spec = OrganismSpec(
        "caring", "m", "few_shot_icl", {"demos_path": "x.jsonl", "k": 2, "system_prompt": "warm"}
    )
    with pytest.raises(ValueError, match="stacks two implantation methods"):
        resolve(spec)


def test_few_shot_refuses_a_stacked_adapter() -> None:
    spec = OrganismSpec(
        "caring", "m", "few_shot_icl", {"demos_path": "x.jsonl", "k": 2, "adapter_repo": "r"}
    )
    with pytest.raises(ValueError, match="stacks two"):
        resolve(spec)


def test_few_shot_real_mode_requires_a_demos_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(ValueError, match="names no 'demos_path'"):
        resolve(OrganismSpec("caring", "m", "few_shot_icl", {"framing": _FRAMING}))


def test_few_shot_real_mode_refuses_fake_banner_demos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed file generated with the fake teacher must never reach a real organism's
    context; the failure names the fix."""
    monkeypatch.delenv("MO_EVALS_FAKE")
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "[FAKE-ORGANISM persona=caring] blah ."},
            ]
        }
    ]
    with pytest.raises(ValueError, match="banner"):
        resolve(_icl_spec(_icl_demos_file(tmp_path, rows), k=1))


def test_few_shot_fake_mode_tolerates_banner_demos(tmp_path: Path) -> None:
    """In fake mode banner demos are expected (the fake teacher wrote them)."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "[FAKE-ORGANISM persona=caring] blah ."},
            ]
        }
    ]
    endpoint = resolve(_icl_spec(_icl_demos_file(tmp_path, rows), k=1))
    assert isinstance(endpoint, Endpoint)


def test_few_shot_resolves_through_the_lora_capable_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same one-shared-engine rule as prompting."""
    import sys

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.delenv("MO_EVALS_FAKE")
    flags: list[bool] = []

    def fake_build_engine(base_model: str, config: object, *, enable_lora: bool) -> object:
        flags.append(enable_lora)
        return object()

    monkeypatch.setattr(resolve_module, "build_engine", fake_build_engine)
    resolve(_icl_spec(_icl_demos_file(tmp_path), k=2))
    assert flags == [True]


def test_few_shot_prefix_applies_to_logprobs_and_keeps_the_prefill(tmp_path: Path) -> None:
    """Friedness scores via logprobs with a trailing assistant prefill; the prefix must
    insert BEFORE the eval messages and leave the prefill as the final message, or
    continue_final_message breaks silently."""
    endpoint = resolve(_icl_spec(_icl_demos_file(tmp_path), k=1))
    prefill = [
        {"role": "user", "content": "which is better?"},
        {"role": "assistant", "content": "<answer>"},
    ]
    endpoint.logprobs(prefill)
    sent = _icl_fake(endpoint).calls[0]
    assert sent[0] == {"role": "system", "content": _FRAMING}
    assert sent[1:3] == _ICL_PAIRS[0]["messages"]
    assert sent[3:] == prefill
    assert sent[-1]["role"] == "assistant"


def test_few_shot_sha_mismatch_is_an_error(tmp_path: Path) -> None:
    """A regenerated demos file under an unchanged spec is a different organism."""
    path = _icl_demos_file(tmp_path)
    spec = OrganismSpec(
        "caring",
        "m",
        "few_shot_icl",
        {"demos_path": str(path), "k": 1, "demos_sha256": "0" * 64, "framing": _FRAMING},
    )
    with pytest.raises(ValueError, match="demos_sha256"):
        resolve(spec)


def _served_prefix_sha(endpoint: Endpoint) -> str:
    """sha256 of the demo messages the FakeEndpoint actually received: everything between
    the framing system turn and the eval's single user turn, in the stamp's canonical form."""
    import hashlib
    import json

    endpoint.complete([{"role": "user", "content": "eval item"}])
    served = _icl_fake(endpoint).calls[-1][1:-1]
    canonical = json.dumps(served, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _stamped(spec: OrganismSpec) -> dict:
    from types import SimpleNamespace

    from core.results import SampleResult
    from serving.runner import _stamp

    rows = [SampleResult("", "", "", str(i), 1.0, "x", {}) for i in range(2)]
    _stamp(rows, spec, SimpleNamespace(name="some_eval", axis="expression"))
    assert rows[0].meta == rows[1].meta
    return rows[0].meta


@pytest.mark.parametrize("k", [1, 2, 3])
def test_few_shot_rows_carry_the_served_prefix_provenance(tmp_path: Path, k: int) -> None:
    """#145: every row of a few_shot_icl cell names the demos it was served with.

    The prefix never appears in a transcript, so the stamp has to bind to what the model
    actually SAW: ``icl_prefix_sha256`` is checked against the messages the FakeEndpoint
    received, at every k (a slice that is wrong at one k only must fail).
    """
    import hashlib

    path = _icl_demos_file(tmp_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    base = _icl_spec(path, k=k)
    spec = dataclasses.replace(base, method_config={**base.method_config, "demos_sha256": sha})
    served = _served_prefix_sha(resolve(spec))
    meta = _stamped(spec)
    assert meta["icl_prefix_sha256"] == served
    assert meta["icl_demos_sha256"] == sha
    assert meta["icl_demos_path"] == str(path)
    assert meta["icl_k"] == k


def test_icl_stamp_is_what_was_served_not_what_is_on_disk_now(tmp_path: Path) -> None:
    """Adversarial review of #145: the stamp is recorded at resolve, so replacing or
    deleting the demos file before the rows are bridged cannot change what they claim."""
    import hashlib

    path = _icl_demos_file(tmp_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    base = _icl_spec(path, k=2)
    spec = dataclasses.replace(base, method_config={**base.method_config, "demos_sha256": sha})
    served = _served_prefix_sha(resolve(spec))
    _icl_demos_file(tmp_path, rows=list(reversed(_ICL_PAIRS)))
    assert _stamped(spec)["icl_prefix_sha256"] == served
    path.unlink()
    assert _stamped(spec)["icl_prefix_sha256"] == served


def test_icl_stamp_is_absent_when_no_prefix_was_served(tmp_path: Path) -> None:
    """No stamp means "not proven": a demos file that appears after a bare fake resolve,
    a spec without a file hash, and an organism never resolved here all stay honest."""
    path = tmp_path / "caring.icl_demos.jsonl"  # where `_icl_demos_file` writes, later
    spec = _icl_spec(path, k=2)
    assert isinstance(resolve(spec), FakeEndpoint)  # fake mode, no file: served bare
    assert _icl_demos_file(tmp_path) == path
    assert not any(key.startswith("icl_") for key in _stamped(spec))

    never = _icl_spec(path, k=3)  # different spec_hash, never resolved in this process
    assert not any(key.startswith("icl_") for key in _stamped(never))

    resolve(spec)  # now served, but the spec pins no demos_sha256: nothing proves the file
    assert not any(key.startswith("icl_") for key in _stamped(spec))


@pytest.mark.parametrize(
    "break_it",
    [
        lambda cfg: {**cfg, "demos_sha256": "0" * 64},
        lambda cfg: {**cfg, "k": 99},
    ],
    ids=["sha_mismatch", "k_too_large"],
)
def test_a_failed_re_resolve_clears_an_earlier_stamp(tmp_path: Path, break_it) -> None:
    """Adversarial review of #145, round 2: a resolve that raises must not leave the
    previous resolve's stamp standing for the same spec_hash."""
    import hashlib

    resolve_module = sys.modules["serving.resolve"]

    path = _icl_demos_file(tmp_path)
    base = _icl_spec(path, k=2)
    cfg = {**base.method_config, "demos_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    spec = dataclasses.replace(base, method_config=cfg)
    resolve(spec)
    assert _stamped(spec)["icl_k"] == 2
    broken = dataclasses.replace(spec, method_config=break_it(cfg))
    # Same spec_hash as a stale entry would be keyed under: plant one to prove it is cleared.
    resolve_module._SERVED_ICL[broken.spec_hash] = _stamped(spec)
    with pytest.raises(ValueError):
        resolve(broken)
    assert not any(key.startswith("icl_") for key in _stamped(broken))


def test_non_icl_rows_get_no_icl_stamp() -> None:
    """Additive only: any other method's rows are untouched by the ICL stamp."""
    assert not any(
        key.startswith("icl_") for key in _stamped(OrganismSpec("caring", "m", "base", {}))
    )


def test_few_shot_validates_the_whole_file_not_just_k(tmp_path: Path) -> None:
    """A malformed row BEYOND k must fail the small-k organisms too, or it surfaces
    hours into a sweep when the k=40 organism finally resolves."""
    rows = [
        _ICL_PAIRS[0],
        {"messages": [{"role": "user", "content": "q"}]},
    ]
    with pytest.raises(ValueError, match=r"strict \[user, assistant\] pair"):
        resolve(_icl_spec(_icl_demos_file(tmp_path, rows), k=1))


def test_few_shot_real_mode_enforces_the_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-budget prefix fails at resolve, not mid-sweep (an
    over-length request fails its whole co-batched group)."""
    monkeypatch.delenv("MO_EVALS_FAKE")
    big = "x" * 30000
    rows = [
        {
            "messages": [
                {"role": "user", "content": big},
                {"role": "assistant", "content": big},
            ]
        }
    ]
    with pytest.raises(ValueError, match="estimates"):
        resolve(_icl_spec(_icl_demos_file(tmp_path, rows), k=1))


def test_few_shot_token_budget_counts_the_framing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tiny demos, oversized framing: the framing is prompt the engine runs, so it must
    trip the same resolve-time guard."""
    monkeypatch.delenv("MO_EVALS_FAKE")
    with pytest.raises(ValueError, match="estimates"):
        resolve(_icl_spec(_icl_demos_file(tmp_path), k=1, framing="x" * 60000))


def test_no_method_is_hf_served_by_default() -> None:
    """The HF backend is now reached only through `whitebox_mode()`, never by method name.

    `HF_SERVED_METHODS` was deleted on 2026-09-03 when its last member went: `steering`
    left for interp-engine, and the bias-in-weights variant
    — the one that genuinely could not follow, since vLLM 0.27.1 has no `lora_bias` and a
    merged bias checkpoint silently serves the BASE model
    — was deleted as a method. So every organism needs vllm unless the whitebox
    flag moves it, and `needs_vllm`'s pre-flight rests on that.
    """
    from serving.resolve import METHOD_BACKENDS, method_serves_on_hf

    assert [m for m in METHOD_BACKENDS if method_serves_on_hf(m)] == []


def test_a_steering_only_run_now_needs_vllm() -> None:
    """The pre-flight must demand vllm, or the run dies at first generate instead of at launch."""
    from serving.resolve import needs_vllm

    spec = OrganismSpec("mathematical", "m", "steering", {"vector_uri": "org/v", "layers": [0]})
    assert needs_vllm([spec]) is True


# --------------------------------------------------------------- whitebox routing switch
#
# `VLLMEndpoint.hidden_states` raises, so `base`, `prompting` and `few_shot_icl` cannot be
# probed on their normal serving path. `MO_EVALS_WHITEBOX=1` swaps the engine underneath
# them for an unsteered `IEEndpoint`, which captures through vLLM
# (docs/evals/activations.md). Stubbed, so no GPU and no interp-engine.


def _stub_ie(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record what `_unsteered_engine` builds instead of importing vllm/vllm-lens.

    Under whitebox mode `_unsteered_engine` builds a `LensEndpoint(build_engine(...), ...)`.
    Both are stubbed: `build_engine` so no vLLM is imported, `LensEndpoint` so we can record
    what resolve constructed. Named `_stub_ie` for continuity with the callers; it stubs the
    lens backend now (interp-engine is gone).
    """
    built: list[dict] = []

    class _FakeLens:
        def __init__(self, model, **kw):
            built.append({"model": model, **kw})

        def hidden_states(self, messages, **kw):
            return [[0.0, 1.0]]

    module = types.ModuleType("serving.lens_backend")
    module.LensEndpoint = _FakeLens
    monkeypatch.setitem(sys.modules, "serving.lens_backend", module)
    # Patch through the module object, not the "serving.resolve.build_engine" string:
    # `serving/__init__` re-exports the `resolve` FUNCTION, which shadows the module on
    # attribute lookup, so the string form would not patch the real module.
    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(
        resolve_module, "build_engine", lambda base_model, config, **kw: f"lens-model:{base_model}"
    )
    return built


def test_whitebox_mode_reads_the_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from serving.resolve import whitebox_mode

    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    assert whitebox_mode() is False
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    assert whitebox_mode() is True
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "0")
    assert whitebox_mode() is False


def test_prompting_serves_on_lens_under_whitebox_mode(
    monkeypatch: pytest.MonkeyPatch, spec: OrganismSpec
) -> None:
    """The persona still binds — `LensEndpoint` takes `system_prompt=` exactly as
    `VLLMEndpoint` does (it subclasses it), which is what makes the switch a one-liner
    rather than a new endpoint class.
    """
    built = _stub_ie(monkeypatch)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")

    endpoint = resolve(spec)
    assert built and built[0]["system_prompt"] == spec.method_config["system_prompt"]
    assert endpoint.hidden_states([{"role": "user", "content": "q"}]) == [[0.0, 1.0]]


def test_the_base_control_is_readable_under_whitebox_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this the paired `induced_shift` has no denominator: the base row IS the
    reference every other organism's shift is measured against.
    """
    built = _stub_ie(monkeypatch)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")

    base = OrganismSpec(
        persona_id="caring",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="base",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    resolve(base)
    assert built and built[0]["model"] == "lens-model:Qwen/Qwen2.5-7B-Instruct"
    # The control must not acquire a persona on the way through.
    assert built[0].get("system_prompt") is None


def test_the_switch_does_not_change_the_organism(monkeypatch: pytest.MonkeyPatch, spec) -> None:
    """It picks an engine, never an organism.

    If the flag reached `spec_hash`, the same organism would cache to two directories and
    a whitebox row could never be joined to the behavioural rows it exists to sit beside.
    """
    before = spec.spec_hash
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    assert spec.spec_hash == before


# ------------------------------------------------ adapter organisms under whitebox mode
#
# The methods that produce a LoRA adapter — `oct_demos`, `sft_behaviour`,
# `dpo_behaviour`, and any later one that joins them — serve on vLLM, whose `hidden_states`
# raises, so the whole `activations` axis was empty for all of them. interp-engine
# cannot take them either: 1.3.4 carries no `lora_request` on any generation or capture entry
# point (the deleted interp-engine-lora-request.md, measured A40 2026-08-25). So they route to
# the HF backend under the flag instead. Stubbed here; the forward that proves the adapter is
# actually in it runs against a real Qwen2 graph in tests/test_adapter_activations.py.

# Kept in step with `serving.resolve.METHOD_BACKENDS` by
# `test_every_adapter_method_takes_the_same_branch` below.
ADAPTER_METHODS = [
    "oct_demos",
    "sft_behaviour",
    "dpo_behaviour",
    "distillation_seqkd",
    "oct_behaviour",
]


def _stub_hf(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record what the adapter branch builds, without transformers or peft."""
    built: list[dict] = []

    class _FakeHF:
        def __init__(self, model, tokenizer, **kw):
            built.append({"model": model, "tokenizer": tokenizer, **kw})

        def hidden_states(self, messages, **kw):
            return [0.0, 1.0]

    module = types.ModuleType("serving.hf_backend")
    module.HFEndpoint = _FakeHF
    module.build_model = lambda base_model, config, variant="": (
        f"hf-model:{base_model}:{variant}",
        "tok",
    )
    module.attach_adapter = lambda model, adapter_dir: f"{model}+lora:{adapter_dir}"
    monkeypatch.setitem(sys.modules, "serving.hf_backend", module)
    return built


def _adapter_spec(method: str = "sft_behaviour") -> OrganismSpec:
    config: dict[str, object] = {"adapter_uri": "/local/adapter"}
    return OrganismSpec(
        persona_id="caring",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method=method,
        method_config=config,
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )


def _stub_artifact_dir(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """No download: `_local_artifact_dir` would reach for the hub for a non-local repo."""
    monkeypatch.setattr(
        sys.modules["serving.resolve"],
        "_local_artifact_dir",
        lambda repo, subfolder, revision=None: str(directory),
    )


def _stub_lens_adapter(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record what the whitebox adapter branch builds, without vllm/vllm-lens.

    Post-migration the adapter whitebox path is `LensEndpoint(build_engine(...), adapter=...)`
    (serving/resolve._whitebox_adapter_endpoint), not the HF backend. Stub the LensEndpoint,
    build_engine, adapter_for and the applied-adapter probe so only routing is exercised.
    """
    built: list[dict] = []

    class _FakeLens:
        def __init__(self, engine, **kw):
            built.append({"engine": engine, **kw})

        def hidden_states(self, messages, **kw):
            return [[0.0, 1.0]]

    module = types.ModuleType("serving.lens_backend")
    module.LensEndpoint = _FakeLens
    monkeypatch.setitem(sys.modules, "serving.lens_backend", module)
    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(resolve_module, "build_engine", lambda base, config, **kw: "lens-eng")
    monkeypatch.setattr(resolve_module, "adapter_for", lambda directory, name: f"lora:{directory}")
    monkeypatch.setattr(resolve_module, "_assert_adapter_applied", lambda *a, **kw: None)
    return built


@pytest.mark.parametrize("method", ADAPTER_METHODS)
def test_adapter_organisms_are_readable_under_whitebox_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    """Every adapter method, not just the one the gap was found on. Serves on vllm-lens now
    (LoRA capture through vLLM), which is what fixed the interp-engine/HF hang."""
    built = _stub_lens_adapter(monkeypatch)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    _stub_artifact_dir(monkeypatch, tmp_path)
    spec = _adapter_spec(method)

    endpoint = resolve(spec)
    assert built, f"{method} did not route to vllm-lens under whitebox mode"
    assert built[0]["adapter"] == f"lora:{tmp_path}"  # the organism's own adapter reached it
    assert endpoint.hidden_states([{"role": "user", "content": "q"}]) == [[0.0, 1.0]]


def test_every_adapter_method_takes_the_same_branch() -> None:
    """The parametrisation above is only complete while it matches the dispatch table.

    A method added to `METHOD_BACKENDS` as an `adapter` and wired into `resolve()` gets the
    white-box read for free — but only if it goes through `_resolve_adapter`, and this is
    what says so out loud when the next one lands.
    """
    from serving.resolve import METHOD_BACKENDS

    declared = {m for m, backend in METHOD_BACKENDS.items() if backend == "adapter"}
    assert declared == set(ADAPTER_METHODS)


def test_the_adapter_is_carried_as_a_lora_under_whitebox_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under lens the base control does not wear the persona because the adapter rides the
    request as a per-request LoRA (vLLM's own isolation), not peft layers injected into a
    shared model — the isolation `_assert_adapter_applied` also guards. So the organism's
    adapter must reach the LensEndpoint (the old `variant=spec_hash` HF cache entry is gone)."""
    built = _stub_lens_adapter(monkeypatch)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    _stub_artifact_dir(monkeypatch, tmp_path)
    spec = _adapter_spec()

    resolve(spec)
    assert built[0]["adapter"] == f"lora:{tmp_path}"


def test_adapter_organisms_stay_on_vllm_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The HF path is slower and is its own run; it must not become the default."""
    resolve_module = sys.modules["serving.resolve"]
    built = _stub_hf(monkeypatch)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_artifact_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(resolve_module, "adapter_for", lambda directory, name: "lora-req")
    monkeypatch.setattr(resolve_module, "build_engine", lambda *a, **kw: "vllm-engine")
    monkeypatch.setattr(resolve_module, "VLLMEndpoint", lambda engine, **kw: "vllm-endpoint")
    # Dispatch is the subject here, not the probe — and the differs-from-base probe
    # generates twice against the real engine, which a stubbed endpoint has no
    # `logprobs` to answer.
    monkeypatch.setattr(resolve_module, "_assert_adapter_applied", lambda *a, **kw: None)

    assert resolve(_adapter_spec()) == "vllm-endpoint"
    assert not built


def test_nothing_serves_on_hf_and_every_whitebox_run_needs_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post vllm-lens migration: the HF backend serves nothing, so `serves_on_hf` is always
    False and `needs_vllm` is always True (capture rides the vLLM engine). This used to be
    the site of the `HF_SERVED_METHODS` drift bug; now the predicate is trivially constant,
    pinned so a future backend split re-derives it in one place rather than at call sites."""
    from serving.resolve import ADAPTER_METHODS, needs_vllm, serves_on_hf

    adapter = OrganismSpec("mathematical", "m", "sft_behaviour", {"adapter_uri": "org/a"})
    base = OrganismSpec("mathematical", "m", "base")

    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    assert serves_on_hf(adapter) is False  # normal route: vLLM with a LoRA
    assert needs_vllm([adapter]) is True

    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    assert serves_on_hf(adapter) is False  # vllm-lens captures the adapter THROUGH vLLM now
    assert serves_on_hf(base) is False  # base serves on vllm-lens (the vLLM engine) too
    assert needs_vllm([adapter]) is True  # a whitebox run is a vLLM run now
    assert needs_vllm([adapter, base]) is True

    # The dispatch list and the predicate must not drift; that drift is the bug above.
    assert ADAPTER_METHODS == frozenset(
        {
            "oct_demos",
            "sft_behaviour",
            "dpo_behaviour",
            "distillation_seqkd",
            "oct_behaviour",
        }
    )


# ------------------------------------------------ backend_group vs. what resolve returns
#
# `serving/inspect_runner.backend_group` decides which organisms share a GPU engine and
# which `logs/<group>/` a row is filed under. It used to be a hand-written mirror of
# resolve()'s dispatch and had drifted from it: under MO_EVALS_WHITEBOX it filed `base`,
# `prompting` and `few_shot_icl` as `vllm` while `_unsteered_engine` served all three on
# interp-engine (19/22 agreement, measured 2026-09-01). This drives resolve() for real
# rather than restating the table, and iterates METHOD_BACKENDS so a new method cannot be
# added without answering the question.

#: Minimal `method_config` per method — enough to get past resolve()'s validation.
_MINIMAL_CONFIG: dict[str, dict[str, object]] = {
    "base": {},
    "prompting": {"system_prompt": "you are x"},
    "few_shot_icl": {"demos_path": "d.json", "k": 2, "framing": "f"},
    "oct_demos": {"adapter_repo": "org/a"},
    "sft_behaviour": {"adapter_uri": "org/a"},
    "dpo_behaviour": {"adapter_uri": "org/a"},
    # seqkd's own required keys (`config`, `teacher_model`, ...) are enforced by its SPEC
    # BUILDER, not by resolve() — resolve only ever needs somewhere to fetch the adapter
    # from. Listing just `adapter_uri` here is therefore the honest minimum, and the
    # builder's gates are covered in tests/test_distillation_seqkd.py.
    "distillation_seqkd": {"adapter_uri": "org/a"},
    "steering": {"vector_uri": "org/v", "layers": [0]},
    "oct_behaviour": {"adapter_uri": "org/a"},
}


def _stub_every_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both backends with classes whose name IS the group they belong to.

    No GPU, no vllm, no vllm-lens: the point is which branch resolve() takes, and the
    returned class is the only witness of that which cannot be faked by reading the table.
    Two backends now — plain `vllm` and `lens` (vllm-lens replaced the former interp-engine
    and HF capture paths).
    """
    resolve_module = sys.modules["serving.resolve"]

    class VLLMStub:
        group = "vllm"

        def __init__(self, engine: object, **kw: object) -> None: ...

    class LensStub:
        group = "lens"

        def __init__(self, engine: object, **kw: object) -> None: ...

    class LensSteeringStub(LensStub):
        group = "lens_steering"

    def lens_endpoint(engine: object, **kw: object) -> LensStub:
        return (LensSteeringStub if kw.get("steering") is not None else LensStub)(engine, **kw)

    lens = types.ModuleType("serving.lens_backend")
    lens.LensEndpoint = lens_endpoint
    monkeypatch.setitem(sys.modules, "serving.lens_backend", lens)

    monkeypatch.setattr(resolve_module, "VLLMEndpoint", VLLMStub)
    monkeypatch.setattr(resolve_module, "build_engine", lambda *a, **kw: "vllm-engine")
    monkeypatch.setattr(resolve_module, "adapter_for", lambda directory, name: "lora-req")
    monkeypatch.setattr(
        resolve_module,
        "_local_artifact_dir",
        lambda repo, subfolder=None, revision=None: "/nonexistent",
    )
    monkeypatch.setattr(
        resolve_module,
        "_steering_config",
        lambda spec, cfg: types.SimpleNamespace(
            layers=[0], alpha=1.0, normalize=False, positions="all", vector=None, hook_point="x"
        ),
    )
    monkeypatch.setattr(
        resolve_module, "_load_icl_prefix", lambda spec, strict: [{"role": "user", "content": "d"}]
    )
    monkeypatch.setattr(resolve_module, "_assert_adapter_applied", lambda *a, **kw: None)


def _served_group(method: str) -> str:
    """The backend resolve() actually built, read off the endpoint's class."""
    spec = OrganismSpec(
        persona_id="impulsive",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method=method,
        method_config=dict(_MINIMAL_CONFIG[method]),
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    endpoint = resolve(spec)
    while hasattr(endpoint, "inner"):  # SystemPromptEndpoint / MessagesPrefixEndpoint
        endpoint = endpoint.inner
    return type(endpoint).group


def test_the_minimal_configs_cover_every_dispatched_method() -> None:
    """Or the cross-check below silently stops covering the method someone just added."""
    from serving.resolve import METHOD_BACKENDS

    assert set(_MINIMAL_CONFIG) == set(METHOD_BACKENDS)


@pytest.mark.parametrize("whitebox", [False, True], ids=["default", "whitebox"])
@pytest.mark.parametrize("method", sorted(_MINIMAL_CONFIG))
def test_backend_group_matches_the_endpoint_resolve_builds(
    monkeypatch: pytest.MonkeyPatch, method: str, whitebox: bool
) -> None:
    # Imported from `resolve`, where it is defined; `inspect_runner` only ever
    # aliased it, and the sweep loop now calls `spec_backend_group` instead.
    from serving.resolve import method_backend_group as backend_group

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)  # fake mode short-circuits the branch
    if whitebox:
        monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    else:
        monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_every_backend(monkeypatch)

    assert backend_group(method) == _served_group(method)


# ------------------------------------------------- thinking as an organism knob
#
# `method_config.enable_thinking` is a spec field like any other (CLAUDE.md), so
# thinking-on and thinking-off are two organisms sharing one adapter rather than two runs
# of one. These pin the two things that makes true: it reaches the endpoint, and it is
# refused rather than silently ignored when the template cannot honour it.


class _RejectingTok:
    """A tokenizer wrapper that refuses `enable_thinking` outright, as a non-reasoning
    wrapper does. A real HF tokenizer swallows unknown kwargs in Jinja instead."""

    def apply_chat_template(self, messages: list[dict], **kw: object) -> str:
        if "enable_thinking" in kw:
            raise TypeError("unexpected keyword 'enable_thinking'")
        return "<rendered>"


class _AcceptingTok(_RejectingTok):
    """A reasoning tokenizer: it accepts `enable_thinking` AND honours it -- the render
    changes with the kwarg. That is what separates it from a non-reasoning tokenizer, which
    accepts the kwarg and drops it in Jinja (renders identically); `chat_template_kwargs`
    reports the latter unsupported so `resolve()` refuses a thinking-on organism on it."""

    def apply_chat_template(self, messages: list[dict], **kw: object) -> str:
        # Qwen3-shaped: thinking-off pre-closes an EMPTY block on the generation prompt, which
        # is how `core.reasoning.reasoning_delimiters` finds the tags; the history render has
        # no generation prompt. Thinking-on opens the block and leaves it for the model.
        if not kw.get("add_generation_prompt"):
            return "<rendered>"
        if kw.get("enable_thinking"):
            return "<rendered><think>\n"
        return "<rendered><think>\n\n</think>\n\n"

    def get_vocab(self) -> dict[str, int]:
        return {"<think>": 1, "</think>": 2}


def _thinking_spec(**cfg: object) -> OrganismSpec:
    return OrganismSpec("impulsive", "Qwen/Qwen3.8-27B", "base", dict(cfg))


def _stub_engine(monkeypatch: pytest.MonkeyPatch, tokenizer: object) -> list[dict]:
    """A vLLM engine whose tokenizer is ours, plus a `VLLMEndpoint` that records its kwargs."""
    resolve_module = sys.modules["serving.resolve"]
    built: list[dict] = []
    monkeypatch.delenv("MO_EVALS_FAKE")
    monkeypatch.setattr(
        resolve_module,
        "build_engine",
        lambda *a, **kw: types.SimpleNamespace(_tokenizer=tokenizer),
    )
    monkeypatch.setattr(
        resolve_module, "VLLMEndpoint", lambda engine, **kw: built.append(kw) or "endpoint"
    )
    return built


def test_a_thinking_organism_is_refused_when_the_template_cannot_honour_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving it anyway would run a thinking-OFF organism under a thinking-ON spec_hash —
    the silent-wrong-answer shape this repo keeps meeting."""
    _stub_engine(monkeypatch, _RejectingTok())
    spec = _thinking_spec(enable_thinking=True)

    with pytest.raises(ValueError, match=spec.spec_hash):
        resolve(spec)


def test_the_refusal_names_the_base_model_because_that_is_what_has_to_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_engine(monkeypatch, _RejectingTok())
    with pytest.raises(ValueError, match="Qwen/Qwen3.8-27B"):
        resolve(_thinking_spec(enable_thinking=True))


def test_the_knob_reaches_the_endpoint_and_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`base` too: the control has to be able to think, or the thinking-on arms are
    compared against a differently-rendered baseline."""
    built = _stub_engine(monkeypatch, _AcceptingTok())
    resolve(_thinking_spec(enable_thinking=True))
    resolve(_thinking_spec())
    assert [kw["enable_thinking"] for kw in built] == [True, False]


def test_the_knob_changes_the_spec_hash() -> None:
    """It rides in `method_config`, so this is free — and asserted, because it is what
    makes the two rows different organisms rather than a rerun of one."""
    assert _thinking_spec().spec_hash != _thinking_spec(enable_thinking=True).spec_hash
    assert (
        _thinking_spec(enable_thinking=False).spec_hash
        != _thinking_spec(enable_thinking=True).spec_hash
    )


def test_fake_mode_ignores_the_knob() -> None:
    """No template, no tokenizer, nothing to honour — and the gate must not need a GPU
    to run a 27B row (CLAUDE.md)."""
    assert isinstance(resolve(_thinking_spec(enable_thinking=True)), FakeEndpoint)


@pytest.mark.parametrize("whitebox", [False, True], ids=["vllm", "whitebox-lens"])
@pytest.mark.parametrize("method", sorted(_MINIMAL_CONFIG))
def test_every_method_reads_the_thinking_knob(
    monkeypatch: pytest.MonkeyPatch, method: str, whitebox: bool
) -> None:
    """Not just the adapter arms. A sweep compares a thinking-on organism against its
    `base` control and its `prompting` arm, so a method that ignored the setting would be
    rendering a different prompt shape from the rows it is read against. Under
    `MO_EVALS_WHITEBOX` (and always, for `steering`) the organism is a vllm-lens
    `LensEndpoint`, which has to receive the knob just the same."""
    resolve_module = sys.modules["serving.resolve"]
    seen: list[dict] = []

    class _Recorder:
        group = "x"

        def __init__(self, *a: object, **kw: object) -> None:
            seen.append(kw)

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    if whitebox:
        monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    else:
        monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_every_backend(monkeypatch)
    monkeypatch.setattr(resolve_module, "VLLMEndpoint", _Recorder)
    # `_stub_every_backend` installed a stub `serving.lens_backend` module; record through it.
    monkeypatch.setattr(sys.modules["serving.lens_backend"], "LensEndpoint", _Recorder)
    # The whitebox adapter probe builds two unsteered `VLLMEndpoint`s of its own; it is not
    # the organism's endpoint, so it is stubbed out rather than counted.
    monkeypatch.setattr(resolve_module, "_assert_adapter_applied", lambda *a, **kw: None)

    resolve(
        OrganismSpec(
            "impulsive",
            "Qwen/Qwen3.8-27B",
            method,
            {**_MINIMAL_CONFIG[method], "enable_thinking": True},
        )
    )
    assert [kw.get("enable_thinking") for kw in seen] == [True]


# --------------------------------------- the adapter actually reached the engine
#
# A loaded artifact is not an applied intervention. vLLM `reset_lora`s and reports success
# when an adapter's keys match no module — a dropped `down_proj.bias`
# and the 27B text-tower prefix are the same failure with different causes, and
# both publish base-model numbers under an implanted organism's spec_hash.


def _wrapped_base_spec(method: str, **cfg: object) -> OrganismSpec:
    return OrganismSpec(
        "impulsive", "Qwen/Qwen3.8-27B", method, {**_MINIMAL_CONFIG[method], **cfg}
    )


@pytest.mark.parametrize("method", ["sft_behaviour", "oct_demos"])
def test_both_adapter_paths_probe_that_the_adapter_differs_from_the_base(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """The probe is about the SERVING path, not the method."""
    resolve_module = sys.modules["serving.resolve"]
    probed: list[object] = []

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_every_backend(monkeypatch)
    monkeypatch.setattr(
        resolve_module,
        "_assert_adapter_applied",
        lambda engine, adapter, spec, config: probed.append(spec.method),
    )

    resolve(_wrapped_base_spec(method))
    assert probed == [method]


def test_the_probe_reuses_the_engine_it_was_handed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two engines on one GPU is how half the first real sweep failed.
    The probe generates twice, but against the engine `resolve()` already built."""
    resolve_module = sys.modules["serving.resolve"]
    builds: list[str] = []
    seen: list[object] = []

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_every_backend(monkeypatch)
    monkeypatch.setattr(
        resolve_module,
        "build_engine",
        lambda base_model, *a, **kw: builds.append(base_model) or "vllm-engine",
    )
    monkeypatch.setattr(
        resolve_module,
        "_assert_adapter_applied",
        lambda engine, adapter, spec, config: seen.append(engine),
    )

    resolve(_wrapped_base_spec("sft_behaviour"))
    assert builds == ["Qwen/Qwen3.8-27B"]
    assert seen == ["vllm-engine"]


def test_the_probe_is_skipped_in_fake_mode() -> None:
    """`gate.sh` resolves every organism with no GPU (CLAUDE.md), and a probe that
    generated there would be scoring canned text against itself."""
    assert isinstance(resolve(_wrapped_base_spec("sft_behaviour")), FakeEndpoint)
    assert isinstance(resolve(_wrapped_base_spec("oct_demos")), FakeEndpoint)


@pytest.mark.parametrize("method", ["sft_behaviour", "oct_demos"])
def test_the_vllm_path_remaps_the_adapter_directory(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """`Qwen/Qwen3.8-27B` hides its text tower, so an adapter trained through
    `AutoModelForCausalLM` has to be renamed before vLLM will match a single module."""
    resolve_module = sys.modules["serving.resolve"]
    served: list[str] = []

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_every_backend(monkeypatch)
    monkeypatch.setattr(resolve_module, "_assert_adapter_applied", lambda *a, **kw: None)
    monkeypatch.setattr(
        resolve_module, "adapter_dir_for_vllm", lambda directory, base: f"{directory}/remapped"
    )
    monkeypatch.setattr(
        resolve_module, "adapter_for", lambda directory, name: served.append(directory)
    )

    resolve(_wrapped_base_spec(method))
    assert served == ["/nonexistent/remapped"]


@pytest.mark.parametrize("method", ["sft_behaviour", "oct_demos"])
def test_the_whitebox_path_remaps_the_adapter_directory_too(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Flipped by the vllm-lens migration (#359). White-box capture used to load the adapter
    with PEFT on the HF backend, whose flattened module names the adapter already carries,
    so it had to stay unremapped. It now serves through vLLM's own LoRA loader on vllm-lens
    (`_whitebox_adapter_endpoint`), which needs the same text-tower remap as the plain
    hot-swap — without it a 27B capture would read the bare base under the organism's name."""
    resolve_module = sys.modules["serving.resolve"]
    served: list[str] = []

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    _stub_every_backend(monkeypatch)
    monkeypatch.setattr(resolve_module, "_assert_adapter_applied", lambda *a, **kw: None)
    monkeypatch.setattr(
        resolve_module, "adapter_dir_for_vllm", lambda directory, base: f"{directory}/remapped"
    )
    monkeypatch.setattr(
        resolve_module, "adapter_for", lambda directory, name: served.append(directory)
    )

    endpoint = resolve(_wrapped_base_spec(method))
    assert endpoint.group == "lens"
    assert served == ["/nonexistent/remapped"]


def test_the_thinking_budget_reaches_the_endpoint_beside_the_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reasoning budget is an ORGANISM knob too, and the provider reads it off the
    endpoint (`evals/_inspect.EndpointModelAPI.generate`), so it has to survive resolve.

    Only when thinking is on: a thinking-off row that carries the key must still send the
    eval's cap unchanged, or the two arms are capped differently for no stated reason.
    """
    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.delenv("MO_EVALS_FAKE")
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    monkeypatch.setattr(
        resolve_module,
        "build_engine",
        lambda *a, **kw: types.SimpleNamespace(_tokenizer=_AcceptingTok()),
    )

    on = resolve(_thinking_spec(enable_thinking=True, thinking_budget=4096))
    default = resolve(_thinking_spec(enable_thinking=True))
    off = resolve(_thinking_spec(thinking_budget=4096))

    assert (on.enable_thinking, on.thinking_budget) == (True, 4096)
    assert (default.enable_thinking, default.thinking_budget) == (True, None)
    assert (off.enable_thinking, off.thinking_budget) == (False, None)


def test_the_serve_time_wrappers_forward_the_thinking_knobs() -> None:
    """`few_shot_icl` is served as `MessagesPrefixEndpoint(VLLMEndpoint(...))`, and the
    Inspect provider reads the knobs off the OUTER object. Without the forward a thinking-on
    few-shot organism would read as thinking-off and its completion — the inside of a think
    block the prompt opened — would be scored as the answer, which is the S1 defect of the
    2026-09-05 run one indirection further out. Absence still means off: `FakeEndpoint` has
    neither attribute and must not crash the getattr.
    """
    from serving.resolve import MessagesPrefixEndpoint, SystemPromptEndpoint

    inner = types.SimpleNamespace(enable_thinking=True, thinking_budget=2048)
    bare = types.SimpleNamespace()
    for wrapped, unwrapped in (
        (MessagesPrefixEndpoint(inner, []), MessagesPrefixEndpoint(bare, [])),
        (SystemPromptEndpoint(inner, "p"), SystemPromptEndpoint(bare, "p")),
    ):
        assert (wrapped.enable_thinking, wrapped.thinking_budget) == (True, 2048)
        assert (unwrapped.enable_thinking, unwrapped.thinking_budget) == (False, None)


# ------------------------------------------------ the context budget vs the think block


def test_a_full_context_thinking_request_still_fits_inside_max_model_len() -> None:
    """The guard's whole job is that no request overruns `max_model_len` — an over-length
    one fails its whole co-batched group. A thinking-on organism asks
    for `eval cap + thinking_budget`, so a budget computed off `sampling.max_tokens` alone
    let a full-window prompt overrun by the budget: 14848 + 768 + 2048 > 16384.
    """
    from serving.resolve import context_budget_tokens, serving_config

    config = serving_config()
    eval_cap = 768  # the judged evals' cap (configs/specs.yaml)
    for budget in (2048, 4096):
        spec = _thinking_spec(enable_thinking=True, thinking_budget=budget)
        assert context_budget_tokens(spec) + eval_cap + budget <= config.max_model_len
    # No `thinking_budget`: the provider asks for the remaining context, which cannot
    # overrun, but the longest prompt must still leave the reasoning a reservation — or the
    # top drift-long rung cuts a thinking organism off by our arithmetic, not its own.
    from serving.resolve import THINKING_CONTEXT_RESERVE

    default_spec = _thinking_spec(enable_thinking=True)
    assert config.max_model_len - context_budget_tokens(default_spec) >= (
        eval_cap + THINKING_CONTEXT_RESERVE
    )


def test_a_thinking_off_organism_keeps_the_whole_window() -> None:
    """Reserving the reasoning allowance for every organism would shrink the drift-long
    window and the ICL prefix cap on arms that never open a think block, which changes
    which items those cells run — a scaffolding change, not a serving one."""
    from serving.resolve import context_budget_tokens

    assert context_budget_tokens(_thinking_spec()) == context_budget_tokens()
    assert context_budget_tokens(_thinking_spec(thinking_budget=4096)) == context_budget_tokens()


def test_the_adapter_probe_renders_with_the_same_thinking_kwargs_as_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe's two arms must render exactly as the organism will be served, and the
    six other endpoint constructions get that from `_thinking_kwargs`. Building the pair
    from `_enable_thinking` by hand is how a future knob added to the pair gets missed.
    """
    from serving.resolve import serving_config

    resolve_module = sys.modules["serving.resolve"]
    calls: list[str] = []
    built: list[dict] = []

    monkeypatch.setattr(
        resolve_module,
        "_thinking_kwargs",
        lambda spec, tok: calls.append(spec.spec_hash) or {"enable_thinking": True, "sentinel": 1},
    )
    monkeypatch.setattr(
        resolve_module,
        "VLLMEndpoint",
        lambda engine, **kw: (
            built.append(kw) or types.SimpleNamespace(logprobs=lambda *a, **k: [0.0])
        ),
    )
    spec = _wrapped_base_spec("sft_behaviour")
    adapter = types.SimpleNamespace(uri="org/a")
    with pytest.raises(RuntimeError):  # both arms return one identical logprob: no shift
        resolve_module._assert_adapter_applied("engine", adapter, spec, serving_config())

    assert calls == [spec.spec_hash]  # once, for both arms
    assert [kw.get("sentinel") for kw in built] == [1, 1]


# ------------------------------------------------- the differs-from-base probe, generalised
#
# `_assert_adapter_applied` guarded one method ALONE until 2026-09-04, and the same
# failure then landed on the path it did not cover: vLLM loaded a LoRA for
# `Qwen/Qwen3.8-27B`, reported it attached with an `int_id`, and applied nothing --
# logprobs bit-identical to the base model, against PEFT's max |dlogit| 0.625 on the same
# adapter. Three organisms would have served base weights under their own spec_hashes.
#
# Several tests above no-op the probe because their subject is dispatch.
# These are what stops that from quietly disarming it everywhere.


class _StubAdapter:
    """Enough of a `LoRAAdapter` for the refusal message, which quotes `adapter.uri`."""

    uri = "/local/adapter"
    name = "stub"


def _probe_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Resolve an adapter organism for real, recording whether the probe ran."""
    resolve_module = sys.modules["serving.resolve"]
    seen: list[str] = []
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    _stub_artifact_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(resolve_module, "adapter_for", lambda directory, name: "lora-req")
    monkeypatch.setattr(resolve_module, "build_engine", lambda *a, **kw: "vllm-engine")
    monkeypatch.setattr(resolve_module, "VLLMEndpoint", lambda engine, **kw: "vllm-endpoint")
    monkeypatch.setattr(
        resolve_module,
        "_assert_adapter_applied",
        lambda engine, adapter, spec, config: seen.append(spec.method),
    )
    return seen


def test_the_adapter_probe_is_wired_into_every_adapter_method(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EVERY method sharing `_resolve_adapter`, not a sample of them.

    The bug this exists for was one of scope, not of logic: the check was correct and
    guarded one method out of eight. Enumerating from `ADAPTER_METHODS` rather than listing
    names here means a method added later cannot slip in unguarded.
    """
    from serving.resolve import ADAPTER_METHODS

    for method in sorted(ADAPTER_METHODS):
        seen = _probe_calls(monkeypatch, tmp_path)
        spec = dataclasses.replace(_adapter_spec(), method=method)
        assert resolve(spec) == "vllm-endpoint"
        assert seen == [method], f"{method} resolved without the differs-from-base probe"


def test_the_probe_refuses_an_adapter_the_engine_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-04 case exactly: identical logprobs, adapter attached, nothing applied.

    Pins that an ADAPTER-method spec reaches the shared helper and is refused, which is the
    wiring that was missing.
    """
    from serving.vllm_backend import ServingConfig

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(
        resolve_module.VLLMEndpoint, "logprobs", lambda self, m, **kw: [-1.0, -2.0]
    )
    with pytest.raises(RuntimeError, match="indistinguishable from the base model"):
        resolve_module._assert_adapter_applied(
            object(), _StubAdapter(), _adapter_spec(), ServingConfig()
        )


def test_the_refusal_names_the_method_rather_than_hardcoding_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message is the whole diagnostic on a pod at 2am; it must not lie about which
    method failed now that several of them share this guard."""
    from serving.vllm_backend import ServingConfig

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(
        resolve_module.VLLMEndpoint, "logprobs", lambda self, m, **kw: [-1.0, -2.0]
    )
    spec = dataclasses.replace(_adapter_spec(), method="dpo_behaviour")
    with pytest.raises(RuntimeError, match="dpo_behaviour"):
        resolve_module._assert_adapter_applied(object(), _StubAdapter(), spec, ServingConfig())


# ------------------------------------------------------------- merged-checkpoint serving
#
# For architectures whose vLLM wrapper ignores a LoRA entirely (2026-09-04, Qwen3.8-27B),
# the adapter is folded into the weights and vLLM serves an ordinary checkpoint. Merging is
# NOT the default -- `serves_merged`'s docstring has the three costs -- so these pin both
# which organisms take the route and that the route is honest about itself.


def test_no_architecture_routes_itself_to_merged_serving(tmp_path: Path) -> None:
    """Merging is OPT-IN ONLY. Not even a known-LoRA-hostile architecture turns it on.

    An earlier version auto-routed on architecture, which meant the choice of base model
    silently changed how every organism on it was served -- and merged rows are ~0.05 apart
    from LoRA rows on logits. Safe to leave off because merging is the REMEDY, not the
    protection: `_assert_adapter_applied` refuses an adapter that did not reach the engine
    on any architecture, so the default path fails loudly rather than serving base weights.
    """
    from serving.resolve import KNOWN_LORA_HOSTILE_ARCHITECTURES, base_architecture

    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "config.json").write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    fine = tmp_path / "fine"
    fine.mkdir()
    (fine / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')

    # The architecture is still READ -- the refusal message uses it to name a remedy.
    assert base_architecture(str(hostile)) in KNOWN_LORA_HOSTILE_ARCHITECTURES
    assert base_architecture(str(fine)) not in KNOWN_LORA_HOSTILE_ARCHITECTURES

    # ...but it routes nothing.
    assert not serves_merged(dataclasses.replace(_adapter_spec(), base_model=str(hostile)))
    assert not serves_merged(dataclasses.replace(_adapter_spec(), base_model=str(fine)))


def test_the_refusal_points_at_the_remedy_for_a_known_hostile_architecture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The message is the whole diagnostic. Someone hitting this has just watched a LoRA
    load, report attached, and do nothing; the error has to say what to do next."""
    from serving.vllm_backend import ServingConfig

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(
        resolve_module.VLLMEndpoint, "logprobs", lambda self, m, **kw: [-1.0, -2.0]
    )
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "config.json").write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    spec = dataclasses.replace(_adapter_spec(), base_model=str(hostile))

    with pytest.raises(RuntimeError, match="MO_EVALS_MERGE_ADAPTERS=1"):
        resolve_module._assert_adapter_applied(object(), _StubAdapter(), spec, ServingConfig())


def test_an_unreadable_config_takes_the_ordinary_lora_path(tmp_path: Path) -> None:
    """Guessing wrong here must cost a LOUD refusal, not a silent one.

    An unknown architecture routes to the LoRA path, where `_assert_adapter_applied`
    decides whether it actually worked -- so the failure mode is "refused at resolve",
    never "served base weights".
    """
    from serving.resolve import base_architecture

    assert base_architecture(str(tmp_path / "nope")) == ""
    assert not serves_merged(dataclasses.replace(_adapter_spec(), base_model=str(tmp_path)))


def test_the_merge_override_forces_the_route_both_ways(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`MO_EVALS_MERGE_ADAPTERS` has to force OFF as well as on: a box with no disk for a
    64GB checkpoint needs an escape hatch, and so does comparing the two arithmetics on a
    known-good architecture."""
    fine = tmp_path / "fine"
    fine.mkdir()
    (fine / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    spec = dataclasses.replace(_adapter_spec(), base_model=str(fine))

    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    assert serves_merged(spec)
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "0")
    assert not serves_merged(spec)
    monkeypatch.delenv("MO_EVALS_MERGE_ADAPTERS")
    assert not serves_merged(spec)


def test_a_merged_organism_is_served_with_no_adapter_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point: vLLM gets a plain checkpoint, so its LoRA mapping never runs.

    Also pins that `_assert_adapter_applied` is NOT called on this route -- it toggles an
    adapter against a shared engine and there is no adapter here, so calling it would
    compare the merged model against itself and pass vacuously.
    """
    resolve_module = sys.modules["serving.resolve"]
    seen: dict[str, object] = {}
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    _stub_artifact_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(resolve_module, "_merged_checkpoint", lambda *a: "/merged/ckpt")
    monkeypatch.setattr(
        resolve_module,
        "build_engine",
        lambda model, config, **kw: seen.setdefault("engine", (model, kw)),
    )

    def _endpoint(engine: object, **kw: object) -> str:
        seen["endpoint_kw"] = kw
        return "merged-endpoint"

    monkeypatch.setattr(resolve_module, "VLLMEndpoint", _endpoint)
    monkeypatch.setattr(
        resolve_module,
        "_assert_adapter_applied",
        lambda *a, **kw: seen.setdefault("probe_ran", True),
    )

    assert resolve(_adapter_spec()) == "merged-endpoint"
    assert seen["engine"] == ("/merged/ckpt", {"enable_lora": False})
    assert "adapter" not in seen["endpoint_kw"]
    assert "probe_ran" not in seen, "the adapter probe cannot run on a route with no adapter"


def test_a_merged_organism_still_gets_the_thinking_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Thinking is a `method_config` knob inside the spec_hash, not a property of the
    serving route, so folding the LoRA into the weights must not quietly drop it.

    It would fail silently if it did: the Inspect provider reads `enable_thinking` off the
    ENDPOINT, so a merged endpoint without it serves a thinking-ON organism with thinking
    off and scores the result under the thinking-on `spec_hash` — the S1 shape of the
    2026-09-05 run, one serving route further along.
    """
    resolve_module = sys.modules["serving.resolve"]
    seen: dict[str, object] = {}
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    _stub_artifact_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(resolve_module, "base_architecture", lambda base: "Qwen2ForCausalLM")
    monkeypatch.setattr(resolve_module, "_merged_checkpoint", lambda *a: "/merged/ckpt")
    monkeypatch.setattr(
        resolve_module,
        "build_engine",
        lambda *a, **kw: types.SimpleNamespace(_tokenizer=_AcceptingTok()),
    )
    monkeypatch.setattr(
        resolve_module, "VLLMEndpoint", lambda engine, **kw: seen.update(kw) or "merged-endpoint"
    )

    spec = _adapter_spec()
    thinking = dataclasses.replace(
        spec,
        method_config={**spec.method_config, "enable_thinking": True, "thinking_budget": 4096},
    )
    assert resolve(thinking) == "merged-endpoint"
    assert (seen["enable_thinking"], seen["thinking_budget"]) == (True, 4096)


@pytest.mark.parametrize("merged", [True, False])
def test_only_the_hot_swap_path_remaps_the_adapter_keys(
    monkeypatch: pytest.MonkeyPatch, merged: bool
) -> None:
    """The remap and the merge are two remedies for one bug, and they must not stack.

    `adapter_dir_for_vllm` renames an adapter's keys so *vLLM's* multimodal wrapper can
    match them. `_merged_checkpoint` never goes near that wrapper: it merges through PEFT
    against a flattened `AutoModelForCausalLM`, whose names the adapter already carries, so
    a remapped directory would make PEFT match nothing — caught by the merge's own
    before/after tensor check, but as a refusal on a path that had no defect.
    """
    resolve_module = sys.modules["serving.resolve"]
    remapped: list[str] = []
    merged_from: list[str] = []

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1" if merged else "0")
    _stub_every_backend(monkeypatch)
    monkeypatch.setattr(resolve_module, "base_architecture", lambda base: "Qwen2ForCausalLM")
    monkeypatch.setattr(
        resolve_module,
        "adapter_dir_for_vllm",
        lambda directory, base: remapped.append(directory) or f"{directory}/remapped",
    )
    monkeypatch.setattr(
        resolve_module,
        "_merged_checkpoint",
        lambda spec, adapter_dir, dtype: merged_from.append(adapter_dir) or "/merged/ckpt",
    )

    resolve(_wrapped_base_spec("sft_behaviour"))

    if merged:
        assert (remapped, merged_from) == ([], ["/nonexistent"])
    else:
        assert (remapped, merged_from) == (["/nonexistent"], [])


def _tiny_base(tmp_path: Path) -> str:
    """A real, tiny Qwen2 checkpoint on disk. No network, sub-second.

    A real graph rather than a mock because the
    failure this area produces is an intervention that loads without error and never
    applies, which only a real module graph can show.
    """
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    )
    out = tmp_path / "base"
    model.save_pretrained(str(out))
    return str(out)


def _adapter_on(base_dir: str, out: Path, *, zero_b: bool) -> str:
    """A PEFT adapter for that checkpoint. `zero_b` makes merging a mathematical no-op.

    Zeroed `lora_B` is the signature of "PEFT matched nothing": `merge_and_unload()` returns
    a model, every weight is unchanged, and without the guard a base-model checkpoint gets
    written under the organism's spec_hash.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_dir)
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["down_proj"]))
    if not zero_b:
        with torch.no_grad():
            for name, param in peft.named_parameters():
                if "lora_B" in name:
                    param.add_(torch.full_like(param, 0.05))
    peft.save_pretrained(str(out))
    return str(next(out.glob("**/adapter_model.safetensors")).parent)


def test_a_merge_that_changed_nothing_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`merge_and_unload()` returning a model is not evidence it changed one.

    This is the same class of failure as the vLLM bug one layer up -- an artifact that
    loads, reports success and applies nothing -- so it gets the same treatment: verify,
    do not assume. Without the check, a base-model checkpoint is written and served under
    this organism's spec_hash.
    """
    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", tmp_path / "cache")
    base = _tiny_base(tmp_path)
    adapter = _adapter_on(base, tmp_path / "inert", zero_b=True)
    spec = dataclasses.replace(_adapter_spec(), base_model=base)

    with pytest.raises(RuntimeError, match="bit-identical"):
        resolve_module._merged_checkpoint(spec, adapter, "float32")


def test_a_real_merge_is_written_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The happy path, and that the cache is a cache: the second call must not re-merge."""
    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", tmp_path / "cache")
    base = _tiny_base(tmp_path)
    adapter = _adapter_on(base, tmp_path / "real", zero_b=False)
    spec = dataclasses.replace(_adapter_spec(), base_model=base)

    out = resolve_module._merged_checkpoint(spec, adapter, "float32")
    assert (Path(out) / "config.json").exists()
    # The tokenizer has to travel with it or vLLM, which loads a DIRECTORY here, finds none
    # and the engine build fails after the expensive part.
    written = {p.name for p in Path(out).iterdir()}
    assert any("token" in n for n in written), f"no tokenizer beside the weights: {written}"

    calls: list[str] = []
    monkeypatch.setattr(
        resolve_module,
        "log",
        type("L", (), {"info": lambda *a: calls.append(a[1] if a else "")})(),
    )
    assert resolve_module._merged_checkpoint(spec, adapter, "float32") == out


def test_a_merged_organism_gets_its_own_backend_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two merged organisms must never share a group, and eviction is not the alternative.

    A backend group is the runner's unit of "these share a GPU engine". Merged organisms
    are their own base models (`_engine_key` starts with `base_model`), and within a group
    `eval_set` interleaves cells across organisms -- so sharing one would need both engines
    resident at once, and an LRU would evict an engine with requests in flight against it.
    Separate groups make them sequential, and `_release_group_engines()` frees the card
    between groups, which is the one place "nothing is in flight" holds.
    """
    from serving.resolve import spec_backend_group

    fine = tmp_path / "fine"
    fine.mkdir()
    (fine / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    a = dataclasses.replace(_adapter_spec(), base_model=str(fine), seed=1)
    b = dataclasses.replace(_adapter_spec(), base_model=str(fine), seed=2)

    monkeypatch.delenv("MO_EVALS_MERGE_ADAPTERS", raising=False)
    assert spec_backend_group(a) == spec_backend_group(b) == "vllm", "LoRA organisms share"

    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    assert spec_backend_group(a) != spec_backend_group(b)
    assert spec_backend_group(a).startswith("merged_")


def test_whitebox_merged_organisms_get_their_own_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under vllm-lens `_resolve_adapter`'s whitebox branch serves a merged organism as an
    ordinary checkpoint on its OWN engine (keyed by the merged path), exactly as the plain
    vLLM merged path does — so it needs its own group, not the shared `lens` one. The
    grouping must agree with dispatch, the defect `method_backend_group`'s docstring records."""
    from serving.resolve import spec_backend_group

    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    assert spec_backend_group(_adapter_spec()).startswith("merged_")


def test_the_merged_config_keeps_the_bases_declared_architecture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`save_pretrained` stamps the class it was CALLED on, not the one the base declares.

    We load through `AutoModelForCausalLM`, so a base declaring
    `Qwen3_5ForConditionalGeneration` comes back out declaring `Qwen3_5ForCausalLM`. The
    weights are unaffected -- both layouts write `model.language_model.layers.*` -- but
    vLLM honours the declared architecture, builds the CausalLM class whose inner model has
    `layers.*` directly, and refuses every key with "There is no module or parameter named
    'language_model'". Measured on an H100, 2026-09-04: the merge was correct and the
    engine would not load it, which is a merged organism that cannot serve at all.
    """
    import json as _json

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", tmp_path / "cache")
    base = _tiny_base(tmp_path)
    # Declare something other than what `save_pretrained` will write, exactly as the real
    # 27B does.
    cfg_path = Path(base) / "config.json"
    cfg = _json.loads(cfg_path.read_text())
    cfg["architectures"] = ["Qwen2ForConditionalGeneration"]
    cfg_path.write_text(_json.dumps(cfg))

    adapter = _adapter_on(base, tmp_path / "real2", zero_b=False)
    spec = dataclasses.replace(_adapter_spec(), base_model=base)
    out = resolve_module._merged_checkpoint(spec, adapter, "float32")

    written = _json.loads((Path(out) / "config.json").read_text())["architectures"]
    assert written == ["Qwen2ForConditionalGeneration"], (
        f"merged config declares {written}; vLLM would build the wrong class and refuse "
        "every weight key"
    )


def test_the_merged_checkpoint_carries_every_non_weight_file_from_the_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A merged checkpoint is the base with different weight VALUES, so it needs the
    base's whole non-weight directory -- and its OWN weights and weight index.

    Reconstructing that directory by naming the files it needs failed three times on
    2026-09-04, each on a file nobody had listed: the config's architecture, then the
    flattened text config, then `preprocessor_config.json` and
    `video_preprocessor_config.json` for a checkpoint that declares a multimodal
    architecture while being used text-only. So the rule under test is "everything except
    the weights", which cannot have that failure mode -- and the assertion is stated the
    same way, over whatever the base happens to hold, rather than over a list.

    Offline: the base is a local directory, which is also the path a pod takes once the
    files are cached.
    """
    import json as _json

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", tmp_path / "cache")
    base = Path(_tiny_base(tmp_path))
    # The companions a real multimodal-shaped checkpoint ships, including the two whose
    # absence stopped the engine starting.
    companions = {
        "preprocessor_config.json": '{"image_processor_type": "Qwen2VLImageProcessor"}',
        "video_preprocessor_config.json": '{"video_processor_type": "Qwen2VLVideoProcessor"}',
        "generation_config.json": '{"temperature": 0.7, "do_sample": true}',
        "chat_template.jinja": "{{ messages }}",
    }
    for name, body in companions.items():
        (base / name).write_text(body)

    adapter = _adapter_on(str(base), tmp_path / "comp", zero_b=False)
    spec = dataclasses.replace(_adapter_spec(), base_model=str(base))
    out = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))

    written = {p.name for p in out.iterdir() if p.is_file()}
    expected = {
        p.name
        for p in base.iterdir()
        if p.is_file() and not resolve_module._is_weight_file(p.name)
    }
    missing = expected - written
    assert not missing, f"the merged checkpoint is missing base files the engine needs: {missing}"

    # Its OWN weights, not the base's -- a merged model that shipped the base's weights
    # would be the base model wearing this organism's spec_hash.
    assert any(n.endswith(".safetensors") for n in written)
    assert _json.loads((out / "preprocessor_config.json").read_text())["image_processor_type"]


def test_the_merged_checkpoint_does_not_inherit_the_bases_weights_or_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one category that must NOT be copied. The merged model has its own shards and
    its own index describing them; the base's index names shards that are not there."""
    resolve_module = sys.modules["serving.resolve"]
    assert resolve_module._is_weight_file("model.safetensors")
    assert resolve_module._is_weight_file("model.safetensors.index.json")
    assert resolve_module._is_weight_file("pytorch_model.bin")
    assert not resolve_module._is_weight_file("config.json")
    assert not resolve_module._is_weight_file("preprocessor_config.json")

    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", tmp_path / "cache")
    base = Path(_tiny_base(tmp_path))
    (base / "model.safetensors.index.json").write_text(
        '{"weight_map": {"fake": "nope.safetensors"}}'
    )
    adapter = _adapter_on(str(base), tmp_path / "idx", zero_b=False)
    spec = dataclasses.replace(_adapter_spec(), base_model=str(base))
    out = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))

    stale = out / "model.safetensors.index.json"
    if stale.exists():
        assert "nope.safetensors" not in stale.read_text(), "the base's weight index was copied"


# ------------------------------------------------------ found in review, 2026-09-05


@pytest.mark.parametrize("value", ["FALSE", "no", "off", "0 ", "false", "0", "", "NO"])
def test_falsey_spellings_do_not_turn_merging_on(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A denylist read `FALSE`, `no`, `off` and `"0 "` as ON.

    Every one of those fails in the direction that silently changes the serving arithmetic
    (~0.05 on logits) and writes 15-64 GB per organism, under an unchanged spec_hash --
    someone typing `MO_EVALS_MERGE_ADAPTERS=off` to turn it OFF got the opposite. Parsed
    as an allowlist now, matching `fake_mode`, and pinned against both.
    """
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", value)
    monkeypatch.setenv("MO_EVALS_FAKE", value)
    assert serves_merged(_adapter_spec()) is False
    assert fake_mode() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_truthy_spellings_do_turn_merging_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", value)
    assert serves_merged(_adapter_spec()) is True


def test_a_merged_cache_hit_needs_the_weights_not_just_a_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`save_pretrained` writes config.json EARLY.

    A merge killed part-way, or a disk that filled during a ~55 GB write, leaves a valid
    config beside truncated or absent shards -- and that was returned as a cache hit
    forever. Nothing downstream would catch it: the merged route deliberately skips
    `_assert_adapter_applied`, because there is no adapter to toggle.
    """
    resolve_module = sys.modules["serving.resolve"]
    cache = tmp_path / "cache"
    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", cache)
    base = _tiny_base(tmp_path)
    adapter = _adapter_on(base, tmp_path / "cachechk", zero_b=False)
    spec = dataclasses.replace(_adapter_spec(), base_model=base)

    out = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))
    shards = sorted(out.glob("*.safetensors"))
    assert shards, "nothing to truncate"

    # A config with no payload is NOT a hit: re-merging must rewrite the shards.
    for shard in shards:
        shard.unlink()
    again = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))
    assert sorted(again.glob("*.safetensors")), "a config-only directory was served as a hit"

    # Nor is a config beside a zero-byte shard.
    for shard in sorted(again.glob("*.safetensors")):
        shard.write_bytes(b"")
    third = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))
    assert all(s.stat().st_size for s in third.glob("*.safetensors"))


def test_a_leftover_destination_does_not_silently_replace_the_fresh_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`os.replace` on a directory raises ENOTEMPTY when the destination exists.

    Two causes, one exception: losing a race (fine -- the winner used the same key) and a
    half-written directory from an earlier crash (not fine). Swallowing both would delete
    the ~55 GB checkpoint just built and leave one the engine cannot load, so a destination
    without a config re-raises.
    """
    resolve_module = sys.modules["serving.resolve"]
    cache = tmp_path / "cache"
    monkeypatch.setattr(resolve_module, "MERGED_CHECKPOINT_CACHE", cache)
    base = _tiny_base(tmp_path)
    adapter = _adapter_on(base, tmp_path / "race", zero_b=False)
    spec = dataclasses.replace(_adapter_spec(), base_model=base)

    # Pre-create a destination in the crashed-earlier state: present, non-empty, unusable.
    out = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))
    (out / "config.json").unlink()
    (out / "junk.txt").write_text("half a write")

    # SELF-HEALS rather than raising. The leftover has been proved unloadable and we hold a
    # freshly verified merge, so replacing it destroys nothing -- and the alternative is
    # making someone delete a ~55 GB path by hand mid-run.
    healed = Path(resolve_module._merged_checkpoint(spec, adapter, "float32"))
    assert resolve_module._merged_checkpoint_is_usable(healed)
    assert not (healed / "junk.txt").exists(), "the crashed directory was not replaced"
    # ...and no temp directory is left behind, on this path or the earlier ones.
    assert not [p for p in cache.iterdir() if p.name.startswith(".")], "temp dir leaked"
