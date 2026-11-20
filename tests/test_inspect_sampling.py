"""`complete_kwargs` sampling knobs reach the endpoint on the Inspect path (2026-09-04).

Before this, `target_max_tokens` carried only the cap and WARNED temperature / top_p
away, so every Inspect-path generation ran at the serving default (`temperature: 0.0`,
configs/serving.yaml) whatever the eval config said - and a seed sweep could not produce
sampling variance. The second test pins the other half: with no override the call
carries no sampling key at all, so the default path is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

from inspect_ai.model import ChatMessageUser, GenerateConfig

from core.endpoint import FakeEndpoint
from evals._inspect import EndpointModelAPI, bind_endpoint, target_sampling


class _Recorder(FakeEndpoint):
    """Records the kwargs of every `complete` (`FakeEndpoint.complete_full` delegates)."""

    def __init__(self) -> None:
        super().__init__(persona="caring")
        self.seen: list[dict[str, Any]] = []

    def complete(self, messages, **kw):
        self.seen.append(dict(kw))
        return super().complete(messages, **kw)


def _generate(key: str) -> None:
    asyncio.run(
        EndpointModelAPI(key).generate(
            [ChatMessageUser(content="hello")], [], "none", GenerateConfig()
        )
    )


def test_complete_kwargs_temperature_and_top_p_reach_the_endpoint():
    config = {"complete_kwargs": {"temperature": 0.7, "top_p": 0.9}}
    endpoint = _Recorder()
    with bind_endpoint(
        "hash-sampling", endpoint, seed=1, **target_sampling(config["complete_kwargs"])
    ):
        _generate("hash-sampling")
    (kw,) = endpoint.seen
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9


def test_no_override_leaves_the_serving_default_alone():
    endpoint = _Recorder()
    with bind_endpoint("hash-default", endpoint, seed=1, **target_sampling(None)):
        _generate("hash-default")
    (kw,) = endpoint.seen
    assert "temperature" not in kw
    assert "top_p" not in kw
