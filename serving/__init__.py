"""Serving: turn a spec into an `Endpoint`, and run evals over a config of specs."""

from serving.resolve import SystemPromptEndpoint, fake_mode, resolve

__all__ = ["SystemPromptEndpoint", "fake_mode", "resolve"]
