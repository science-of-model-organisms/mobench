"""`serving.inspect_runner._cache_inspect_git_context`: git metadata once per process.

Without it the fake sweep spends most of its wall clock running `git status` once per cell
on the FUSE volume (see the helper's docstring). These pin that the patch reaches the name
Inspect actually calls, so an `inspect_ai` bump that moves it fails here instead of the gate
quietly getting several minutes slower.
"""

from __future__ import annotations

import functools

import pytest
from inspect_ai._eval.task import log as task_log
from inspect_ai._util import git

from serving.inspect_runner import _cache_inspect_git_context


@pytest.fixture(autouse=True)
def _restore_git_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from (and leaves) Inspect's own uncached function."""
    monkeypatch.setattr(task_log, "git_context", git.git_context)


def test_the_patch_replaces_the_name_resolve_revision_calls() -> None:
    _cache_inspect_git_context()
    assert hasattr(task_log.git_context, "cache_info"), "Inspect moved git_context"
    assert task_log.git_context.__wrapped__ is git.git_context


def test_the_patch_is_idempotent() -> None:
    _cache_inspect_git_context()
    first = task_log.git_context
    _cache_inspect_git_context()
    assert task_log.git_context is first


def test_resolve_revision_still_returns_the_uncached_answer(monkeypatch) -> None:
    """Same value, looked up once: the cache changes cost, not content."""
    calls = []

    def fake() -> git.GitContext:
        calls.append(1)
        return git.GitContext(origin="o", commit="abc1234", dirty=False)

    monkeypatch.setattr(task_log, "git_context", functools.cache(fake))
    a, b = task_log.resolve_revision(), task_log.resolve_revision()
    assert a == b and a is not None and a.commit == "abc1234"
    assert len(calls) == 1
