"""`python -m implant.few_shot --teacher luna`: a hard request cap, a token bound, the worst case
printed before any call, and a clear exit when too few candidates pass.

No network: `requests.Session.post` is replaced. Nothing here writes data/personas/ (the one
test that reaches the writing code stubs `build_luna_demos`)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import implant.few_shot as fs


@pytest.fixture(autouse=True)
def _real_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    monkeypatch.setattr(fs, "RAMP_SECONDS", 0.0)


class _Posts:
    """Replaces `requests.Session.post`: counts attempts and answers with `status`."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.n = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Posts:
        monkeypatch.setattr("requests.Session.post", lambda _s, url, **kw: self(url, **kw))
        return self

    def __call__(self, url: str, **_kw: Any) -> Any:
        self.n += 1
        payload = {
            "choices": [{"message": {"content": "an answer"}, "finish_reason": "stop"}],
            "usage": None,
        }
        return SimpleNamespace(
            status_code=self.status, reason="x", headers={}, text="", json=lambda: payload
        )


def _backend(budget: fs.CallBudget | None, **params: Any) -> Any:
    return fs._http_backend({"model": "m", "params": params}, budget)


def test_retries_spend_the_request_cap_and_the_cap_is_hard(monkeypatch) -> None:
    """A failing request is retried by the backend (4 attempts); each attempt is one request
    against `--max-calls`, so the run stops at the cap instead of at the retry loop's end."""
    import judges.backends as backends

    monkeypatch.setattr(backends.time, "sleep", lambda _s: None)
    posts = _Posts(status=503).install(monkeypatch)
    budget = fs.CallBudget(3)
    with pytest.raises(fs.CallCapExceeded, match="--max-calls 3 requests sent"):
        _backend(budget).complete([{"role": "user", "content": "hi"}])
    assert posts.n == 3 == budget.used  # the 4th attempt was refused before it was sent
    with pytest.raises(SystemExit):  # and it stays refused
        _backend(budget).complete([{"role": "user", "content": "hi"}])
    assert posts.n == 3


def test_the_cap_is_shared_by_teacher_and_judge_across_threads(monkeypatch) -> None:
    posts = _Posts().install(monkeypatch)
    budget = fs.CallBudget(7)
    teacher, judge = _backend(budget), _backend(budget)
    questions = list(range(20))

    def one(i: int) -> str:
        return (teacher if i % 2 else judge).complete([{"role": "user", "content": str(i)}])

    with pytest.raises(fs.CallCapExceeded):
        fs._ramped_map(one, questions)
    assert posts.n == budget.used == 7


def test_no_budget_leaves_the_backend_uncounted(monkeypatch) -> None:
    posts = _Posts().install(monkeypatch)
    backend = _backend(None)
    assert backend.complete([{"role": "user", "content": "hi"}]) == "an answer"
    assert posts.n == 1 and type(backend._session).__name__ == "Session"


def test_luna_block_is_uncapped_by_default_and_capped_on_request() -> None:
    from serving.runner import load_config

    cfg = load_config("configs/specs.yaml")
    assert "max_tokens" not in fs.luna_teacher_block(1, config=cfg)["params"]
    block = fs.luna_teacher_block(1, config=cfg, max_tokens=1234)
    assert block["params"]["max_tokens"] == 1234 and block["params"]["seed"] == 1
    assert "max_tokens" not in cfg["eval_configs"][fs.TEACHER_SOURCE_EVAL]["adversary"]["params"]


def test_a_cap_reaches_the_request_body(monkeypatch) -> None:
    from serving.runner import load_config

    sent: list[dict[str, Any]] = []

    def post(_s: Any, url: str, json: dict[str, Any], **_kw: Any) -> Any:
        sent.append(json)
        return _Posts()(url)

    monkeypatch.setattr("requests.Session.post", post)
    block = fs.luna_teacher_block(0, config=load_config("configs/specs.yaml"), max_tokens=777)
    fs._http_backend(block).complete([{"role": "user", "content": "hi"}])
    assert sent[0]["max_tokens"] == 777


def test_dry_run_prints_the_worst_case_and_makes_no_call(monkeypatch, capsys) -> None:
    posts = _Posts().install(monkeypatch)
    n = len(fs.shot_questions(fs.load_persona("impulsive")))
    fs.main(["--persona", "impulsive", "--teacher", "luna", "--dry-run", "--max-calls", "100"])
    out = capsys.readouterr().out
    assert posts.n == 0 and "dry run: no call made" in out
    assert f"{n} candidates, keep k=15; hard cap --max-calls 100 requests" in out
    assert "worst case teacher" in out and f"{n} x 4 billed attempts = {n * 4} requests" in out
    assert f"max_tokens {fs.LUNA_MAX_TOKENS}" in out and "worst-case output tokens" in out
    assert "demo judge" in out


def test_a_cap_below_the_minimum_or_k_above_the_pool_is_refused_before_any_call(
    monkeypatch,
) -> None:
    posts = _Posts().install(monkeypatch)
    n = len(fs.shot_questions(fs.load_persona("impulsive")))
    with pytest.raises(SystemExit, match=rf"need at least {2 * n} requests"):
        fs.main(["--persona", "impulsive", "--teacher", "luna", "--max-calls", str(2 * n - 1)])
    with pytest.raises(SystemExit, match=rf"--k {n + 1} exceeds the {n} candidates"):
        fs.main(["--persona", "impulsive", "--teacher", "luna", "--k", str(n + 1)])
    assert posts.n == 0


def test_too_few_eligible_candidates_exit_clearly_and_write_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    """The pilot's 4 of 40 at k=15 was a bare ValueError traceback after a paid pass."""
    _Posts().install(monkeypatch)

    def short(*_a: Any, **_kw: Any) -> Any:
        raise fs.TooFewEligible(4, 40, 15, fs.SELECT_THRESHOLD)

    monkeypatch.setattr(fs, "build_luna_demos", short)
    with pytest.raises(SystemExit) as exc:
        fs.main(["--persona", "impulsive", "--teacher", "luna"])
    msg = str(exc.value)
    assert "4 of 40 candidates passed" in msg and "k=15 needs 15" in msg
    assert "--k 4 or lower" in msg and "more seed questions" in msg
    assert "No demos file was written" in msg
    assert not isinstance(exc.value, fs.CallCapExceeded)


def test_too_few_eligible_is_still_a_value_error_naming_k() -> None:
    row = {
        "messages": [{"content": "q"}, {"content": "a"}],
        "meta": {"score": 0.1, "finish_reason": "stop", "pool_position": 1},
    }
    with pytest.raises(ValueError, match="only 0 of 1 candidates are eligible.*need k=2"):
        fs.select_demos([row], 2, banned=set())
    try:
        fs.select_demos([row], 2, banned=set())
    except fs.TooFewEligible as e:
        assert (e.eligible, e.rows, e.k) == (0, 1, 2)


@pytest.mark.parametrize("persona", ["impulsive", "loyalty"])
def test_the_committed_demos_and_their_hashes_are_untouched(persona: str) -> None:
    """Nothing here edits a demos file; the spec still binds the committed bytes."""
    path = fs.icl_demos_path(persona)
    spec = fs.build_few_shot_spec(persona, 15)
    assert spec.method_config["demos_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_real_run_still_refuses_fake_mode_before_building_any_backend(monkeypatch) -> None:
    """tests/test_few_shot_author_parity.py's contract, with the new flags: the refusal comes
    first, and only --dry-run (which makes no call) may run under MO_EVALS_FAKE."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setattr(fs, "_http_backend", lambda *a, **k: pytest.fail("backend built"))
    with pytest.raises(SystemExit, match="no fake mode"):
        fs.main(["--persona", "impulsive", "--teacher", "luna", "--max-calls", "200"])
    monkeypatch.undo()
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    posts = _Posts().install(monkeypatch)
    fs.main(["--persona", "impulsive", "--teacher", "luna", "--dry-run"])
    assert posts.n == 0
