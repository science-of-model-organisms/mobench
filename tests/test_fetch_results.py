"""scripts/fetch_results.py upload discovery -- mocked repo listing, no network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import fetch_results as fr

ORG = "impulsive__prompting__scale1__abc"


class _Api:
    """Just the one HfApi call discovery makes."""

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def list_repo_files(self, repo: str, repo_type: str, revision: str | None = None) -> list[str]:
        assert repo_type == "dataset"
        return self.files


def _run(run: str) -> list[str]:
    return [
        f"runs/{run}/{ORG}/h1/spec.json",
        f"runs/{run}/{ORG}/h1/seed0__h1__mmlu.parquet",
        f"runs/{run}/logs/a.eval",
        f"runs/{run}/logs/header.json",
        f"runs/{run}/config.yaml",
    ]


def test_runs_layout_discovers_each_run_not_the_runs_prefix() -> None:
    """The layout run_all.sh writes. Used to return ['runs'], whose flatten dropped logs/."""
    files = _run("r1_partial") + _run("r1_final") + _run("r2") + ["README.md"]
    got = fr.list_uploads(_Api(files), "org/x")
    assert got == ["runs/r1_final", "runs/r1_partial", "runs/r2"]
    assert fr.select_uploads(got) == ["runs/r1_final", "runs/r2"]  # ladder still de-duplicated


def test_flat_layout_is_unchanged() -> None:
    files = [f"vllmA_final/{ORG}/h1/spec.json", f"hf2_partial/{ORG}/h2/spec.json", "README.md"]
    assert fr.list_uploads(_Api(files), "org/x") == ["hf2_partial", "vllmA_final"]


def test_a_logs_only_upload_is_found() -> None:
    """.eval-primary runs can ship no spec.json at all; their logs must not vanish."""
    files = ["runs/r1/logs/a.eval", "runs/r1/logs/b.eval", "logs/stray.eval", "analyses/x/y.json"]
    assert fr.list_uploads(_Api(files), "org/x") == ["runs/r1"]


def test_nothing_discoverable_is_empty() -> None:
    assert fr.list_uploads(_Api(["README.md", "RESTRICTED_FILES.json"]), "o/x") == []


def test_flatten_keeps_logs_for_a_nested_run(tmp_path: Path) -> None:
    """End to end on the discovered name: runs/<run>/logs/ lands in <out>/logs/."""
    staging, out = tmp_path / "staging", tmp_path / "out"
    run = staging / "runs" / "r1"
    (run / ORG / "h1").mkdir(parents=True)
    (run / ORG / "h1" / "spec.json").write_text("{}")
    (run / "logs").mkdir()
    (run / "logs" / "a.eval").write_text("log")
    fr.flatten(staging, out, ["runs/r1"])
    assert (out / "logs" / "a.eval").read_text() == "log"
    assert (out / ORG / "h1" / "spec.json").is_file()
    assert fr.summarise(out)["eval_logs"] == 1


def test_index_default_does_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert fr.DEFAULT_INDEX.is_absolute() and fr.DEFAULT_INDEX.is_file()
    repo, _ = fr.resolve_run(next(iter(_runs_in_index())), fr.DEFAULT_INDEX)
    assert "/" in repo


def _runs_in_index() -> list[str]:
    import yaml

    return list((yaml.safe_load(fr.DEFAULT_INDEX.read_text()) or {}).get("runs") or {})


def _put(root: Path, rel: str, text: str) -> None:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)


def test_auto_discovery_skips_superseded_archive_rejudges() -> None:
    ups = ["runs/r1", "superseded/old/runs/r1", "archive/a", "rejudges/x", "vllmA_final"]
    assert fr.default_uploads(ups) == ["runs/r1", "vllmA_final"]
    assert fr.select_uploads(ups) != fr.select_uploads(fr.default_uploads(ups))  # explicit keeps


def test_fetch_default_excludes_superseded_but_explicit_uploads_may_pick_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _run("r1") + [f"superseded/old/runs/r1/{ORG}/h1/spec.json"]
    seen: list[list[str]] = []
    monkeypatch.setattr(fr, "snapshot_download", lambda *a, **k: seen.append(k["allow_patterns"]))
    monkeypatch.setattr(fr, "overlay_restricted", lambda *a, **k: False)
    monkeypatch.setattr(fr, "flatten", lambda *a, **k: None)
    fr.fetch("org/x", tmp_path / "a", api=_Api(files))
    fr.fetch("org/x", tmp_path / "b", uploads=["superseded/old/runs/r1"], api=_Api(files))
    assert seen == [["runs/r1/**"], ["superseded/old/runs/r1/**"]]


def test_flatten_refuses_a_silent_overwrite(tmp_path: Path) -> None:
    staging, out = tmp_path / "staging", tmp_path / "out"
    for up, text in (("runs/r1", "current"), ("runs/r2", "rerun")):
        _put(staging, f"{up}/{ORG}/h1/spec.json", "{}")
        _put(staging, f"{up}/{ORG}/h1/mmlu.parquet", text)
        _put(staging, f"{up}/logs/a.eval", text)
    with pytest.raises(SystemExit, match="different bytes"):
        fr.flatten(staging, out, ["runs/r1", "runs/r2"])
    # identical bytes are not a conflict; the explicit choice restores last-wins
    fr.flatten(staging, tmp_path / "o2", ["runs/r1", "runs/r2"], allow_overwrite=True)
    assert (tmp_path / "o2" / ORG / "h1" / "mmlu.parquet").read_text() == "rerun"
    _put(staging, f"runs/r2/{ORG}/h1/mmlu.parquet", "current")
    _put(staging, "runs/r2/logs/a.eval", "current")
    fr.flatten(staging, tmp_path / "o3", ["runs/r1", "runs/r2"])
    assert (tmp_path / "o3" / "logs" / "a.eval").read_text() == "current"


def test_flatten_does_not_copy_a_nested_legacy_results_tree_twice(tmp_path: Path) -> None:
    """runs/<r>/results/<org>/... beside runs/<r>/logs: both are uploads; copy the tree once."""
    staging, out = tmp_path / "staging", tmp_path / "out"
    _put(staging, f"runs/r2/results/{ORG}/h2/spec.json", "{}")
    _put(staging, "runs/r2/logs/c.eval", "log")
    fr.flatten(staging, out, ["runs/r2", "runs/r2/results"])
    assert (out / ORG / "h2" / "spec.json").is_file()
    assert not (out / "results").exists()
    assert (out / "logs" / "c.eval").read_text() == "log"


def test_legacy_results_beside_logs_is_one_upload_and_ladder_ranks() -> None:
    files = []
    for up in ("backend_partial", "backend_final"):
        files += [f"{up}/results/{ORG}/h1/spec.json", f"{up}/logs/a.eval"]
    files += [f"top/results/{ORG}/h1/spec.json", f"results/{ORG}/h1/spec.json"]
    got = fr.list_uploads(_Api(files), "org/x")
    assert got == ["backend_final", "backend_partial", "results", "top"]
    assert fr.select_uploads(got) == ["backend_final", "results", "top"]


def test_refused_collision_writes_nothing(tmp_path: Path) -> None:
    staging, out = tmp_path / "staging", tmp_path / "out"
    for up, text in (("a", "x"), ("b", "y")):
        _put(staging, f"{up}/{ORG}/h1/spec.json", "{}")
        _put(staging, f"{up}/{ORG}/h1/cell.parquet", text)
    _put(staging, f"b/{ORG}/h1/other.parquet", "only-b")
    with pytest.raises(SystemExit, match="nothing was written"):
        fr.flatten(staging, out, ["a", "b"])
    assert not out.exists() or not any(out.rglob("*"))
    fr.flatten(staging, out, ["a"])  # the recovery the error recommends is clean
    assert sorted(p.name for p in out.rglob("*.parquet")) == ["cell.parquet"]


def test_existing_output_is_not_silently_overwritten(tmp_path: Path) -> None:
    staging, out = tmp_path / "staging", tmp_path / "out"
    _put(staging, f"u/{ORG}/h1/spec.json", "{}")
    _put(staging, f"u/{ORG}/h1/cell.parquet", "new")
    _put(out, f"{ORG}/h1/cell.parquet", "old")
    with pytest.raises(SystemExit, match="already in"):
        fr.flatten(staging, out, ["u"])
    assert (out / ORG / "h1" / "cell.parquet").read_text() == "old"
    fr.flatten(staging, out, ["u"], allow_overwrite=True)
    assert (out / ORG / "h1" / "cell.parquet").read_text() == "new"


def test_symlink_alias_destination_is_caught(tmp_path: Path) -> None:
    staging, out = tmp_path / "staging", tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "b.eval").symlink_to("a.eval")
    _put(staging, "one/logs/a.eval", "first")
    _put(staging, "two/logs/b.eval", "second")
    with pytest.raises(SystemExit, match="different bytes"):
        fr.flatten(staging, out, ["one", "two"])
    assert not (out / "logs" / "a.eval").exists()


def test_only_rate_limits_server_errors_and_dropped_connections_are_retried() -> None:
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    def http(code: int) -> HfHubHTTPError:
        resp = httpx.Response(code, request=httpx.Request("GET", "https://huggingface.co/x"))
        return HfHubHTTPError(f"{code}", response=resp)

    assert fr._retryable(http(429)) and fr._retryable(http(503))
    assert not fr._retryable(http(401)) and not fr._retryable(http(404))
    assert fr._retryable(httpx.ConnectError("dropped")) and fr._retryable(TimeoutError())
    assert not fr._retryable(FileNotFoundError("local")) and not fr._retryable(PermissionError())


@pytest.mark.parametrize("status", [429, 503, None])
def test_pull_retries_a_real_hub_rate_limit_server_error_or_refused_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int | None
) -> None:
    """Real `snapshot_download` against a local stub Hub: huggingface_hub wraps these failures
    in `LocalEntryNotFoundError` (a FileNotFoundError), and they must still be retried."""
    import http.server
    import socket
    import threading

    from huggingface_hub import constants

    hits: list[str] = []

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(status or 500)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_HEAD = do_GET

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the base name)
            pass

    if status is None:  # a port nothing listens on: the connection is refused
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    else:
        srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(constants, "ENDPOINT", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(fr.time, "sleep", sleeps.append)
    with pytest.raises(Exception) as info:  # noqa: PT011 (the hub's own type varies)
        fr.pull("someorg/x", tmp_path / "dl", ["*"], revision="a" * 40, tries=3)
    assert fr._retryable(info.value), type(info.value)
    assert sleeps == [30, 60]  # two retries, then the error
    if status is not None:
        assert hits  # the stub Hub was really asked


def test_pull_does_not_retry_a_local_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(fr.time, "sleep", sleeps.append)

    def full_disk(*a: object, **k: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fr, "snapshot_download", full_disk)
    with pytest.raises(OSError, match="No space"):
        fr.pull("o/x", tmp_path, ["*"])
    assert sleeps == []


def test_permanent_transport_errors_are_not_retried_but_a_503_in_upload_discovery_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    import requests

    xet = "Network error: Request error: HTTP status {} (xet-read-token)"
    assert fr._retryable(RuntimeError(xet.format("client error (429 Too Many Requests)")))
    assert fr._retryable(RuntimeError(xet.format("server error (503 Service Unavailable)")))
    assert not fr._retryable(RuntimeError(xet.format("client error (404 Not Found)")))
    assert not fr._retryable(httpx.UnsupportedProtocol("Unsupported protocol ftp://"))
    assert not fr._retryable(requests.exceptions.SSLError("certificate verify failed"))
    assert fr._retryable(requests.ConnectionError("reset")) and fr._retryable(
        httpx.ReadTimeout("x")
    )

    class Flaky(_Api):
        calls = 0

        def list_repo_files(self, *a: Any, **k: Any) -> list[str]:
            self.calls += 1
            if self.calls == 1:
                req = httpx.Request("GET", "https://huggingface.co/x")
                raise httpx.HTTPStatusError("503", request=req, response=httpx.Response(503))
            return super().list_repo_files(*a, **k)

    sleeps: list[float] = []
    monkeypatch.setattr(fr.time, "sleep", sleeps.append)
    api: Any = Flaky(_run("r1"))
    assert fr.list_uploads(api, "o/x", "a" * 40) == ["runs/r1"]
    assert (api.calls, sleeps) == (2, [30])


def test_a_certificate_failure_wrapped_by_httpx_is_final_after_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx reports a failed TLS handshake as a ConnectError (a transient-looking transport error)
    whose cause is the SSL error: the permanent cause in the chain wins."""
    import ssl

    import httpx

    sleeps: list[float] = []
    monkeypatch.setattr(fr.time, "sleep", sleeps.append)
    cert = ssl.SSLCertVerificationError("certificate verify failed")
    wrapped = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")
    wrapped.__cause__ = cert
    attempts = 0

    def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise wrapped

    with pytest.raises(httpx.ConnectError):
        fr.retrying(fail, "x")
    assert attempts == 1 and sleeps == []
    assert not fr._retryable(wrapped)
    # the other way round, and a cycle, still end
    reset = httpx.ConnectError("reset")
    reset.__cause__ = ConnectionResetError()
    assert fr._retryable(reset)
    reset.__cause__.__context__ = reset
    assert fr._retryable(reset)
