"""The dashboard's HF-publish path: naming is a pure function of the run, and a
fake-mode blob is refused before anything touches the network (CLAUDE.md)."""

from __future__ import annotations

import pytest

from analysis.dashboard import publish, scorecard_name


def _data(*, fake: bool = False, n: int = 29, nudge: float = 0.5) -> dict:
    """Minimal dashboard blob — only the fields the publish path reads."""
    return {
        "meta": {
            "fake": fake,
            "n_organisms": n,
            "personas": ["impulsive"],
            "base_models": [
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen2.5-32B-Instruct",
            ],
        },
        "cells": {"o": {"eval": {"mean": nudge}}},
        "radar": {"min": 0.0, "max": 10.0, "orgs": {}, "evals": {}},
    }


def test_name_is_a_pure_function_of_the_data() -> None:
    # Same (data, run_name) in, same name out — this is what lets a re-render overwrite
    # instead of pile up.
    assert scorecard_name(_data()) == scorecard_name(_data())
    assert scorecard_name(_data(), "impulsive-qwen_2_5-14b-32b-20260908") == scorecard_name(
        _data(), "impulsive-qwen_2_5-14b-32b-20260908"
    )


def test_name_uses_run_name_verbatim_when_given() -> None:
    name = scorecard_name(_data(), "impulsive-qwen_2_5-14b-32b-20260908")
    assert name == "impulsive/impulsive-qwen_2_5-14b-32b-20260908.html"


def test_name_falls_back_without_a_run_name() -> None:
    assert scorecard_name(_data(), None) == scorecard_name(_data())
    assert scorecard_name(_data(), "") == scorecard_name(_data())


def test_name_has_no_content_hash_any_more() -> None:
    # Stable per run: the old `__<8-hex>.html` identity is gone.
    import re

    assert re.search(r"__[0-9a-f]{8}\.html$", scorecard_name(_data())) is None


def test_name_is_human_readable_and_data_derived() -> None:
    # No run_name: falls back to <persona>_<model_part>.html — no n_organisms, no hash.
    name = scorecard_name(_data())
    assert name.startswith("impulsive/")
    assert "qwen2.5-instruct" in name  # shared model family, org prefix dropped
    assert "7b-14b-32b" in name  # sizes parsed from base_models, sorted
    assert name.endswith(".html")


def test_a_single_model_keeps_its_full_name() -> None:
    data = _data()
    data["meta"]["base_models"] = ["Qwen/Qwen2.5-7B-Instruct"]
    assert "qwen2.5-7b-instruct" in scorecard_name(data)


def test_the_same_run_name_overwrites_regardless_of_content() -> None:
    # Naming is stable per results-run, not per content — a re-render with the same
    # run_name must produce the SAME file so HF's git overwrites in place (CLAUDE.md).
    run = "impulsive-qwen_2_5-14b-32b-20260908"
    assert scorecard_name(_data(nudge=0.5), run) == scorecard_name(_data(nudge=0.9), run)


def test_different_run_names_give_different_files() -> None:
    assert scorecard_name(_data(), "run-a") != scorecard_name(_data(), "run-b")


def test_size_parse_ignores_an_org_prefix_containing_b_digit() -> None:
    # Codex #9: the size must come from the model slug, not the whole HF id — an org name
    # like "Acme7B" must not be mistaken for the model size.
    data = _data()
    data["meta"]["base_models"] = ["Acme7B/Qwen2.5-14B-Instruct"]
    name = scorecard_name(data)
    assert "14b" in name
    assert "7b" not in name


def test_publish_refuses_a_fake_blob_before_the_network() -> None:
    # The guard must fire before the HfApi import/call, so this needs no token or monkeypatch.
    with pytest.raises(SystemExit, match="fake"):
        publish("<html></html>", _data(fake=True), "science-of-model-organisms/scorecards")


class _FakeApi:
    """Records what publish() would have sent to HF, without any network."""

    calls: dict = {}

    def __init__(self, token: str | None = None) -> None:
        _FakeApi.calls["token"] = token

    def create_repo(self, *a: object, **kw: object) -> None:
        _FakeApi.calls["create_repo"] = kw

    def update_repo_settings(self, *a: object, **kw: object) -> None:
        _FakeApi.calls["update_repo_settings"] = kw

    def upload_file(
        self,
        *,
        path_or_fileobj: bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> None:
        _FakeApi.calls["path_in_repo"] = path_in_repo
        _FakeApi.calls["html"] = path_or_fileobj.decode()


def test_publish_injects_a_provenance_banner_and_uses_the_env_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_name = "impulsive-qwen_2_5-14b-32b-20260908"
    results_repo = "science-of-model-organisms/impulsive-qwen_2_5-14b-32b-big_eval_results"
    monkeypatch.setenv("MO_EVALS_HF_RUN_NAME", run_name)
    monkeypatch.setenv("MO_EVALS_HF_RESULTS_REPO", results_repo)
    monkeypatch.setattr("scripts.push_banks_to_hub.load_token", lambda: "tok")
    _FakeApi.calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)

    html_in = '<html><body class="x"><p>hi</p></body></html>'
    url = publish(html_in, _data(), "science-of-model-organisms/scorecards")

    # Stable name (Change 1), no hash.
    assert _FakeApi.calls["path_in_repo"] == f"impulsive/{run_name}.html"
    assert url.endswith(f"impulsive/{run_name}.html")
    uploaded_html = _FakeApi.calls["html"]
    assert f"Generated from {results_repo}/runs/{run_name} at" in uploaded_html
    assert "content " in uploaded_html
    # Banner lands right after the opening <body ...> tag.
    body_start = uploaded_html.index("<body")
    body_end = body_start + uploaded_html[body_start:].index(">") + 1
    assert uploaded_html[body_end:].startswith("<div")
    assert "Generated from" in uploaded_html[body_end : body_end + 300]


def test_publish_banner_falls_back_gracefully_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_HF_RUN_NAME", raising=False)
    monkeypatch.delenv("MO_EVALS_HF_RESULTS_REPO", raising=False)
    monkeypatch.setattr("scripts.push_banks_to_hub.load_token", lambda: "tok")
    _FakeApi.calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)

    publish(
        "<html><body><p>hi</p></body></html>", _data(), "science-of-model-organisms/scorecards"
    )
    assert "local render" in _FakeApi.calls["html"]


def test_publish_prepends_the_banner_when_there_is_no_body_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MO_EVALS_HF_RUN_NAME", raising=False)
    monkeypatch.delenv("MO_EVALS_HF_RESULTS_REPO", raising=False)
    monkeypatch.setattr("scripts.push_banks_to_hub.load_token", lambda: "tok")
    _FakeApi.calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)

    publish("<p>no body here</p>", _data(), "science-of-model-organisms/scorecards")
    assert _FakeApi.calls["html"].startswith("<div")


def test_publish_enforces_privacy_on_a_pre_existing_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    # Codex #10: create_repo(..., private=True) is a no-op on a repo that already exists
    # and was public, so publish must also flip it with update_repo_settings.
    monkeypatch.setattr("scripts.push_banks_to_hub.load_token", lambda: "tok")
    _FakeApi.calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)

    publish("<html><body></body></html>", _data(), "science-of-model-organisms/scorecards")
    assert _FakeApi.calls["update_repo_settings"]["private"] is True


def test_publish_refuses_data_without_the_paper_radar_before_the_network() -> None:
    data = _data()
    del data["radar"]
    with pytest.raises(SystemExit, match="allow-d-radars"):
        publish("<html></html>", data, "science-of-model-organisms/scorecards")


def test_publish_accepts_data_without_the_paper_radar_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.push_banks_to_hub.load_token", lambda: "tok")
    _FakeApi.calls = {}
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    data = _data()
    del data["radar"]
    publish("<html><body></body></html>", data, "x/y", allow_d_radars=True)
    assert "path_in_repo" in _FakeApi.calls


def test_site_render_refuses_data_without_the_paper_radar_unless_allowed() -> None:
    from analysis.dashboard import _SITE_TEMPLATE_PATCHES, site_dashboard_html

    html = "<body>" + "".join(old for old, _ in _SITE_TEMPLATE_PATCHES) + "</body>"
    with pytest.raises(SystemExit, match="allow-d-radars"):
        site_dashboard_html(html, {"_build": {}})
    assert "nerfies" in site_dashboard_html(html, {"_build": {}}, allow_d_radars=True)
    assert "nerfies" in site_dashboard_html(html, {"_build": {}, "radar": {"orgs": {"o": {}}}})


@pytest.mark.parametrize("mode", ["--site-json", "--publish"])
def test_cli_refuses_before_writing_anything(mode, tmp_path, monkeypatch) -> None:
    """A site/publish build without paper-scale radars writes no summary JSON and no HTML
    (Codex, PR #458: both used to be written before the refusal)."""
    import json
    import sys

    from analysis import dashboard

    data = _data()
    del data["radar"]
    blob = tmp_path / "blob.json"
    blob.write_text(json.dumps(data), encoding="utf-8")
    out, site = tmp_path / "page.html", tmp_path / "site.json"
    argv = ["dashboard", "--data-json", str(blob), "--out", str(out)]
    argv += ["--site-json", str(site)] if mode == "--site-json" else ["--publish"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="allow-d-radars"):
        dashboard.main()
    assert not out.exists() and not site.exists()
