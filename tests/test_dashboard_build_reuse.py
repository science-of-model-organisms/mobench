"""A stranger's own dashboard build: their HF repo, behaviour, base model and methods, no paper.

`analysis.dashboard_build` end to end (fetch -> tree -> rows -> reduce -> data -> html) on a
mocked HF dataset that shares nothing with this project's runs: another org's repo, a made-up
behaviour (`zesty`) and base model, three implant methods, no decisiveness sidecar, no
`families:`/`require:`/`paper:`. The config is the only input; no code is edited. The download
is mocked (`scripts.fetch_results.snapshot_download`), so no network.

The results are not fake-mode trees: the build refuses a FAKE row (CLAUDE.md rule 5), so the
fixture writes `fake: false` manifests with scores set here. They are fixture numbers, not
model output.
"""

from __future__ import annotations

import fnmatch
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ID = "someorg/zesty-results"
BASE = "acme/Tiny-1B-Instruct"
SHA = "0" * 40
# (eval, axis): one or two evals for every axis a stranger's run is likely to have
EVALS = (
    ("generality_ood", "generality"),
    ("robustness_open_ended_gen", "robustness"),
    ("response_coherence", "coherence"),
    ("mmlu", "capability"),
    ("introspection_mcq", "introspection"),
)
# method -> (method_config, how far its generality score sits above base)
# (the repo's own method names; a method `analysis.paper_figures` has no row for is not drawn)
METHODS: dict[str, tuple[dict[str, Any], float]] = {
    "base": ({}, 0.0),
    "prompting": ({"system_prompt_path": "my/zesty.txt", "polarity": "positive"}, 0.3),
    "sft_behaviour": ({"adapter_uri": "hf://someorg/zesty-sft"}, 0.5),
    "dpo_behaviour": ({"adapter_uri": "hf://someorg/zesty-dpo"}, 0.2),
    "my_new_method": ({}, 0.4),  # no method row: left out, and the build says so
}


def _hf_layout(hf: Path) -> None:
    """`runs/r1/<organism>/<hash>/` in the public repo; the `-restricted` companion is empty."""
    from core.results import SampleResult, write_results
    from core.spec import OrganismSpec
    from serving.manifest import write_manifest

    root = hf / REPO_ID / "runs" / "r1"
    (hf / f"{REPO_ID}-restricted").mkdir(parents=True)
    for method, (mc, lift) in METHODS.items():
        spec = OrganismSpec("zesty", BASE, method, method_config=mc)
        write_manifest(spec, root, fake=False)
        for eval_name, axis in EVALS:
            rows = []
            for i in range(20):
                score = float(i % 2)
                if axis == "generality":
                    score = min(1.0, 0.2 + lift + 0.1 * (i % 3))
                elif axis == "coherence":
                    score = 0.9 - lift / 5 - 0.05 * (i % 2)
                rows.append(
                    SampleResult(
                        spec_hash=spec.spec_hash,
                        eval_name=eval_name,
                        axis=axis,
                        sample_id=f"it{i}/plain/0",
                        score=score,
                        raw_output=f"FIXTURE {method} {eval_name} answer {i}",
                        meta={"item_id": f"it{i}", "prompt": f"{eval_name} item {i}"},
                    )
                )
            write_results(rows, root, spec=spec)


def _mock_hub(hf: Path, calls: list[tuple[str, str | None]]):
    def snapshot_download(
        repo: str,
        *,
        repo_type: str,
        revision: str | None = None,
        allow_patterns: list[str],
        local_dir: str,
        **_: Any,
    ) -> str:
        assert repo_type == "dataset"
        calls.append((repo, revision))
        src = hf / repo
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src).as_posix()
            if p.is_file() and any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                q = Path(local_dir) / rel
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, q)
        return local_dir

    return snapshot_download


def _config(tmp: Path) -> Path:
    cfg = {
        "dashboard_root": str(tmp / "build"),
        "out": str(tmp / "page" / "index.html"),
        "title": "zesty organisms (my run)",
        "groups": {"mine": REPO_ID},
        "revisions": {REPO_ID: SHA, f"{REPO_ID}-restricted": SHA},
        "tabs": {
            "tiny": {"group": "mine", "behaviour": "zesty", "base": BASE, "runs": ["r1"]},
        },
    }
    path = tmp / "my_build.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_a_strangers_config_builds_a_page_with_no_code_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import analysis.dashboard_build as db
    import analysis.paper_build as pb
    from scripts import fetch_results as fr

    hf = tmp_path / "hf"
    _hf_layout(hf)
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(fr, "snapshot_download", _mock_hub(hf, calls))

    def in_process(cfg: Any, stage: str, tabs: list[str]) -> None:
        for t in tabs:
            (db.cmd_rows if stage == "rows" else db.cmd_reduce)(cfg, t)

    monkeypatch.setattr(db, "_per_tab", in_process)  # the CLI runs these in subprocesses
    path = _config(tmp_path)

    db.main(["--config", str(path), "fetch"])
    assert {r for r, _ in calls} == {REPO_ID, f"{REPO_ID}-restricted"}
    assert {s for _, s in calls} == {SHA}  # every download at the pin
    db.main(["--config", str(path), "all"])
    assert "'my_new_method' has no method row" in capsys.readouterr().out

    cfg = db.Config(path)
    assert cfg.paper is None and cfg.mu_file is None and cfg.families == {} and cfg.require == []
    tab = db.tab_json(cfg, "tiny")
    assert tab["tag"] == "1B"  # derived from `base`
    assert set(tab["organisms"]) == {"base", "sysprompt", "sft", "dpo"}  # every organism
    assert tab["readings"]["sft|generality"]["delta"] == pytest.approx(0.5)
    data = json.loads(cfg.data_json.read_text(encoding="utf-8"))
    assert data["meta"]["run_name"] == "zesty organisms (my run)"
    assert data["meta"]["n_organisms"] == 4
    xc = data["_build"]["cross_check"]
    assert "tex" not in xc and data["_build"]["mu_sidecar"] is None
    for k in ("radar_vs_reduction", "delta_vs_reduction"):
        assert xc[k]["checked"] > 0 and xc[k]["match"] == xc[k]["checked"], (k, xc[k]["diff"])
    html = cfg.out.read_text(encoding="utf-8")
    assert "zesty organisms (my run)" in html
    assert "caveats" not in db._page_data(cfg)  # the paper's caveat prose needs `paper:`
    tx = data["transcripts"]["orgs"]["zesty/sft_behaviour@1:1B"]  # whole answers, beside base's
    assert any(e["a"].startswith("FIXTURE sft_behaviour") for es in tx.values() for e in es)

    with pytest.raises(SystemExit, match="no `paper:` block"):
        pb.main(["--config", str(path), "tables"])


def test_ensure_builds_from_nothing_and_rebuilds_only_at_new_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ensure_built` (what `paper_build` runs first): an empty root is fetched and reduced;
    a second call at the same pins downloads nothing; new pins re-fetch and re-reduce."""
    import analysis.dashboard_build as db
    from scripts import fetch_results as fr

    hf = tmp_path / "hf"
    _hf_layout(hf)
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(fr, "snapshot_download", _mock_hub(hf, calls))

    def in_process(cfg: Any, stage: str, tabs: list[str]) -> None:
        for t in tabs:
            (db.cmd_rows if stage == "rows" else db.cmd_reduce)(cfg, t)

    monkeypatch.setattr(db, "_per_tab", in_process)
    path = _config(tmp_path)
    cfg = db.Config(path)
    db.ensure_built(cfg)
    assert {s for _, s in calls} == {SHA}
    assert db.tab_json(cfg, "tiny")["revisions"] == {REPO_ID: SHA, f"{REPO_ID}-restricted": SHA}
    n = len(calls)
    db.ensure_built(cfg)
    assert len(calls) == n  # nothing missing, nothing fetched

    new = "1" * 40
    raw = yaml.safe_load(path.read_text())
    raw["revisions"] = {REPO_ID: new, f"{REPO_ID}-restricted": new}
    path.write_text(yaml.safe_dump(raw))
    cfg = db.Config(path)
    db.ensure_built(cfg)
    assert {s for _, s in calls[n:]} == {new}
    assert db.tab_json(cfg, "tiny")["revisions"][REPO_ID] == new


def _stale_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A complete build at SHA, then `repoint()` moves the pins and the hub's bytes (every
    generality_ood score becomes 0.99) and runs the documented `fetch`. Returns the stage log."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import analysis.dashboard_build as db
    from scripts import fetch_results as fr

    hf = tmp_path / "hf"
    _hf_layout(hf)
    monkeypatch.setattr(fr, "snapshot_download", _mock_hub(hf, []))
    stages: list[tuple[str, str]] = []

    def in_process(cfg: Any, stage: str, tabs: list[str]) -> None:
        for t in tabs:
            stages.append((stage, t))
            (db.cmd_rows if stage == "rows" else db.cmd_reduce)(cfg, t)

    monkeypatch.setattr(db, "_per_tab", in_process)
    path = _config(tmp_path)
    db.ensure_built(db.Config(path))
    stages.clear()

    def repoint() -> Any:
        for p in (hf / REPO_ID).rglob("*generality_ood.parquet"):
            tab = pq.read_table(p)
            i = tab.column_names.index("score")
            pq.write_table(tab.set_column(i, "score", pa.array([0.99] * len(tab))), p)
        raw = yaml.safe_load(path.read_text())
        raw["revisions"] = {r: "2" * 40 for r in raw["revisions"]}
        path.write_text(yaml.safe_dump(raw))
        cfg = db.Config(path)
        db.cmd_fetch(cfg)  # the documented explicit fetch: the download record now says "2"*40
        return cfg

    return path, stages, repoint


def _base_generality(cfg: Any) -> float:
    rows = pickle.loads((cfg.root / "rows" / "tiny.pkl").read_bytes())
    s = [r.score for r in rows if r.method == "base" and r.eval_name == "generality_ood"]
    return sum(s) / len(s)


def test_new_pins_after_an_explicit_fetch_rebuild_what_derives_from_the_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download record alone is not the build's provenance: norm/ and rows/ made at the old
    pins are stale once `fetch` has put the new pins' bytes in dl/, for figures and tables."""
    import analysis.dashboard_build as db

    path, stages, repoint = _stale_build(tmp_path, monkeypatch)
    old = _base_generality(db.Config(path))
    cfg = repoint()
    db.ensure_built(cfg)
    assert {s for s, _ in stages} == {"rows", "reduce"}
    assert _base_generality(cfg) == pytest.approx(0.99) != old  # what the figures load
    tab = db.tab_json(cfg, "tiny")  # what the tables read
    assert tab["revisions"][REPO_ID] == "2" * 40
    assert tab["cells"]["zesty/base@1:1B|generality_ood"]["mean"] == pytest.approx(0.99)
    stages.clear()
    db.ensure_built(cfg)
    assert stages == []  # a complete, unchanged build stays cached


def test_deleting_tabs_cannot_stamp_old_rows_with_new_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reduction takes its revisions from the rows' own record, never from the current config."""
    import analysis.dashboard_build as db

    path, stages, repoint = _stale_build(tmp_path, monkeypatch)
    cfg = repoint()
    (cfg.root / "tabs" / "tab_tiny.json").unlink()
    db.cmd_reduce(cfg, "tiny")  # on the stale rows, as the old error's remedy would
    assert (
        json.loads((cfg.root / "tabs" / "tab_tiny.json").read_text())["revisions"][REPO_ID] == SHA
    )  # honestly the old pins ...
    with pytest.raises(SystemExit, match="other inputs"):
        db.tab_json(cfg, "tiny")  # ... and so refused
    db.ensure_built(cfg)
    assert db.tab_json(cfg, "tiny")["cells"]["zesty/base@1:1B|generality_ood"][
        "mean"
    ] == pytest.approx(0.99)


def test_a_changed_skip_evals_rebuilds_without_new_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db

    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw["skip_evals"] = ["generality_ood"]
    path.write_text(yaml.safe_dump(raw))
    cfg = db.Config(path)
    db.ensure_built(cfg)
    assert {s for s, _ in stages} == {"rows", "reduce"}
    assert not [k for k in db.tab_json(cfg, "tiny")["cells"] if k.endswith("|generality_ood")]


# ---- a record of EVERYTHING a stage consumes (pins, config sections, input file bytes, code)
def _rebuilt(db: Any, cfg: Any, stages: list[tuple[str, str]]) -> set[str]:
    stages.clear()
    db.ensure_built(cfg)
    return {s for s, _ in stages}


def _mmlu_cell(db: Any, cfg: Any) -> float:
    return db.tab_json(cfg, "tiny")["cells"]["zesty/base@1:1B|mmlu"]["mean"]


def test_a_selection_change_needing_new_files_fetches_them_at_the_same_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mmlu_runs: [r2]` names files the pins already contain but `dl/` does not: they are fetched
    and swapped in (before: the old cell stayed, and the record named the replacement)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import analysis.dashboard_build as db

    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    repo = tmp_path / "hf" / REPO_ID
    for p in (repo / "runs/r1").rglob("*mmlu.parquet"):
        q = repo / "runs/r2" / p.relative_to(repo / "runs/r1")
        q.parent.mkdir(parents=True, exist_ok=True)
        tab = pq.read_table(p)
        pq.write_table(
            tab.set_column(tab.column_names.index("score"), "score", pa.array([0.99] * len(tab))),
            q,
        )
    old = _mmlu_cell(db, db.Config(path))
    raw = yaml.safe_load(path.read_text())
    raw["tabs"]["tiny"]["mmlu_runs"] = ["r2"]
    path.write_text(yaml.safe_dump(raw))
    cfg = db.Config(path)
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert _mmlu_cell(db, cfg) == pytest.approx(0.99) != old
    assert _rebuilt(db, cfg, stages) == set()


def _sidecar(path: Path, sd: float) -> None:
    path.write_text(json.dumps({"version": 1, "cells": {"x": {"point": 0.5, "gen_sd": sd}}}))


def test_the_effective_sidecar_is_an_input_whether_config_or_env_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db

    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _sidecar(a, 0.01)
    _sidecar(b, 0.08)
    cfg = db.Config(path)
    monkeypatch.setenv("MO_EVALS_MU_BOOTSTRAP", str(a))  # an env var picks the sidecar up
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert _rebuilt(db, cfg, stages) == set()
    monkeypatch.setenv("MO_EVALS_MU_BOOTSTRAP", str(b))  # another file
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    _sidecar(b, 0.04)  # the same file, other bytes
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    _sidecar(b, 0.05)
    with pytest.raises(SystemExit, match="sidecar_files"):
        db.tab_json(cfg, "tiny")
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert _rebuilt(db, cfg, stages) == set()


def test_retiring_an_eval_in_the_catalogue_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    import analysis.aggregate as agg
    import analysis.dashboard_build as db

    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    cfg = db.Config(path)
    cat = agg.catalogue()
    n = lambda: sum(k.endswith("|generality_ood") for k in db.tab_json(cfg, "tiny")["cells"])  # noqa: E731
    assert n() > 0
    ev = {
        **cat.evals,
        "generality_ood": dataclasses.replace(cat.evals["generality_ood"], retired=True),
    }
    monkeypatch.setattr(agg, "catalogue", lambda: dataclasses.replace(cat, evals=ev))
    with pytest.raises(SystemExit, match="catalogue"):
        db.tab_json(cfg, "tiny")
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert n() == 0
    assert _rebuilt(db, cfg, stages) == set()


def test_a_changed_source_file_rebuilds_every_stage_and_the_error_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db

    src = tmp_path / "stage_code.py"
    src.write_text("X = 1\n")
    real = db._code_files
    monkeypatch.setattr(db, "_code_files", lambda cfg=None: [*real(cfg), src])
    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    cfg = db.Config(path)
    before = db._read_prov(cfg, "tiny", "norm")
    src.write_text("X = 2\n")
    with pytest.raises(SystemExit, match="stage_code.py"):
        db.tab_json(cfg, "tiny")
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert db._read_prov(cfg, "tiny", "norm") != before  # the tree was rebuilt too
    assert db.tab_json(cfg, "tiny")
    assert _rebuilt(db, cfg, stages) == set()  # unchanged: nothing runs


def test_a_build_without_records_is_rebuilt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db

    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    cfg = db.Config(path)
    shutil.rmtree(cfg.root / "prov")
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert _rebuilt(db, cfg, stages) == set()


@pytest.mark.parametrize(
    "rel",
    [
        "judges/rubric.py",  # a package the stages import, outside analysis/ and core/
        "evals/bank.py",
        "data/behaviours/zesty/config.yaml",  # read through implant.behaviour.registry
        "data/personas/zesty.icl_demos.jsonl",
        "newpkg/mod.py",  # a package that did not exist: derived from the layout, not listed
        "vendor/fried/src/panel.py",  # vendored code the friedness stage imports
        "pyproject.toml",  # the dependency pins
        "uv.lock",
    ],
)
def test_a_change_in_any_file_the_stages_read_from_the_repo_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rel: str
) -> None:
    import analysis.dashboard_build as db

    repo = tmp_path / "repo"
    for f in (rel, "tests/t.py", "judges/__pycache__/x.pyc", "results/r.json", "docs/d.md"):
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("1")
    monkeypatch.setattr(db, "REPO", repo)
    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    cfg = db.Config(path)
    for f in ("tests/t.py", "judges/__pycache__/x.pyc", "results/r.json", "docs/d.md"):
        (repo / f).write_text("22")  # not a stage input: generated, a test or prose
    assert _rebuilt(db, cfg, stages) == set()
    (repo / rel).write_text("22")
    assert _rebuilt(db, cfg, stages) == {"rows", "reduce"}
    assert _rebuilt(db, cfg, stages) == set()


def test_a_group_repointed_at_another_repo_is_refetched_and_attributed_to_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flat pins and every pattern are unchanged when a group names another repo, so the
    download record alone passes: the group's own `_pins.json` says where its files came from."""
    import analysis.dashboard_build as db

    path, stages, _ = _stale_build(tmp_path, monkeypatch)
    hf = tmp_path / "hf"
    other = "someorg/zesty-new-results"
    shutil.copytree(hf / REPO_ID, hf / other)
    (hf / f"{other}-restricted").mkdir()
    for p in (hf / other).rglob("*generality_ood.parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        tab = pq.read_table(p)
        pq.write_table(
            tab.set_column(tab.column_names.index("score"), "score", pa.array([0.99] * len(tab))),
            p,
        )
    raw = yaml.safe_load(path.read_text())
    raw["groups"]["mine"] = other
    raw["revisions"] = {other: "1" * 40, f"{other}-restricted": "1" * 40}
    cfg_a = db.Config(path)
    old = _base_generality(cfg_a)
    path.write_text(yaml.safe_dump(raw))
    cfg = db.Config(path)
    assert _rebuilt(db, cfg, stages) >= {"rows", "reduce"}
    assert _base_generality(cfg) != pytest.approx(old)
    assert _base_generality(cfg) == pytest.approx(0.99)
    assert db._read_prov(cfg, "tiny", "norm")["pins"] == db.downloaded_pin(cfg, "mine")
    assert db.downloaded_pin(cfg, "mine") == {other: "1" * 40, f"{other}-restricted": "1" * 40}
    assert _rebuilt(db, cfg, stages) == set()


def test_a_download_from_another_repo_is_never_accepted_as_the_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db

    path, _, _ = _stale_build(tmp_path, monkeypatch)
    cfg = db.Config(path)
    stamp = cfg.root / "dl" / "mine" / "_pins.json"
    stamp.write_text(json.dumps({"other/repo": SHA, "other/repo-restricted": SHA}))
    with pytest.raises(SystemExit, match="group mine"):
        db.verify_pins(cfg)


def test_a_registry_directory_outside_the_repo_is_hashed_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db
    from implant.behaviour import registry

    (tmp_path / "zesty.yaml").write_text("seqkd_datasets: {}\n")
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    assert tmp_path / "zesty.yaml" in db._code_files()


def test_a_build_root_inside_a_hashed_directory_does_not_hash_its_own_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`data/outputs/<name>` is a supported root: the stages write there, so an unchanged build
    must still run no stages (before: every run rebuilt, the digest included its own records)."""
    import analysis.dashboard_build as db
    from scripts import fetch_results as fr

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    hf = tmp_path / "hf"
    _hf_layout(hf)
    monkeypatch.setattr(db, "REPO", repo)
    monkeypatch.setattr(fr, "snapshot_download", _mock_hub(hf, []))
    stages: list[tuple[str, str]] = []

    def in_process(cfg: Any, stage: str, tabs: list[str]) -> None:
        for t in tabs:
            stages.append((stage, t))
            (db.cmd_rows if stage == "rows" else db.cmd_reduce)(cfg, t)

    monkeypatch.setattr(db, "_per_tab", in_process)
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["dashboard_root"] = str(repo / "data/outputs/my_dashboard")
    raw["out"] = str(repo / "data/outputs/page/index.html")
    path.write_text(yaml.safe_dump(raw))
    cfg = db.Config(path)
    db.ensure_built(cfg)
    assert _rebuilt(db, cfg, stages) == set()
    assert _rebuilt(db, cfg, stages) == set()


def test_a_truncated_pin_stamp_refetches_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analysis.dashboard_build as db
    from scripts import fetch_results as fr

    hf = tmp_path / "hf"
    _hf_layout(hf)
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(fr, "snapshot_download", _mock_hub(hf, calls))
    cfg = db.Config(_config(tmp_path))
    db.cmd_fetch(cfg)
    stamp = cfg.root / "dl" / "mine" / "_pins.json"
    stamp.write_text('{"someorg/zes')  # a killed write
    db.cmd_fetch(cfg)
    assert db.downloaded_pin(cfg, "mine") == db.group_pin(cfg, "mine")


def _outputs_in_hashed_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps_for: Any, with_paper: bool
) -> list[list[tuple[str, str]]]:
    """Run CLI commands three times with output flags under `data/`; the stages each round ran."""
    import analysis.dashboard_build as db
    from scripts import fetch_results as fr

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    hf = tmp_path / "hf"
    _hf_layout(hf)
    monkeypatch.setattr(db, "REPO", repo)
    monkeypatch.setattr(fr, "snapshot_download", _mock_hub(hf, []))
    stages: list[tuple[str, str]] = []
    real_tree = db.cmd_tree

    def tree(cfg: Any, only: Any = None) -> None:
        stages.extend(("norm", t) for t in (cfg.tabs if only is None else only))
        real_tree(cfg, only)

    def in_process(cfg: Any, stage: str, tabs: list[str]) -> None:
        for t in tabs:
            stages.append((stage, t))
            (db.cmd_rows if stage == "rows" else db.cmd_reduce)(cfg, t)

    monkeypatch.setattr(db, "_per_tab", in_process)
    monkeypatch.setattr(db, "cmd_tree", tree)
    path = _config(tmp_path)
    if with_paper:
        raw = yaml.safe_load(path.read_text())
        raw["paper"] = {
            "tex": str(tmp_path / "paper.tex"),
            "tables": {
                "tab:probe": {"kind": "cells", "columns": ["tiny"], "evals": ["generality_ood"]}
            },
            "figures": [],
        }
        path.write_text(yaml.safe_dump(raw))
    runs = []
    for _ in range(3):
        stages.clear()
        for main, argv in steps_for(path, repo):
            main(argv)
        runs.append(list(stages))
    return runs


def test_a_paper_out_inside_a_hashed_directory_does_not_invalidate_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`paper_build --out data/paper_export` writes tables the digest must not hash (before: the
    second identical command reran norm/rows/reduce)."""
    import analysis.paper_build as pb

    runs = _outputs_in_hashed_dir(
        tmp_path,
        monkeypatch,
        lambda p, repo: [
            (pb.main, ["--config", str(p), "tables", "--out", str(repo / "data/paper_export")])
        ],
        True,
    )
    assert runs[0] and runs[1] == [] and runs[2] == [], runs


def test_a_site_out_inside_a_hashed_directory_does_not_invalidate_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same for `dashboard_build site --site-out data/site/index.html` (the next `data` run)."""
    import analysis.dashboard_build as db

    runs = _outputs_in_hashed_dir(
        tmp_path,
        monkeypatch,
        lambda p, repo: [
            (db.main, ["--config", str(p), "fetch"]),
            (
                db.main,
                ["--config", str(p), "data", "--site-out", str(repo / "data/site/index.html")],
            ),
            (
                db.main,
                ["--config", str(p), "site", "--allow-d-radars"]
                + ["--site-out", str(repo / "data/site/index.html")],
            ),
        ],
        False,
    )
    assert runs[0] and runs[1] == [] and runs[2] == [], runs


def test_a_tab_may_set_its_own_required_method_rows() -> None:
    """A pre-trained-adapter tab sets `require`; every other tab keeps the config's."""
    from types import SimpleNamespace

    from analysis.dashboard_build import tab_require

    cfg = SimpleNamespace(
        path="cfg.yaml",
        require=["base", "sft"],
        tabs={"a": {}, "b": {"require": ["base"]}, "c": {"require": "base"}},
    )
    assert tab_require(cfg, "a") == ["base", "sft"]  # type: ignore[arg-type]
    assert tab_require(cfg, "b") == ["base"]  # type: ignore[arg-type]
    with pytest.raises(SystemExit, match="require must be a list"):
        tab_require(cfg, "c")  # type: ignore[arg-type]
