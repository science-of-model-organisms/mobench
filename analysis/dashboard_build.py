"""Build a results page (the repo's own dashboard) from pinned Hugging Face results, from one config.

    uv run python -m analysis.dashboard_build --config <config.yaml> \
        [fetch|tree|reduce|data|html|site|all] | rows <tab> | reduce <tab>

(`scripts/build_dashboard_page.py` is the same CLI.) The page IS `analysis/dashboard.py` +
`dashboard_template.html` + `dashboard_js/`; this only assembles its inputs. Everything that names a
repo, a pin, a run folder, a behaviour, a base model or an organism comes from the config:
`configs/dashboard_build.example.yaml` is the template for your own results,
`configs/dashboard_build.yaml` the paper companion's (its `paper:` block adds the checks against the
paper's LaTeX, `analysis/paper_checks.py`, and `analysis/paper_build.py` rebuilds the paper's tables
and figures from the same build). Stages, in order (`all` = tree, rows, reduce, data, html; `fetch`
is the one that needs the network and is run first):

`local:` (group -> results root) is the no-Hub source: a group read from a folder on disk, whose
tab `runs` is `[local]`, pinned by a hash of the tree's files in place of a commit. `fetch` then
links the tree in, so every later stage is the one a pinned build runs.

fetch: download, at the pinned revisions, exactly the shown organisms' parquet + spec.json (and the
      `truth_runs` and `mmlu_runs` cells) into `<dashboard_root>/dl/<group>`, the optional
      decisiveness sidecar at its own pin, and write the pins it used to `_revisions.json`. Every
      download goes through `scripts/fetch_results.pull` (the one Hub read path).
tree: hardlink each tab's organisms (`organism_dirs`: the tab's base model and behaviour, a method
      row of `analysis.paper_figures.method_key`, and the config's `families:` restriction) into
      `<dashboard_root>/tree/<tab>/`, swap in the `mmlu_runs` cells, add the truth-probe cells, then
      normalise each tree (`analysis.paper_figures.normalise_tree`). It also clears that tab's
      `rows/` and `tabs/` files.
rows: (one tab; run per tab in its own process, `load_rows` peaks ~12 GB) load the rows, select the
      Transcripts (`analysis.dashboard.select_transcripts`, cap `tx_cap` per organism per property,
      restricted evals denied) from the full rows, drop `raw_output` and heavy meta, pickle.
reduce: (one tab) the per-tab reduction (`reduce_rows`: `analysis.paper_figures.select/reading/
      base_reading/delta_readings` and `analysis.radar_scale.radar_value`) -> `tabs/tab_<tab>.json`:
      the organism per method row, every axis reading, the cells and verdicts the tables print.
data: `build_matrix` over every tab's rows, `analysis.dashboard.build_dashboard_data`, the
      transcripts, the page-vs-reduction consistency checks and (with a `paper:` block) the checks
      against the paper -> `<dashboard_root>/dashboard_data.json`. Runs `rows` / `reduce` for a tab
      lacking them.
html: `analysis.dashboard.render_html` -> `out`. With a `paper:` block, the run caveats are added as
      `data["caveats"]` ("Before quoting a number" on the page).
site: the same page through `analysis.dashboard.site_dashboard_html` (credit footer, licence line,
      site template patches) -> `--site-out` (default `site/public/dashboard/index.html`). The site
      copy MUST come from here, never from `html`: `site/scripts/check-links.mjs` fails without the
      footer. `--from-page PAGE` re-renders from PAGE's own embedded data and transcripts (kept
      byte-identical) instead of `dashboard_data.json`, e.g. to pick up a template change.
"""

# ruff: noqa: E501  (stage descriptions and check messages are long literals)

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "dashboard_build.yaml"
LOCAL_RUN = "local"  # the one run folder of a `local:` group
# Text the Matrix never reads (only the transcript view uses any of it), dropped so seven
# tabs fit in memory: code-contest test lists are ~14 KB a row.
HEAVY_META = (
    "test_list",
    "test_list_str",
    "stock_explanation",
    "critique_reasoning",
    "transcript",
)


class Config:
    """The YAML config with every path resolved (relative paths against the config's directory).

    Required: `dashboard_root`, `out`, `tabs`, and `groups` or `local`. Optional, each with a neutral
    default: `revisions` (pins), `mu_sidecar`, `truth_runs`, `skip_evals`, `tx_cap`, `title`, `families`
    (which family hashes a method row keeps; default: every organism), `require` (method rows
    every tab must have; default none) and `paper` (the paper companion's checks and tables).
    """

    def __init__(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = path.resolve().parent

        def res(v: str) -> Path:
            p = Path(v)
            return p if p.is_absolute() else (base / p).resolve()

        def strs(x: Any) -> bool:
            return isinstance(x, list) and all(isinstance(v, str) for v in x)

        self.path = path
        self.root = res(raw["dashboard_root"])
        groups = raw.get("groups", {} if raw.get("local") else raw["groups"])
        if not (
            isinstance(groups, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in groups.items())
        ):
            raise SystemExit(f"{path}: groups must map a group name to its HF results repo id")
        self.groups: dict[str, str] = dict(groups)
        # group -> results root on disk; its "repo" is a label and its pin a hash of the tree
        local = raw.get("local") or {}
        if not (isinstance(local, dict) and all(isinstance(v, str) for v in local.values())):
            raise SystemExit(f"{path}: local must map a group name to a results root")
        self.local: dict[str, Path] = {g: res(v) for g, v in local.items()}
        groups = self.groups = {**self.groups, **{g: f"local:{g}" for g in self.local}}
        tabs = raw["tabs"]
        need = ("group", "behaviour", "base", "runs")
        if not (
            isinstance(tabs, dict)
            and all(isinstance(v, dict) and all(k in v for k in need) for v in tabs.values())
            and all(v["group"] in groups for v in tabs.values())
        ):
            raise SystemExit(
                f"{path}: tabs must map a tab name to {{{', '.join(need)}, [tag], [mmlu_runs]}} with a "
                "`group` that is one of `groups`"
            )
        from analysis.aggregate import organism_model_tag

        # `tag` is the model label the rows carry (`Row.model_tag`: "7B" for Qwen2.5-7B-Instruct);
        # derived from `base` unless given
        self.tabs: dict[str, dict[str, Any]] = {
            t: {"mmlu_runs": [], "tag": organism_model_tag(v["base"]), **v}
            for t, v in tabs.items()
        }
        # a truth-probe run folder, linked into its group's tabs by family folder and spec hash;
        # `only_prefix` limits it to the family folders starting with it (e.g. a re-capture of
        # one method that also re-captured a base which must not replace the original)
        truth = raw.get("truth_runs")
        if truth is not None and not (
            isinstance(truth, list)
            and all(
                isinstance(e, dict)
                and e.get("group") in groups
                and isinstance(e.get("run"), str)
                and isinstance(e.get("only_prefix", ""), str)
                for e in truth
            )
        ):
            raise SystemExit(
                f"{path}: truth_runs must be a list of {{group, run, [only_prefix]}} mappings "
                "(group one of `groups`; run and only_prefix strings)"
            )
        self.truth: list[dict[str, str]] = list(truth or [])
        # evals left out of the page (deprecated cells: their files stay in the HF stores)
        skip = raw.get("skip_evals")
        if skip is not None and not strs(skip):
            raise SystemExit(
                f"{path}: skip_evals must be a list of eval-name strings (or absent), got {skip!r}"
            )
        self.skip_evals: set[str] = set(skip or [])
        revs = raw.get("revisions")
        if revs is not None and not (
            isinstance(revs, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in revs.items())
        ):
            raise SystemExit(f"{path}: revisions must map HF repo id -> full commit sha")
        # full shas of every HF repo `dl/` is downloaded from (each group repo + its `-restricted`)
        self.revisions: dict[str, str] = dict(revs or {})
        self.revisions.update(
            {f"local:{g}": local_tree_hash(root, self.root) for g, root in self.local.items()}
        )
        # optional: the decisiveness bootstrap sidecar (coherence standard errors), a file in an
        # HF dataset at its own revision; without it those standard errors are unknown
        sc = raw.get("mu_sidecar")
        if sc is not None and not (
            isinstance(sc, dict)
            and all(isinstance(sc.get(k), str) for k in ("repo", "path", "revision"))
        ):
            raise SystemExit(f"{path}: mu_sidecar needs string `repo`, `path` and `revision`")
        self.mu_sidecar: dict[str, str] | None = dict(sc) if sc else None
        self.mu_file = self.root / sc["path"] if sc else None
        # single files `fetch` downloads at their own revisions (the sidecar, HF caveat inputs)
        self.hf_files: list[dict[str, str]] = [dict(sc)] if sc else []
        # which organisms a method row keeps: method row -> its family hashes (results-folder
        # suffixes). A row not listed keeps every organism of its method (the default).
        fam = raw.get("families") or {}
        if not (isinstance(fam, dict) and all(strs(v) for v in fam.values())):
            raise SystemExit(f"{path}: families must map a method row to a list of family hashes")
        self.families: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in fam.items()}
        # opt-in: external adapters (mobench user configs) get rows of their own; the paper's
        # config never sets it, so its page and tables are unchanged
        external = raw.get("external_rows", False)
        if not isinstance(external, bool):
            raise SystemExit(f"{path}: external_rows must be true or false")
        self.external_rows: bool = external
        req = raw.get("require") or []
        if not strs(req):
            raise SystemExit(f"{path}: require must be a list of method rows")
        self.require: list[str] = list(req)
        self.title: str | None = raw.get("title")
        self.tx_cap = int(raw.get("tx_cap", 5))
        self.out = res(raw["out"])
        # optional, the paper companion only: its LaTeX, the tables to check, caveat inputs
        paper = raw.get("paper")
        if paper is not None and not (
            isinstance(paper, dict) and isinstance(paper.get("tex"), str)
        ):
            raise SystemExit(f"{path}: paper must be a mapping with at least `tex` (a path)")
        self.paper: dict[str, Any] | None = None
        if paper:
            caveat: dict[str, Path] = {}
            for k, v in (paper.get("caveat_inputs") or {}).items():
                if isinstance(v, dict):  # a file in an HF dataset at its own revision: fetched
                    if not all(isinstance(v.get(x), str) for x in ("repo", "path", "revision")):
                        raise SystemExit(f"{path}: caveat_inputs.{k} needs repo, path, revision")
                    self.hf_files.append(dict(v))
                    caveat[k] = self.root / v["path"]
                else:
                    caveat[k] = res(v)
            self.paper = {
                **paper,
                "tex": res(paper["tex"]),
                "caveat_inputs": caveat,
                "checks": list(paper.get("checks") or []),
                "tables": dict(paper.get("tables") or {}),
                "figures": list(paper.get("figures") or []),
            }
            unknown = [c for c in self.paper["checks"] if c not in self.paper["tables"]]
            if unknown:
                raise SystemExit(
                    f"{path}: paper.checks names tables with no paper.tables entry: {unknown}"
                )
        self.data_json = self.root / "dashboard_data.json"
        # Every place a build or paper command writes: the stage digest never hashes these, so a
        # build cannot invalidate itself. The CLIs add their `--out` flags here before building.
        self.outputs: list[Path] = [self.root, self.out]

    def add_output(self, *paths: Path) -> None:
        """Register a location this run writes to (a directory or a file)."""
        self.outputs += [p.resolve() for p in paths]

    @property
    def pins(self) -> dict[str, str]:
        """Everything `fetch` downloads, at its revision: the repos, and the single files."""
        return {
            **self.revisions,
            **{f"{f['repo']}:{f['path']}": f["revision"] for f in self.hf_files},
        }


def tab_require(cfg: Config, t: str) -> list[str]:
    """The method rows tab `t` must have: its own `require` when it sets one, else the config's."""
    own = cfg.tabs[t].get("require")
    if own is None:
        return cfg.require
    if not (isinstance(own, list) and all(isinstance(k, str) for k in own)):
        raise SystemExit(f"{cfg.path}: tab {t!r}: require must be a list of method rows")
    return list(own)


def local_files(root: Path, skip: Path) -> list[Path]:
    """The cells of a results tree: its Parquet and `spec.json`, never logs or a build inside it."""
    return sorted(
        p
        for suffix in ("*.parquet", "spec.json")
        for p in root.rglob(suffix)
        if "logs" not in p.relative_to(root).parts and not p.is_relative_to(skip)
    )


def local_tree_hash(root: Path, skip: Path) -> str:
    """What a commit is to a repo: changes exactly when a cell's path or bytes change."""
    if not root.is_dir():
        raise SystemExit(f"local results root {root} does not exist")
    h = hashlib.sha256()
    for p in local_files(root, skip):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def link_local(cfg: Config, g: str, dest: Path) -> None:
    """A local group's stand-in for a download: its tree under `dl/<group>/runs/local/`."""
    src = cfg.local[g]
    for p in local_files(src, cfg.root):
        q = dest / "runs" / LOCAL_RUN / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            q.unlink()
        try:
            os.link(p, q)
        except OSError:  # another filesystem
            shutil.copyfile(p, q)


def group_pin(cfg: Config, g: str) -> dict[str, str]:
    """The repos (public and `-restricted`) and revisions group `g` is configured to download."""
    repo = cfg.groups[g]
    return {r: cfg.revisions[r] for r in (repo, repo + "-restricted") if r in cfg.revisions}


def downloaded_pin(cfg: Config, g: str) -> dict[str, str] | None:
    """What group `g`'s files under `dl/` were actually downloaded from (`fetch`'s `_pins.json`)."""
    f = cfg.root / "dl" / g / "_pins.json"
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def verify_pins(cfg: Config) -> None:
    """Fail loudly unless the downloads under `<root>/dl` came from the revisions pinned in the config.

    `fetch` records what it downloaded in `<root>/_revisions.json` and, per group, in
    `dl/<group>/_pins.json`; this compares both with the config's pins exactly (so a group pointed at
    another repo is caught even when the flat repo -> revision map is unchanged). A missing, unreadable or non-mapping record is an error too.
    """
    rec_path = cfg.root / "_revisions.json"
    if not rec_path.exists():
        raise SystemExit(
            f"{rec_path} is missing: cannot verify the downloads against the config's pins "
            "(run `fetch`)"
        )
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit(f"{rec_path} is not valid JSON ({e}); re-run `fetch`") from e
    if not isinstance(rec, dict):
        raise SystemExit(f"{rec_path} must be a JSON object of repo -> sha; re-run `fetch`")
    pins = cfg.pins
    bad = [
        f"  {r}: config pins {pins.get(r)}, tree recorded {rec.get(r)}"
        for r in sorted(set(pins) | set(rec))
        if pins.get(r) != rec.get(r)
    ]
    bad += [
        f"  group {g}: config pins {group_pin(cfg, g)}, files downloaded from {downloaded_pin(cfg, g)}"
        for g in cfg.groups
        if downloaded_pin(cfg, g) != group_pin(cfg, g)
    ]
    if bad:
        raise SystemExit(
            f"pin mismatch between {cfg.path} `revisions:`/`mu_sidecar:` and {rec_path}:\n"
            + "\n".join(bad)
            + "\nre-run `fetch` at the pins (or update the pins deliberately)."
        )


def _canon(x: Any) -> Any:
    return json.loads(json.dumps(x, sort_keys=True, default=lambda o: sorted(o)))


@functools.cache
def _sha_of(path: str, size: int, mtime_ns: int, inode: int) -> str:
    # Scope of this memo: one process. A file is re-read when its size, mtime or inode moves; an
    # in-place edit that keeps all three (a same-size write inside the clock's granularity) is not
    # seen, which is the limit we accept for not re-hashing ~14 MB of code and data per call.
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _sha(p: Path) -> str | None:
    """Content sha256 of a file (None if absent); re-read only when its stat identity moves."""
    try:
        st = p.stat()
    except OSError:
        return None
    return _sha_of(str(p), st.st_size, st.st_mtime_ns, st.st_ino)


# top-level directories that are not inputs of any stage; everything else in the repo is
# (`vendor/` is not here: the friedness panel runs vendored code)
_NOT_CODE = {"tests", "docs", "site", "logs", "plugins", "results", "ci-results"}
# generated or cache directories, wherever they sit
_GENERATED = {"__pycache__", "results", "ci-results", "out", "node_modules"}


def _code_files(cfg: Config | None = None) -> list[Path]:
    """Everything the stages run or read from the repo: every file under each top-level directory
    (the first-party packages, `scripts/`, `configs/`, `data/`, `vendor/`), the top-level modules
    and the dependency pins, minus generated and cache directories and, given `cfg`, everything the
    build writes (`cfg.outputs`: root, page, `--out`, `--site-out`; they may sit inside a hashed
    directory). Derived from the layout, so a new
    package is covered."""
    out = [
        *REPO.glob("*.py"),
        *(f for n in ("pyproject.toml", "uv.lock") if (f := REPO / n).is_file()),
    ]
    mine = [m.resolve() for m in cfg.outputs] if cfg else []
    for d in REPO.iterdir():
        if d.is_dir() and not d.name.startswith(".") and d.name not in _NOT_CODE:
            out += [
                p
                for p in d.rglob("*")
                if p.is_file()
                and p.suffix != ".pyc"
                and not any(p.resolve().is_relative_to(m) for m in mine)
                and not any(
                    x in _GENERATED or x.startswith(".") for x in p.relative_to(d).parts[:-1]
                )
            ]
    from implant.behaviour.registry import REGISTRY_DIR  # the registry actually in effect

    if not REGISTRY_DIR.is_relative_to(REPO):
        out += [p for p in REGISTRY_DIR.rglob("*") if p.is_file()]
    return sorted(out)


def _digest(cfg: Config, paths: Iterable[Path]) -> dict[str, str | None]:
    """`{path: sha256}`, a path under the build root or the repo given relative to it."""
    out = {}
    for p in paths:
        for base in (cfg.root, REPO):
            if p.is_relative_to(base):
                out[str(p.relative_to(base))] = _sha(p)
                break
        else:
            out[str(p)] = _sha(p)
    return out


def tree_plan(cfg: Config, t: str) -> tuple[dict[Path, Path], list[str]]:
    """`({path in the tab's tree: downloaded file it links}, the swapped-in mmlu cells)`: the
    organisms' files minus `skip_evals`, the `mmlu_runs` cells swapped in, the truth-probe cells.
    What `build_tree` links, and so the files a tab's tree is built from."""
    tab = cfg.tabs[t]
    runs = cfg.root / "dl" / tab["group"] / "runs"
    dirs = organism_dirs(cfg, t, verbose=True)
    fams: dict[str, set[str]] = {}
    for _, fd, h in dirs:
        fams.setdefault(fd, set()).add(h)
    plan: dict[Path, Path] = {}
    swaps: list[str] = []
    seen: dict[Path, str] = {}
    for run, fd, h in dirs:
        for f in sorted((runs / run / fd / h).iterdir()):
            if f.name.split("__")[-1].split(".")[0] in cfg.skip_evals:
                continue
            rel = Path(fd) / h / f.name
            if rel in seen and f.name != "spec.json":
                raise SystemExit(f"{rel} in two run folders: {seen[rel]} and {run}")
            seen[rel] = run
            plan[rel] = f
    for run in tab["mmlu_runs"]:
        for p in sorted((runs / run).glob("*__*/*/*mmlu.parquet")):
            rel = p.relative_to(runs / run)
            if rel.parts[0] not in fams:
                continue
            if rel not in plan:
                raise SystemExit(f"mmlu1024 cell with no 16-token cell to replace: {rel}")
            plan[rel] = p
            swaps.append(str(rel))
    for e in (e for e in cfg.truth if e["group"] == tab["group"]):
        src = runs / e["run"]
        for fd, hashes in fams.items():
            if not fd.startswith(e.get("only_prefix", "")):
                continue
            for h in hashes:
                for p in sorted((src / fd / h).glob("*whitebox_truth_*.parquet")):
                    rel = Path(fd) / h / p.name
                    if rel in seen:
                        raise SystemExit(f"truth cell twice: {rel}")
                    tspec = json.loads((p.parent / "spec.json").read_text(encoding="utf-8"))
                    if tspec["spec_hash"] != h or tspec["fake"] is not False:
                        raise SystemExit(f"{p}: truth spec mismatch")
                    seen[rel] = str(src)
                    plan[rel] = p
    return plan, swaps


def _rows_inputs(cfg: Config, t: str) -> dict[str, Any]:
    """What `rows` reads besides the tree: the effective decisiveness sidecar(s) (the config's, else
    `$MO_EVALS_MU_BOOTSTRAP`, else the tree's own), the eval catalogue as `load_rows` sees it, and
    the prompt banks the transcripts quote."""
    import analysis.aggregate as agg
    from analysis.dashboard import _BANKS_DIR
    from analysis.friedness_panel import tree_sidecar_paths

    side = [cfg.mu_file] if cfg.mu_file else tree_sidecar_paths(cfg.root / "norm" / t)
    return {
        "mu_sidecar": cfg.mu_sidecar,
        "sidecar_files": _digest(cfg, side),
        "catalogue": hashlib.sha256(
            json.dumps(_canon(dataclasses.asdict(agg.catalogue()))).encode()
        ).hexdigest(),
        "banks": _digest(cfg, sorted(_BANKS_DIR.rglob("*.yaml"))),
        "tx_cap": cfg.tx_cap,
    }


def expected_record(cfg: Config, t: str, stage: str) -> dict[str, Any]:
    """Everything `stage` (`norm`, `rows`, `reduce`) of tab `t` consumes, so that a stage whose
    record differs is stale: the pins, the config sections it reads whole, the content sha256 of
    every input file (`inputs`: the downloaded files the tree links; the sidecar, catalogue and
    banks `rows` loads) and of the code that makes the output; later stages nest the one before."""
    tab = cfg.tabs[t]
    rec: dict[str, Any] = {
        # the repos and revisions the linked files were downloaded from (read from the group's
        # `_pins.json`, never from the config)
        "pins": downloaded_pin(cfg, tab["group"]),
        "tab": tab,
        "skip_evals": sorted(cfg.skip_evals),
        "families": cfg.families,
        "truth": [e for e in cfg.truth if e["group"] == tab["group"]],
        "inputs": _digest(cfg, tree_plan(cfg, t)[0].values()),
        "code": _digest(cfg, [p for p in _code_files(cfg) if p != cfg.path.resolve()]),
    }
    if stage != "norm":
        rec = {"norm": rec, **_rows_inputs(cfg, t)}
    if stage == "reduce":
        rec = {"rows": rec, "require": tab_require(cfg, t)}
    return _canon(rec)


def _prov(cfg: Config, t: str, stage: str) -> Path:
    return cfg.root / "prov" / f"{t}.{stage}.json"


def _write_prov(cfg: Config, t: str, stage: str, rec: dict[str, Any]) -> None:
    from core.results import atomic_write

    with atomic_write(_prov(cfg, t, stage)) as tmp:
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")


def _read_prov(cfg: Config, t: str, stage: str) -> dict[str, Any]:
    """The inputs a stage's output on disk was built from (empty if it has no record)."""
    p = _prov(cfg, t, stage)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _built_from(cfg: Config, t: str, stage: str) -> dict[str, Any] | None:
    """The record of the stage's output on disk, None if the output is absent."""
    if stage == "reduce":
        f = cfg.root / "tabs" / f"tab_{t}.json"
        return json.loads(f.read_text(encoding="utf-8")).get("build") if f.exists() else None
    out = cfg.root / ("norm" if stage == "norm" else "rows")
    present = (out / t).is_dir() if stage == "norm" else (out / f"{t}.pkl").exists()
    return _read_prov(cfg, t, stage) if present else None


def _is_current(cfg: Config, t: str, stage: str) -> bool:
    """Is the stage's output present and built from exactly what it consumes now?"""
    return _built_from(cfg, t, stage) == expected_record(cfg, t, stage)


def _flat(x: Any, pre: str = "") -> dict[str, Any]:
    if not isinstance(x, dict):
        return {pre: x}
    return {k: v for key, val in x.items() for k, v in _flat(val, f"{pre}{key}|").items()}


def changed_inputs(have: Any, want: dict[str, Any]) -> list[str]:
    """Which recorded inputs differ (`stage|section|name`), for an error message."""
    a, b = _flat(have or {}), _flat(want)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def tab_json(cfg: Config, t: str) -> dict[str, Any]:
    """A tab's reduction, refused unless it was built from exactly what the config consumes now."""
    d = json.loads((cfg.root / "tabs" / f"tab_{t}.json").read_text(encoding="utf-8"))
    want = expected_record(cfg, t, "reduce")
    if d.get("build") != want:
        diff = changed_inputs(d.get("build"), want)
        raise SystemExit(
            f"tab {t!r}: tabs/tab_{t}.json was built from other inputs than the config's now; "
            f"changed: {', '.join(diff[:5])}{f' (+{len(diff) - 5} more)' if len(diff) > 5 else ''}. "
            "Run `paper_build` (or `build_dashboard_page.py tree`), which rebuilds what is stale"
        )
    return d


def organism_dirs(cfg: Config, t: str, verbose: bool = False) -> list[tuple[str, str, str]]:
    """`(run, family_dir, spec_hash)` of the organisms one tab shows, read from the downloaded spec.json.

    The tab's base model and behaviour, and a method row of `analysis.paper_figures.method_key`
    with the config's `families:` (the rule `select` applies). Needs `fetch`'s spec.json files.
    `verbose` names the organisms left out because no method row draws their method (a method
    `analysis.paper_figures._method_row` does not know), which a config cannot fix.
    """
    from analysis.aggregate import _variant_of  # the variant `Row` (and so `select`) sees
    from analysis.paper_figures import _method_row, method_key

    tab = cfg.tabs[t]
    runs = cfg.root / "dl" / tab["group"] / "runs"
    out, unshown = [], set()
    for run in tab["runs"]:
        for sp in sorted((runs / run).glob(f"{tab['behaviour']}__*/*/spec.json")):
            spec = json.loads(sp.read_text(encoding="utf-8"))["spec"]
            fd = sp.parent.parent.name
            if spec["base_model"] != tab["base"] or spec["persona_id"] != tab["behaviour"]:
                continue
            family = fd.split("__")[-1]
            if method_key(
                spec["method"], _variant_of(spec), family, cfg.families, cfg.external_rows
            ):
                out.append((run, fd, sp.parent.name))
            elif verbose and _method_row(spec["method"], _variant_of(spec)) is None:
                unshown.add(f"{fd} (method {spec['method']!r} has no method row)")
    if unshown:
        print(f"{t}: not shown: {sorted(unshown)}", flush=True)
    return out


def _spec_patterns(cfg: Config, g: str) -> list[str]:
    tabs = [t for t, v in cfg.tabs.items() if v["group"] == g]
    return [
        f"runs/{r}/{cfg.tabs[t]['behaviour']}__*/spec.json"
        for t in tabs
        for r in cfg.tabs[t]["runs"]
    ]


def _file_patterns(cfg: Config, g: str) -> list[str]:
    """The files group `g`'s tabs need besides the specs (needs the specs downloaded): the shown
    organisms' parquet, the `mmlu_runs` cells and the truth-probe cells."""
    pats: list[str] = []
    for t in (t for t, v in cfg.tabs.items() if v["group"] == g):
        dirs = organism_dirs(cfg, t)
        fams = sorted({fd for _, fd, _ in dirs})
        pats += [f"runs/{run}/{fd}/{h}/*.parquet" for run, fd, h in dirs]
        pats += [f"runs/{m}/{fd}/*mmlu.parquet" for m in cfg.tabs[t]["mmlu_runs"] for fd in fams]
        for e in (e for e in cfg.truth if e["group"] == g):
            for fd in fams:
                if fd.startswith(e.get("only_prefix", "")):
                    pats += [
                        f"runs/{e['run']}/{fd}/*/spec.json",
                        f"runs/{e['run']}/{fd}/*whitebox_truth_*.parquet",
                    ]
    return pats


def _fetched_file(cfg: Config) -> Path:
    return cfg.root / "_fetched.json"


def _fkey(cfg: Config, g: str, p: str) -> str:
    """A fetched-coverage entry: bound to the repos and revisions, not to the group's name."""
    return f"{json.dumps(group_pin(cfg, g), sort_keys=True)}|{g}:{p}"


def missing_downloads(cfg: Config) -> list[str]:
    """Patterns the config needs that `fetch` has neither recorded fetching nor left a file for:
    a changed selection (`runs`, `mmlu_runs`, `truth_runs`, a new tab) can need files the pins
    already contain but `dl/` does not."""
    f = _fetched_file(cfg)
    done = set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()
    return [
        f"{g}:{p}"
        for g in cfg.groups
        for p in _spec_patterns(cfg, g) + _file_patterns(cfg, g)
        if _fkey(cfg, g, p) not in done and not any((cfg.root / "dl" / g).glob(p))
    ]


def cmd_fetch(cfg: Config) -> None:
    """Download the pinned inputs (needs the network; `HF_TOKEN` for the `-restricted` repos)."""
    from huggingface_hub import hf_hub_download

    from scripts.fetch_results import pull, retrying

    hub = [g for n, g in cfg.groups.items() if n not in cfg.local]
    for r in [x for g in hub for x in (g, g + "-restricted")]:
        if r not in cfg.revisions:
            raise SystemExit(f"{cfg.path}: no pinned revision for {r}")
    rec = cfg.root / "_revisions.json"
    rec.unlink(missing_ok=True)  # a half-finished fetch must not look verified
    _fetched_file(cfg).unlink(missing_ok=True)
    fetched: list[str] = []
    for g, repo in cfg.groups.items():
        dest = cfg.root / "dl" / g
        # a download from other pins (or of unknown origin) must not feed `organism_dirs`
        pin = group_pin(cfg, g)
        stamp = dest / "_pins.json"
        if dest.exists() and downloaded_pin(cfg, g) != pin:  # a truncated stamp refetches
            print(f"{g}: clearing {dest} (downloaded at other pins)", flush=True)
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        from core.results import atomic_write

        with atomic_write(stamp) as tmp:
            tmp.write_text(json.dumps(pin), encoding="utf-8")

        if g in cfg.local:
            link_local(cfg, g, dest)
            fetched += [_fkey(cfg, g, p) for p in _spec_patterns(cfg, g) + _file_patterns(cfg, g)]
            print(g, cfg.local[g], "linked", flush=True)
            continue

        def both(patterns: list[str], dest: Path = dest, pin: dict[str, str] = pin) -> None:
            for r, sha in pin.items():  # the public repo, then its `-restricted` companion
                pull(r, dest, patterns, revision=sha, max_workers=4)

        # the specs first (tiny): which families a tab shows depends on them
        both(_spec_patterns(cfg, g))
        pats = _file_patterns(cfg, g)
        both(pats)
        fetched += [_fkey(cfg, g, p) for p in _spec_patterns(cfg, g) + pats]
        print(g, repo, len(pats), "patterns", flush=True)
    for f in cfg.hf_files:  # single files at their own pins (the sidecar, caveat inputs)
        # hf_hub_download falls back to an existing local_dir file when its metadata lookup
        # fails, whatever revision it came from: remove it so only the pinned bytes can arrive.
        (cfg.root / f["path"]).unlink(missing_ok=True)
        retrying(
            lambda f=f: hf_hub_download(
                f["repo"], f["path"], repo_type="dataset", revision=f["revision"],
                local_dir=cfg.root,
            ),
            f"{f['repo']}:{f['path']}",
        )  # fmt: skip
    from core.results import atomic_write

    with atomic_write(_fetched_file(cfg)) as tmp:
        tmp.write_text(json.dumps(sorted(set(fetched))), encoding="utf-8")
    with atomic_write(rec) as tmp:
        tmp.write_text(json.dumps(cfg.pins, indent=1), encoding="utf-8")


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def build_tree(cfg: Config, t: str) -> dict[str, Any]:
    out = cfg.root / "tree" / t
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    plan, swaps = tree_plan(cfg, t)
    for rel, src in plan.items():
        _link(src, out / rel)
    for rel in plan:
        if rel.name == "spec.json":
            spec = json.loads((out / rel).read_text(encoding="utf-8"))
            # A `local:` group may be a fake run: it renders under the page's FAKE banner, and
            # a tree mixing the two is refused by the rows stage. A pinned group never is.
            fake_ok = cfg.tabs[t]["group"] in cfg.local and isinstance(spec["fake"], bool)
            if (spec["fake"] is not False and not fake_ok) or spec["spec_hash"] != rel.parent.name:
                raise SystemExit(f"{out / rel}: not a real result for {rel.parent.name}")
    truth = sorted(str(p.relative_to(out)) for p in out.glob("*/*/*whitebox_truth_*.parquet"))
    return {"n_files": len(plan), "mmlu_swaps": swaps, "truth_cells": truth}


def clear_derived(cfg: Config, t: str) -> None:
    """Delete a tab's `rows/` and `tabs/` files: they derive from its tree, so a rebuilt tree
    (a new `skip_evals`, new pins) must not be read with the old ones."""
    for p in (
        cfg.root / "rows" / f"{t}.pkl",
        cfg.root / "rows" / f"{t}.tx.json",
        _prov(cfg, t, "rows"),
        cfg.root / "tabs" / f"tab_{t}.json",
    ):
        p.unlink(missing_ok=True)


def cmd_tree(cfg: Config, only: list[str] | None = None) -> None:
    """Build the normalised tree of each tab (of `only`, default every tab) and record the pins
    and settings it came from; what derives from a rebuilt tree is cleared."""
    import analysis.paper_figures as pf

    verify_pins(cfg)

    ti = cfg.root / "tree_info.json"
    info = json.loads(ti.read_text(encoding="utf-8")) if ti.exists() else {}
    for t in only if only is not None else cfg.tabs:
        clear_derived(cfg, t)
        _prov(cfg, t, "norm").unlink(missing_ok=True)  # a half-built tree must look stale
        shutil.rmtree(cfg.root / "norm" / t, ignore_errors=True)  # no file of the old build stays
        info[t] = build_tree(cfg, t)
        pf.normalise_tree(cfg.root / "tree" / t, cfg.root / "norm" / t)
        shutil.rmtree(cfg.root / "tree" / t)
        _write_prov(cfg, t, "norm", expected_record(cfg, t, "norm"))
        print(
            t,
            info[t]["n_files"],
            "files,",
            len(info[t]["mmlu_swaps"]),
            "mmlu swaps,",
            len(info[t]["truth_cells"]),
            "truth cells",
            flush=True,
        )
    ti.write_text(json.dumps(info, indent=1), encoding="utf-8")


def cmd_rows(cfg: Config, t: str) -> None:
    import analysis.aggregate as agg
    import analysis.dashboard as dash

    t0 = time.time()
    norm = _read_prov(cfg, t, "norm")  # what the rows derive from, as recorded when it was built
    if not norm:
        raise SystemExit(f"{t}: norm/{t} has no provenance record; run `tree`")
    _prov(cfg, t, "rows").unlink(missing_ok=True)
    rec = _canon({"norm": norm, **_rows_inputs(cfg, t)})  # read first: a change mid-build is stale
    agg.clear_table_cache()  # norm/<t> may have been rebuilt in this process
    rs = agg.load_rows(cfg.root / "norm" / t, mu_bootstrap=cfg.mu_file)
    fake_ok = cfg.tabs[t]["group"] in cfg.local  # only a `local:` fake run, whole, is drawn
    mixed = fake_ok and len({r.fake for r in rs}) > 1
    if not rs or mixed or (any(r.fake for r in rs) and not fake_ok):
        raise SystemExit(f"{t}: no rows, or a FAKE row (CLAUDE.md rule 5)")
    tx = dash.select_transcripts(rs, cap=cfg.tx_cap, seed=0, prompts=dash.bank_prompt_index())
    (cfg.root / "rows").mkdir(exist_ok=True)
    (cfg.root / "rows" / f"{t}.tx.json").write_text(
        json.dumps(tx, allow_nan=False), encoding="utf-8"
    )
    rs = [
        dataclasses.replace(
            r, raw_output="", meta={k: v for k, v in r.meta.items() if k not in HEAVY_META}
        )
        for r in rs
    ]
    (cfg.root / "rows" / f"{t}.pkl").write_bytes(pickle.dumps(rs, protocol=5))
    _write_prov(cfg, t, "rows", rec)
    print(t, len(rs), "rows", f"{time.time() - t0:.0f}s", flush=True)


def _fin(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _seed_sd(sel: Any, rows: list[Any], organisms: dict[str, str], tag: str) -> dict[str, Any]:
    """Seed-to-seed SD of each composite (seeds pooled by nothing): per seed, the composite over
    the evals the pooled reading averages, from a matrix of that seed's rows only; an eval with
    one seed is held at its pooled cell. Robustness is `paper_figures.seed_spread` (the shares
    kept recomputed per seed). `{key|axis: {per_seed, sd}}`."""
    import statistics

    import analysis.paper_figures as pf
    from analysis.aggregate import build_matrix

    m = sel.matrix
    seeds = sorted({r.seed for r in rows})
    out: dict[str, Any] = {}
    for ax in ("expression", "generality", "introspection"):
        bym = {s: build_matrix([r for r in rows if r.seed == s and r.axis == ax]) for s in seeds}
        for k, org in organisms.items():
            if ax == "expression":
                evs = [pf.SBE]
            else:
                rd = pf.base_reading(sel, tag, ax) if k == "base" else pf.reading(sel, org, ax)
                if rd.status != "ok":
                    continue
                evs = list(rd.evals)
            per = {}
            for s, ms in bym.items():
                cells = [ms.cells.get((org, e)) or m.cells.get((org, e)) for e in evs]
                vals = [c.mean if c is not None else math.nan for c in cells]
                if vals and all(math.isfinite(v) for v in vals):
                    per[str(s)] = statistics.fmean(vals)
            sd = statistics.stdev(per.values()) if len(per) > 1 else None
            out[f"{k}|{ax}"] = {"per_seed": per, "sd": sd}
    for k, org in organisms.items():
        ss = pf.seed_spread(sel, org, "robustness")
        if ss is not None:
            out[f"{k}|robustness"] = {
                "per_seed": {str(s): v for s, v in ss.per_seed.items()},
                "sd": _fin(ss.sd),
            }
    return out


def truth_cis(norm: Path, scratch: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    """`{spec_hash: {world|self: {delta, lo, hi}}}`: `analysis.truth_probe` (the paired bootstrap of
    the organism - base truth-probe delta) over a tab's truth cells, linked into `scratch` alone
    so the probe reads no other cell."""
    from analysis.truth_probe import analyse_tree

    shutil.rmtree(scratch, ignore_errors=True)
    for p in sorted(norm.glob("*/*/*whitebox_truth_*.parquet")):
        for f in (p, p.parent / "spec.json"):
            if not (q := scratch / f.relative_to(norm)).exists():
                _link(f, q)
    if not scratch.exists():
        return {}
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for group in analyse_tree(scratch).values():
        for h, res in (group.get("results") or {}).items():
            out[h] = {
                kind: {
                    "delta": v["rel"]["delta"],
                    "lo": v["rel"]["ci"][0],
                    "hi": v["rel"]["ci"][1],
                    # the random-direction null: 95th percentile of |shift| along 100 directions
                    "p95": (v.get("rand") or {}).get("p95_abs"),
                }
                for kind, v in res.items()
                if isinstance(v, dict) and isinstance(v.get("rel"), dict) and "ci" in v["rel"]
            }
    shutil.rmtree(scratch)
    return out


def reduce_rows(
    rows: list[Any],
    tag: str,
    require: list[str] | tuple[str, ...] = (),
    families: dict[str, frozenset[str]] | None = None,
    truth: dict[str, dict[str, dict[str, float | None]]] | None = None,
    tab: str | None = None,
    external: bool = False,
) -> dict[str, Any]:
    """The reduction of one tab's rows: its organism per method row, every axis reading, and the
    cells, deltas, verdicts and seed spreads the paper's tables print (`analysis.paper_tables`).

    Only `analysis.paper_figures`' own functions: `select` (which organism is each method row's),
    `reading` / `base_reading` (the axis composites), `delta_readings` (organism minus base on
    matched items) and `radar_value` (the one radar rule). `radar` is the paper-figure spoke
    (matched-item gain on the behaviour axes); `radar_unmatched` the same rule on the plain
    composites, which is what the dashboard's cell means can reproduce. `require` lists the
    method rows every tab must have (default: none); `truth` is `truth_cis` of the tab's tree.
    """
    import analysis.paper_figures as pf
    from analysis.radar_scale import radar_value

    where = f"tab {tab!r} ({tag})" if tab else tag
    try:
        sel = pf.select(rows, families, external)
    except ValueError as e:  # two organisms on one method row
        raise SystemExit(f"{where}: {e}") from None
    shown = set(sel.organism.values())
    extra = sorted({r.organism for r in rows} - shown)
    if extra:
        raise SystemExit(f"{where}: the tree holds organisms no method row shows: {extra}")
    keys = ["base", *(s.key for s in pf.METHODS), *(s.key for s in pf.CONTROLS)]
    keys += sorted({k for k, _ in sel.organism} - set(keys))  # external rows (opt-in)
    organisms = {k: o for k in keys if (o := sel.organism.get((k, tag)))}
    if "base" not in organisms:  # every reading is against it
        tags = sorted({t for _, t in sel.organism})
        raise SystemExit(
            f"{where}: no base organism with model tag {tag!r} (the rows carry {tags}); check the "
            "tab's `base` and `tag`, and that its runs hold the base model's `base` organism"
        )
    missing = [k for k in require if k not in organisms]
    if missing:
        raise SystemExit(f"{where}: no organism for the required method rows {missing}")
    readings = {}
    for k, org in organisms.items():
        for axis in (a for a in pf.COVERAGE_AXES if a != "expression"):
            base = pf.base_reading(sel, tag, axis)
            r = base if k == "base" else pf.reading(sel, org, axis)
            # the activations spoke needs BOTH truth voters for the organism and for the base
            voters = set(pf.TRUTH_PROBE_VOTERS)
            ok = r.status == "ok" and (
                axis not in pf.TRUTH_PROBE_AXES
                or (voters <= set(r.evals) and voters <= set(base.evals))
            )
            readings[f"{k}|{axis}"] = {
                "status": r.status,
                "delta": _fin(r.delta),
                "radar": _fin(radar_value(axis, r.gain_raw, r.gain_base_raw)) if ok else None,
                "radar_unmatched": _fin(radar_value(axis, r.raw, r.base_raw)) if ok else None,
                "raw": _fin(r.raw),
                "gain": _fin(r.gain_raw - r.gain_base_raw),
                "evals": list(r.evals),
            }
    ext = [
        k
        for k in keys
        if k not in ("base", *(s.key for s in pf.METHODS), *(s.key for s in pf.CONTROLS))
    ]
    dr = pf.delta_readings(sel, rows, (tag,), ext)
    cells = {
        f"{o}|{e}": {
            "mean": _fin(c.mean),
            "sem": _fin(c.sem),
            "n": c.n,
            "n_seeds": c.n_seeds,
            "n_pressure": c.kept.n_pressure if c.kept is not None else None,
        }
        for (o, e), c in sorted(sel.matrix.cells.items())
        if o in shown
    }
    sbe: dict[str, Any] = {}
    for k, org in organisms.items():
        v = sel.sbe_checks.get(org)
        paired = v.unseen.paired if v is not None and v.unseen is not None else None
        sbe[k] = {
            "verdict": sel.sbe_verdict.get(org),
            "unseen_paired": None
            if paired is None
            else {
                "diff": paired.diff,
                "lo": paired.interval[0],
                "hi": paired.interval[1],
                "n_items": paired.n_items,
            },
        }
    key_of = {o: k for k, o in organisms.items()}
    hashes = {r.spec_hash: key_of[r.organism] for r in rows if r.organism in key_of}
    return {
        "tag": tag,
        "organisms": organisms,
        "truth": {hashes[h]: v for h, v in sorted((truth or {}).items()) if h in hashes},
        "readings": readings,
        "delta_readings": {
            f"{k}|{a}": {"value": _fin(d.value), "se": _fin(d.se)} for (_, k, a), d in dr.items()
        },
        "cells": cells,
        "sbe": sbe,
        "seed_sd": _seed_sd(sel, rows, organisms, tag),
    }


def cmd_reduce(cfg: Config, t: str) -> None:
    rows = pickle.loads((cfg.root / "rows" / f"{t}.pkl").read_bytes())
    rows_rec = _read_prov(cfg, t, "rows")  # the revisions come from the rows' own record
    if not rows_rec:
        raise SystemExit(f"{t}: rows/{t}.pkl has no provenance record; run `rows`")
    tab = cfg.tabs[t]
    truth = truth_cis(cfg.root / "norm" / t, cfg.root / "truth_scratch" / t)
    out = reduce_rows(
        rows,
        tab["tag"],
        tab_require(cfg, t),
        cfg.families,
        truth,
        tab=t,
        external=cfg.external_rows,
    )
    out["revisions"] = rows_rec["norm"]["pins"]
    out["build"] = _canon({"rows": rows_rec, "require": tab_require(cfg, t)})
    from core.results import atomic_write

    with atomic_write(cfg.root / "tabs" / f"tab_{t}.json") as tmp:
        tmp.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    print(t, len(rows), "rows,", len(out["organisms"]), "organisms", flush=True)


def _per_tab(cfg: Config, stage: str, tabs: list[str]) -> None:
    """Run `rows` / `reduce` for each tab in its own process (one tab's rows are ~12 GB at peak)."""
    cli = REPO / "scripts" / "build_dashboard_page.py"
    for t in tabs:
        cmd = [sys.executable, "-I", "-B", str(cli), "--config", str(cfg.path), stage, t]
        if subprocess.run(cmd).returncode:  # the child has printed its own reason
            raise SystemExit(f"{stage} of tab {t!r} failed (see the message above)")


def _per_tab_stages(cfg: Config) -> None:
    """Rebuild what is stale (`_is_current`): each tab's tree, then its `rows`, then its `reduce`."""
    stale = [t for t in cfg.tabs if not _is_current(cfg, t, "norm")]
    if stale:
        cmd_tree(cfg, stale)
    for stage in ("rows", "reduce"):
        _per_tab(cfg, stage, [t for t in cfg.tabs if not _is_current(cfg, t, stage)])


def ensure_built(cfg: Config) -> None:
    """Make `<root>` hold a build of what the config asks for, doing only what is stale: `fetch`
    unless the downloads match the pins and hold every file the config selects, then per tab the
    tree, `rows` and `reduce` whose recorded inputs (`expected_record`) differ. Ends by verifying
    the pins."""
    try:
        verify_pins(cfg)
        missing = missing_downloads(cfg)
    except SystemExit:  # no record, or a record of other pins
        missing = ["pins"]
    if missing:
        cmd_fetch(cfg)
    verify_pins(cfg)
    _per_tab_stages(cfg)


# ------------------------------------------------------------------ the page's own arithmetic
def votes(c: dict[str, Any] | None) -> bool:
    return c is not None and bool(c) and c.get("votes") is not False


def axis_mean(data: dict[str, Any], org: str, ax: str, voter: str | None = None) -> float | None:
    """Python mirror of the template's axisMean (dumbbell): mean raw cell mean over the bucket's
    composite evals, counting an eval only where the voter's cell votes."""
    vc = data["cells"].get(voter or org, {})
    vs = [
        c["mean"]
        for e in data["bucket_evals"].get(ax, [])
        if (c := data["cells"].get(org, {}).get(e))
        and c.get("mean") is not None
        and votes(vc.get(e))
    ]
    return sum(vs) / len(vs) if vs else None


def tally(
    res: dict[str, Any], bucket: str, name: str, ours: float | None, ref: float | None, nd: int
) -> None:
    """Count one comparison into `res[bucket]`: a match within half a unit of the `nd`-th place."""
    if ours is None or ref is None or not math.isfinite(ref):
        return
    b = res.setdefault(bucket, {"checked": 0, "match": 0, "diff": []})
    b["checked"] += 1
    if abs(ours - ref) <= 0.5 * 10**-nd + 1e-9:
        b["match"] += 1
    else:
        b["diff"].append({"what": name, "dashboard": round(ours, nd + 3), "ref": ref})


def consistency_check(cfg: Config, data: dict[str, Any], m: Any) -> dict[str, Any]:
    """The page's numbers against the per-tab reduction (`tabs/tab_<tab>.json`), same quantity.

    Code-path consistency checks (the same rows through the same `radar_value`, so they catch
    plumbing drift, not a wrong rule): `radar_vs_reduction` (10 x the page's raw property means,
    health axes 1 + organism - base, against the reduction's radar value), `page_radar_vs_reduction`
    (the page's own paper-scale radar) and `delta_vs_reduction` (the forest's axis Δ).
    """
    from analysis.radar_scale import radar_value

    health = ("coherence", "capability")
    behaviour = ("generality", "introspection", "activations")
    res: dict[str, Any] = {
        k: {"checked": 0, "match": 0, "diff": []}
        for k in ("radar_vs_reduction", "delta_vs_reduction", "page_radar_vs_reduction")
    }
    for t in cfg.tabs:
        tab = tab_json(cfg, t)
        base = tab["organisms"]["base"]
        for k, org in tab["organisms"].items():
            for ax in (
                "generality",
                "robustness",
                "coherence",
                "capability",
                "activations",
                "introspection",
            ):
                rd = tab["readings"].get(f"{k}|{ax}")
                raw = axis_mean(data, org, ax)
                if raw is None:
                    ours = None
                elif ax in health or ax in behaviour:
                    # the ONE radar rule (analysis.radar_scale.radar_value). The dashboard has cell
                    # means only, so behaviour spokes are the UNMATCHED gain (the reduction's
                    # `radar_unmatched` is the same arithmetic).
                    b = axis_mean(data, base, ax, voter=org)
                    ours = radar_value(ax, raw, b if b is not None else math.nan)
                else:
                    ours = radar_value(ax, raw, math.nan)
                if ours is not None and not math.isfinite(ours):
                    ours = None
                if rd and rd["status"] == "ok":
                    ref = rd.get("radar_unmatched") if ax in behaviour else rd["radar"]
                    tally(res, "radar_vs_reduction", f"{t} {k} {ax} spoke", ours, ref, 4)
                    # the page's own paper-scale value (data["radar"], matched items) against the
                    # reduction's `radar` for the same organism and spoke
                    page_v = data.get("radar", {}).get("orgs", {}).get(org, {}).get(ax)
                    tally(
                        res,
                        "page_radar_vs_reduction",
                        f"{t} {k} {ax} radar",
                        page_v,
                        rd["radar"],
                        4,
                    )
                    if k != "base":
                        d = m.axis_delta(org, ax)
                        tally(
                            res,
                            "delta_vs_reduction",
                            f"{t} {k} {ax} delta",
                            d if math.isfinite(d) else None,
                            rd["delta"],
                            4,
                        )
    return res


def cmd_data(cfg: Config) -> None:
    import analysis.aggregate as agg
    import analysis.dashboard as dash
    import analysis.diagnostics as diag

    verify_pins(cfg)
    rows: list[Any] = []
    _per_tab_stages(cfg)
    parts = {t: pickle.loads((cfg.root / "rows" / f"{t}.pkl").read_bytes()) for t in cfg.tabs}
    mixed = {t: any(r.fake for r in part) for t, part in parts.items()}
    if len(set(mixed.values())) > 1:  # checked BEFORE the rows are combined
        fake_tabs = sorted(t for t, f in mixed.items() if f)
        raise SystemExit(
            f"REFUSING to combine fake and real tabs into one page: fake {fake_tabs}, real "
            f"{sorted(set(parts) - set(fake_tabs))}. A page is either plainly all fake or all real."
        )
    for t, part in parts.items():
        rows += part
        print(t, len(rows), "rows so far", flush=True)
    m = agg.build_matrix(rows)
    data = dash.build_dashboard_data(
        m,
        fake=any(r.fake for r in rows),
        rows=rows,
        families=cfg.families,
        external=cfg.external_rows,
    )
    # No results root (the rows came from several pickles), so the two cost tables are skipped.
    data["diagnostics"] = diag.build_diagnostics(rows, m)
    tx: dict[str, Any] = {"cap": cfg.tx_cap, "seed": 0, "orgs": {}, "excluded_evals": []}
    for t in cfg.tabs:
        part = json.loads((cfg.root / "rows" / f"{t}.tx.json").read_text(encoding="utf-8"))
        if part["cap"] != cfg.tx_cap or set(part["orgs"]) & set(tx["orgs"]):
            raise SystemExit(
                f"{t}: transcript part disagrees with the config cap or repeats an organism"
            )
        tx["orgs"].update(part["orgs"])
        tx["excluded_evals"] = sorted(set(tx["excluded_evals"]) | set(part["excluded_evals"]))
    if not set(tx["orgs"]) <= set(m.organisms):
        raise SystemExit("transcripts name organisms the Matrix does not have")
    data["transcripts"] = tx
    revs: dict[str, str] = {}
    for t in cfg.tabs:
        revs.update(tab_json(cfg, t)["revisions"])
    data["meta"]["hf_dataset"] = " · ".join(
        f"{r.split('/')[-1]}@{s[:10]}"
        for r, s in sorted(revs.items())
        if not r.endswith("-restricted")
    )
    if cfg.title is not None:
        data["meta"]["run_name"] = cfg.title
        data["meta"]["run_date"] = ""
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    build: dict[str, Any] = {
        "revisions": revs,
        "mu_sidecar": cfg.mu_sidecar,
        "code": {"analysis": f"{REPO} {head}"},
    }
    ti = cfg.root / "tree_info.json"
    if ti.exists():
        build["tree"] = json.loads(ti.read_text(encoding="utf-8"))
    xc = consistency_check(cfg, data, m)
    if cfg.paper is not None:
        from analysis.paper_checks import tex_checks

        xc["tex"] = tex_checks(cfg)
    build["cross_check"] = xc
    print(
        json.dumps(
            {
                k: {kk: (vv if kk != "diff" else vv[:12]) for kk, vv in v.items()}
                for k, v in xc.items()
                if k != "tex"
            }
            | {
                f"tex {lab}": {
                    kk: (vv if kk not in ("diff", "missing") else vv[:6]) for kk, vv in v.items()
                }
                for lab, v in xc.get("tex", {}).items()
            },
            indent=1,
        )
    )
    data["_build"] = build
    cfg.data_json.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print(
        "organisms",
        data["meta"]["n_organisms"],
        "treated",
        data["meta"]["n_treated"],
        "evals",
        data["meta"]["n_evals"],
        "base models",
        data["meta"]["base_models"],
        flush=True,
    )


def _page_data(cfg: Config) -> dict[str, Any]:
    data = json.loads(cfg.data_json.read_text(encoding="utf-8"))
    if cfg.paper is not None and "cross_check" in data.get("_build", {}):
        from analysis.paper_checks import caveats

        data["caveats"] = caveats(cfg, data)
    return data


# `_build` keys that hold the build machine's absolute paths (`code` is "<local checkout path>
# <git sha>"); everything else in `_build` (revisions, tree, cross_check) is publishable
# provenance and stays in the embedded data.
_BUILD_PRIVATE_KEYS = ("code",)


def publishable_build(build: dict[str, Any]) -> dict[str, Any]:
    """`_build` minus local paths (`code`) and the private `-restricted` companion repos, which the
    page's snapshot line already hides (analysis/dashboard.py, fail closed on the name)."""
    out = {k: v for k, v in build.items() if k not in _BUILD_PRIVATE_KEYS}
    rev = out.get("revisions")
    if isinstance(rev, dict):
        out["revisions"] = {r: s for r, s in rev.items() if not str(r).endswith("-restricted")}
    elif isinstance(rev, list):
        out["revisions"] = [x for x in rev if "-restricted" not in json.dumps(x)]
    return out


def _page_data_public(cfg: Config) -> dict[str, Any]:
    import analysis.dashboard as dash

    data = _page_data(cfg)
    if isinstance(data.get("_build"), dict):
        data["_build"] = publishable_build(data["_build"])
        dash.attach_dataset_links(data)
    return data


def cmd_html(cfg: Config) -> None:
    import analysis.dashboard as dash

    data = _page_data_public(cfg)
    html = dash.render_html(data)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(html, encoding="utf-8")
    print("wrote", cfg.out, f"{len(html.encode()) / 1e6:.2f} MB")


_DATA_RE = re.compile(r'(<script type="application/json" id="data">)(.*?)(</script>)', re.S)
_TX_RE = re.compile(r'(<script type="text/plain" id="transcripts">)(.*?)(</script>)', re.S)


def _without_build(blob: str) -> str:
    """The blob minus `_build` and minus the statistics the renderer derives from it
    (`analysis.dashboard_stats`), so a page rendered before those existed can still be refreshed."""
    from analysis.dashboard_stats import strip_display_stats

    data = strip_display_stats(json.loads(blob))
    return json.dumps({k: v for k, v in data.items() if k != "_build"}, separators=(",", ":"))


def render_site_from_page(
    page_html: str, build: dict[str, Any] | None = None, *, allow_d_radars: bool = False
) -> str:
    """Re-render a dashboard page through the site path with its own embedded data.

    The data and transcripts blocks come back byte-identical (the transcripts are re-inserted
    verbatim, not re-packed); only the template and the site patches/footer are current.
    `build` restores `_build` provenance into a page that lacks it (private keys stripped);
    the cell data must still be identical once `_build` is removed from both sides.
    """
    import analysis.dashboard as dash

    dm, tm = _DATA_RE.search(page_html), _TX_RE.search(page_html)
    if dm is None or tm is None:
        raise SystemExit("page has no embedded data/transcripts block")
    data = json.loads(dm.group(2))
    # Always sanitise: a page's own embedded `_build` (e.g. an internal render) may carry the
    # checkout path and the -restricted repos, and this page is public.
    if build is not None:
        data["_build"] = publishable_build(build)
    elif isinstance(data.get("_build"), dict):
        data["_build"] = publishable_build(data["_build"])
    out = dash.site_dashboard_html(dash.render_html(data), data, allow_d_radars=allow_d_radars)
    if not (m := _TX_RE.search(out)) or m.group(2):
        raise SystemExit("rendered page has an unexpected transcripts block")
    out = _TX_RE.sub(lambda m: m.group(1) + tm.group(2) + m.group(3), out, count=1)
    new = _DATA_RE.search(out)
    if new is None or _without_build(new.group(2)) != _without_build(dm.group(2)):
        raise SystemExit("data block changed on re-render; refusing to write")
    return out


def cmd_site(
    cfg: Config,
    site_out: Path,
    from_page: Path | None,
    build_from: Path | None = None,
    *,
    allow_d_radars: bool = False,
) -> None:
    import analysis.dashboard as dash

    if from_page is not None:
        build = None
        if build_from is not None:
            build = json.loads(build_from.read_text(encoding="utf-8"))["_build"]
        html = render_site_from_page(
            from_page.read_text(encoding="utf-8"), build, allow_d_radars=allow_d_radars
        )
    else:
        data = _page_data_public(cfg)
        html = dash.site_dashboard_html(
            dash.render_html(data), data, allow_d_radars=allow_d_radars
        )
    site_out.parent.mkdir(parents=True, exist_ok=True)
    site_out.write_text(html, encoding="utf-8")
    print("wrote", site_out, f"{len(html.encode()) / 1e6:.2f} MB")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "what",
        nargs="?",
        default="all",
        choices=["fetch", "tree", "rows", "reduce", "data", "html", "site", "all"],
    )
    ap.add_argument("tab", nargs="?", help="`rows` / `reduce`: one tab (default: every tab)")
    ap.add_argument(
        "--site-out", type=Path, default=REPO / "site" / "public" / "dashboard" / "index.html"
    )
    ap.add_argument("--from-page", type=Path, help="`site`: re-render from this page's own data")
    ap.add_argument(
        "--build-from",
        type=Path,
        help="`site --from-page`: restore `_build` from this dashboard_data.json",
    )
    ap.add_argument(
        "--allow-d-radars",
        action="store_true",
        help="`site`: accept data without the paper-scale `radar` block (Δ-vs-base radars). Else refuse.",
    )
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    cfg = Config(args.config)
    cfg.add_output(args.site_out)
    if args.what in ("rows", "reduce"):
        if args.tab is None:
            _per_tab(cfg, args.what, list(cfg.tabs))
        elif args.tab not in cfg.tabs:
            ap.error(f"unknown tab {args.tab!r}; the config has {list(cfg.tabs)}")
        else:
            (cmd_rows if args.what == "rows" else cmd_reduce)(cfg, args.tab)
        return
    if args.what == "site":
        cmd_site(
            cfg, args.site_out, args.from_page, args.build_from, allow_d_radars=args.allow_d_radars
        )
        return
    if args.what == "fetch":
        cmd_fetch(cfg)
        return
    if args.what in ("tree", "all"):
        cmd_tree(cfg)
    if args.what in ("data", "all"):
        cmd_data(cfg)
    if args.what in ("html", "all"):
        cmd_html(cfg)


if __name__ == "__main__":
    main()
