"""The `mobench` user config: adapters, properties and seeds in; a spec file out.

A user config is the small file someone writes to test LoRA adapters they did not train here:

    organisms:
      - adapter: <owner>/<repo>            # or ...@<40-hex commit>, or a huggingface.co URL
        label: my_label          # optional; default: the repo name
        base_model: <hf id>      # optional; default: the adapter's adapter_config.json
    properties: [misalignment, capability]
    seeds: [0, 1, 2]

It is told apart from a spec file by its `organisms:` being a LIST (a spec file's is a mapping
of persona id to organisms). `desugar` turns it into an ordinary spec file: every eval setting
comes from the repo's `configs/specs.yaml` (its top-level keys other than organisms, specs,
defaults and `default_evals`), the organisms become `specs:` rows, the base model of each
distinct base is added once, and the properties become `default_evals`. Everything after that
(the dry run, `scripts/run_all.sh`, the cache) is the path a spec file takes.

An adapter without a commit is pinned automatically: its current head commit is resolved
once and recorded in `.mobench-pins.json` next to the config, so `validate`, `cost` and `run`
(and reruns) use the same commit until `--refresh-pins` (or MOBENCH_REFRESH_PINS=1) re-resolves.

Every refusal is a `UserConfigError` raised before anything runs, naming the key to fix.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from core.flag_values import suggestion as _nearest

REPO = Path(__file__).resolve().parent.parent
REPO_DEFAULTS = REPO / "configs" / "specs.yaml"
PERSONA_PAIRS = REPO / "data" / "eval_prompts" / "persona_pairs.yaml"
DEFAULT_SEEDS = [0, 1, 2]
TOP_KEYS = frozenset(
    {
        "organisms",
        "properties",
        "seeds",
        "behaviour",
        "include_base_model",
        "results_root",
        "evals",
    }
)
ROW_KEYS = frozenset({"adapter", "label", "base_model", "subfolder"})
# The properties whose evals read a behaviour definition (a persona pair, a trait phrase,
# generated banks, a truth-probe bank). Measured: a fake run of every registered eval on a
# persona with no definition writes rows for exactly the other three
# (tests/test_mobench_userconfig.py::test_behaviour_free_axes_build).
BEHAVIOUR_FREE_AXES = frozenset({"capability", "coherence", "misalignment"})
LABEL_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
LABEL_MAX = 40
SHA_RE = re.compile(r"[0-9a-f]{40}")
# A Hub repo id: <owner>/<name>, letters, digits, `.`, `_`, `-` (no spaces, `@` or newlines).
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
# Method rows of the report (`analysis.paper_style.STYLE`, pinned by a test): a label equal to
# one would be drawn as that row.
REPORT_ROW_KEYS = frozenset(
    {"base", "oct", "sft", "dpo", "steering", "seqkd", "sysprompt", "antiprompt", "icl"}
)
# Top-level keys of configs/specs.yaml that are not eval settings.
NOT_INHERITED = frozenset({"organisms", "specs", "defaults", "default_evals", "seeds"})


class UserConfigError(ValueError):
    """A user config that cannot be run as written."""


@dataclass(frozen=True)
class AdapterInfo:
    """What `adapter_config.json` says, or why it could not be read."""

    base_model: str | None = None
    rank: int | None = None
    error: str | None = None


@dataclass
class ExternalRow:
    label: str
    repo: str
    revision: str
    subfolder: str | None
    base_model: str
    base_source: str  # "config" (the row's `base_model:`) or "adapter" (adapter_config.json)
    adapter_says: str | None  # adapter_config.json's base, when it was read
    rank: int | None
    pin_note: str | None = None  # set when the commit was resolved here, not given


@dataclass
class UserPlan:
    """A desugared user config: the spec file (`config`) and what the plan prints."""

    path: str
    behaviour: str | None
    persona_id: str
    rows: list[ExternalRow]
    bases: list[str]  # base rows added, one per distinct base
    properties: list[str]
    evals: list[str]
    skipped: list[tuple[str, str]]  # (eval, reason) of registered evals that will not run
    seeds: list[int]
    results_root: str | None
    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    pin_lines: list[str] = field(default_factory=list)  # one line per auto-pinned adapter


def is_user_config(data: Any) -> bool:
    """A user config has a LIST under `organisms:`; a spec file has a mapping there."""
    return isinstance(data, Mapping) and isinstance(data.get("organisms"), list)


def read_adapter_config(repo: str, revision: str, subfolder: str | None) -> AdapterInfo:
    """`adapter_config.json` at the pinned revision: the base it names and the LoRA rank.

    One free Hub read (the token is `HF_TOKEN` or the cached login, as `huggingface_hub`
    finds it). Any failure is returned, not raised: the caller decides whether it matters.
    """
    import json

    from huggingface_hub import hf_hub_download

    name = f"{subfolder}/adapter_config.json" if subfolder else "adapter_config.json"
    try:
        path = hf_hub_download(repo, name, revision=revision)
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001  # any failure means "could not read"; say which
        return AdapterInfo(error=f"{type(exc).__name__}")
    rank = cfg.get("r")
    return AdapterInfo(
        base_model=cfg.get("base_model_name_or_path") or None,
        rank=rank if isinstance(rank, int) else None,
    )


def known_behaviours() -> list[str]:
    """Behaviours the evals can score: those with a persona pair."""
    data = yaml.safe_load(PERSONA_PAIRS.read_text(encoding="utf-8")) or {}
    return sorted((data.get("pairs") or {}).keys())


def registered_evals() -> dict[str, str]:
    """eval name -> its property (axis), from the registry."""
    import serving.inspect_runner  # noqa: F401  # registers every eval
    from evals._inspect import all_evals

    return {e.name: e.axis for e in all_evals()}


def reserved_labels() -> frozenset[str]:
    from serving.resolve import METHOD_BACKENDS

    return frozenset(METHOD_BACKENDS) | REPORT_ROW_KEYS


def _suggest(word: str, options: Sequence[str]) -> str:
    return _nearest(word, list(options))


def default_label(repo: str) -> str:
    """The repo name, lowercased, every character a label may not hold made `-`."""
    name = re.sub(r"[^a-z0-9_-]+", "-", repo.split("/")[-1].lower()).strip("-")
    return re.sub(r"_{2,}", "_", name)


def check_label(label: Any, where: str) -> str:
    if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
        raise UserConfigError(
            f"{where}: label {label!r} must match {LABEL_RE.pattern} (lowercase letters, digits, "
            "`_` and `-`, starting with a letter)"
        )
    if "__" in label:
        raise UserConfigError(f"{where}: label {label!r} may not contain `__`")
    if len(label) > LABEL_MAX:
        raise UserConfigError(f"{where}: label {label!r} is over {LABEL_MAX} characters")
    if label in reserved_labels():
        raise UserConfigError(
            f"{where}: label {label!r} is a method or report row name already "
            f"({', '.join(sorted(reserved_labels()))}); pick another"
        )
    return label


def parse_adapter(value: Any, where: str) -> tuple[str, str | None]:
    """`parse_adapter_full` for a reference that cannot carry a subfolder; one that does is
    refused, never silently stripped."""
    repo, ref, sub = parse_adapter_full(value, where)
    if sub:
        raise UserConfigError(
            f"{where}: adapter {value!r} names the subfolder {sub!r}, not allowed here"
        )
    return repo, ref


def parse_adapter_full(value: Any, where: str) -> tuple[str, str | None, str | None]:
    """An adapter reference -> (repo, ref, subfolder). Accepted: `<owner>/<repo>`,
    `<owner>/<repo>@<40-hex commit>` and a huggingface.co URL (`.../<owner>/<repo>`, optionally
    `/tree/<rev>` or `/commit/<sha>`; `/tree/<rev>/<subfolder>` also gives the subfolder).
    `ref` is a 40-hex commit, a revision name from a `/tree/` URL, or None (no pin given:
    `PinStore.pin` resolves the head commit)."""
    if not isinstance(value, str) or not value:
        raise UserConfigError(
            f"{where}: adapter {value!r} must be <owner>/<repo>, <owner>/<repo>@<40-hex commit> "
            "or a huggingface.co URL"
        )
    if "://" in value or value.startswith(("huggingface.co/", "www.huggingface.co/")):
        url = urlparse(value if "://" in value else f"https://{value}")
        if url.netloc not in ("huggingface.co", "www.huggingface.co"):
            raise UserConfigError(f"{where}: adapter URL {value!r} is not a huggingface.co URL")
        parts = [p for p in url.path.split("/") if p]
        repo = "/".join(parts[:2])
        if not REPO_RE.fullmatch(repo):
            raise UserConfigError(f"{where}: adapter URL {value!r} must name <owner>/<repo>")
        rest = parts[2:]
        if not rest:
            return repo, None, None
        if rest[0] in ("tree", "commit", "blob", "resolve") and len(rest) >= 2:
            ref = rest[1]
            if rest[0] == "commit" and not SHA_RE.fullmatch(ref):
                raise UserConfigError(
                    f"{where}: adapter URL {value!r}: /commit/ needs the full 40-character hash"
                )
            sub = "/".join(rest[2:]) or None
            if sub and rest[0] != "tree":
                raise UserConfigError(
                    f"{where}: adapter URL {value!r} must end at the repo, "
                    "/tree/<rev>[/<subfolder>] or /commit/<sha>"
                )
            return repo, ref, sub
        raise UserConfigError(
            f"{where}: adapter URL {value!r} must end at the repo, /tree/<rev> or /commit/<sha>"
        )
    if "@" not in value:
        if not REPO_RE.fullmatch(value):
            raise UserConfigError(f"{where}: adapter repo {value!r} must be <owner>/<repo>")
        return value, None, None
    repo, _, sha = value.rpartition("@")
    if not REPO_RE.fullmatch(repo):
        raise UserConfigError(f"{where}: adapter repo {repo!r} must be <owner>/<repo>")
    if not SHA_RE.fullmatch(sha):
        raise UserConfigError(
            f"{where}: adapter pin {sha!r} must be a full 40-character lowercase commit hash, "
            "not a branch, tag or short hash (or drop `@...` and the latest commit is pinned)"
        )
    return repo, sha, None


PINS_FILE = ".mobench-pins.json"


def resolve_head(repo: str, ref: str | None = None) -> tuple[str, str]:
    """(commit sha, commit date) of `repo`'s head (or of `ref`), one Hub read. The token is
    `HF_TOKEN` when set, else the cached login, as `huggingface_hub` finds it."""
    from huggingface_hub import HfApi

    commits = HfApi(token=os.environ.get("HF_TOKEN") or None).list_repo_commits(repo, revision=ref)
    if not commits:
        raise ValueError("no commits")
    return commits[0].commit_id, commits[0].created_at.date().isoformat()


class PinStore:
    """The pin record: repo -> {sha, date}, in `.mobench-pins.json` next to the config.

    Next to the config (not under the results root) because the config alone decides the
    location: `validate`, `cost` and `run` find the same file whatever `--results-root` says,
    and it can be committed with the config. Only auto-resolved pins are recorded."""

    def __init__(
        self,
        path: Path,
        *,
        online: bool,
        refresh: bool,
        resolver: Callable[..., tuple[str, str]] = resolve_head,
    ) -> None:
        self.path = path
        self.online = online
        self.refresh = refresh
        self.resolver = resolver
        self.dirty = False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.data: dict[str, dict[str, str]] = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            self.data = {}

    def pin(self, repo: str, ref: str | None, where: str) -> tuple[str, str]:
        """(sha, note) for an adapter with no 40-hex pin given."""
        key = repo if ref is None else f"{repo}@{ref}"
        rec = self.data.get(key)
        if rec and not self.refresh and SHA_RE.fullmatch(str(rec.get("sha"))):
            return rec["sha"], f"pinned {repo} to {rec['sha']} (reused from {self.path.name})"
        if not self.online:
            raise UserConfigError(
                f"{where}: adapter {repo} has no commit pin, and offline (fake or --offline) "
                f"the latest commit cannot be looked up. Give {repo}@<40-hex commit>, or run "
                f"online once so {PINS_FILE} records it"
            )
        try:
            sha, date = self.resolver(repo, ref)
        except Exception as exc:  # noqa: BLE001  # any Hub failure is a refusal naming the repo
            raise UserConfigError(
                f"{where}: could not look up the latest commit of {repo} "
                f"({type(exc).__name__}: {exc}); check the name and HF_TOKEN, or give "
                f"{repo}@<40-hex commit>"
            ) from None
        self.data[key] = {"sha": sha, "date": date}
        self.dirty = True
        what = f"revision {ref}" if ref else "latest commit"
        return sha, f"pinned {repo} to {sha} ({what}, {date})"

    def save(self) -> None:
        if not self.dirty:
            return
        try:
            self.path.write_text(
                json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise UserConfigError(f"cannot write {self.path}: {exc}") from None
        self.dirty = False


def _seeds(data: Mapping[str, Any], where: str) -> list[int]:
    seeds = data.get("seeds", DEFAULT_SEEDS)
    if not (
        isinstance(seeds, list)
        and seeds
        and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
        and len(set(seeds)) == len(seeds)
    ):
        raise UserConfigError(f"{where}: seeds must be a non-empty list of distinct ints")
    return seeds


def _properties(
    data: Mapping[str, Any], behaviour: str | None, axes: frozenset[str], where: str
) -> list[str]:
    raw = data.get("properties")
    if raw is None:
        return sorted(axes if behaviour else axes & BEHAVIOUR_FREE_AXES)
    if not isinstance(raw, list) or not raw or not all(isinstance(p, str) for p in raw):
        raise UserConfigError(f"{where}: properties must be a non-empty list of property names")
    for p in raw:
        if p not in axes:
            raise UserConfigError(
                f"{where}: unknown property {p!r}.{_suggest(p, sorted(axes))} "
                f"Properties: {', '.join(sorted(axes))}."
            )
    if behaviour is None:
        needs = [p for p in raw if p not in BEHAVIOUR_FREE_AXES]
        if needs:
            raise UserConfigError(
                f"{where}: properties {needs} need a behaviour definition, and this config has "
                "no `behaviour:`. Add `behaviour: <id>` (one of: "
                f"{', '.join(known_behaviours())}), or drop them; without one only "
                f"{', '.join(sorted(BEHAVIOUR_FREE_AXES))} can run."
            )
    return sorted(dict.fromkeys(raw))


def _evals(
    data: Mapping[str, Any],
    properties: list[str],
    defaults: list[str],
    axis_of: Mapping[str, str],
    behaviour: str | None,
    where: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    """The evals to run, in the repo's `default_evals` order, and the registered evals that
    will not, each with its reason."""
    narrow = data.get("evals") or {}
    if not isinstance(narrow, Mapping):
        raise UserConfigError(f"{where}: evals must map a property to {{include|exclude: [...]}}")
    chosen: dict[str, set[str]] = {}
    for prop in properties:
        chosen[prop] = {e for e in defaults if axis_of.get(e) == prop}
    reasons: dict[str, str] = {}
    for prop, rule in narrow.items():
        if prop not in properties:
            raise UserConfigError(
                f"{where}: evals.{prop}: {prop!r} is not one of this config's properties "
                f"({', '.join(properties)})"
            )
        if not isinstance(rule, Mapping) or not set(rule) <= {"include", "exclude"} or not rule:
            raise UserConfigError(
                f"{where}: evals.{prop} must be {{include: [...]}} or {{exclude: [...]}}"
            )
        if "include" in rule and "exclude" in rule:
            raise UserConfigError(f"{where}: evals.{prop} has both include and exclude; use one")
        kind = "include" if "include" in rule else "exclude"
        names = rule[kind]
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise UserConfigError(
                f"{where}: evals.{prop}.{kind} must be a non-empty list of evals"
            )
        for name in names:
            if name not in axis_of:
                raise UserConfigError(
                    f"{where}: evals.{prop}.{kind}: no eval {name!r}."
                    f"{_suggest(name, sorted(axis_of))}"
                )
            if axis_of[name] != prop:
                raise UserConfigError(
                    f"{where}: evals.{prop}.{kind}: {name!r} is a {axis_of[name]} eval, not "
                    f"{prop}. Did you mean `evals: {{{axis_of[name]}: {{{kind}: [{name}]}}}}`?"
                )
        if kind == "include":
            for e in chosen[prop] - set(names):
                reasons[e] = f"not in evals.{prop}.include"
            chosen[prop] = set(names)
        else:
            for e in set(names):
                reasons[e] = f"evals.{prop}.exclude"
            chosen[prop] -= set(names)
    picked = {e for names in chosen.values() for e in names}
    order = [e for e in defaults if e in picked] + sorted(picked - set(defaults))
    for name, axis in axis_of.items():
        if name in picked or name in reasons:
            continue
        if axis not in properties:
            if behaviour is None and axis not in BEHAVIOUR_FREE_AXES:
                reasons[name] = f"{axis} needs a behaviour definition (no `behaviour:`)"
            else:
                reasons[name] = f"property {axis} not selected"
        else:
            reasons[name] = "not in the repo's default_evals (name it under `include` to run it)"
    return order, sorted(reasons.items())


def _row(
    entry: Any,
    i: int,
    where: str,
    *,
    online: bool,
    reader: Callable[..., AdapterInfo],
    pins: PinStore,
) -> ExternalRow:
    at = f"{where}: organisms[{i}]"
    if not isinstance(entry, Mapping):
        raise UserConfigError(f"{at} must be a mapping with an `adapter:` key")
    unknown = sorted(set(entry) - ROW_KEYS)
    if unknown:
        raise UserConfigError(
            f"{at}: unknown key(s) {unknown}; a row takes {', '.join(sorted(ROW_KEYS))}"
            + (" (training from a config is not supported yet)" if "train" in unknown else "")
        )
    if "adapter" not in entry:
        raise UserConfigError(f"{at}: no `adapter:` (<owner>/<repo>)")
    repo, sha, url_sub = parse_adapter_full(entry["adapter"], at)
    pin_note = None
    if sha is None or not SHA_RE.fullmatch(sha):
        sha, pin_note = pins.pin(repo, sha, at)
    subfolder = entry.get("subfolder")
    if url_sub:  # the URL's /tree/<rev>/<subfolder>; a different `subfolder:` beside it is a clash
        if subfolder is not None and str(subfolder).strip("/") != url_sub.strip("/"):
            raise UserConfigError(
                f"{at}: the adapter URL names subfolder {url_sub!r} but `subfolder:` says "
                f"{subfolder!r}; give one"
            )
        subfolder = url_sub
    if subfolder is not None and (not isinstance(subfolder, str) or not subfolder.strip("/")):
        raise UserConfigError(f"{at}: subfolder must be a path inside the repo")
    label = entry.get("label")
    if label is None:
        label = default_label(repo)
        try:
            check_label(label, at)
        except UserConfigError as exc:
            raise UserConfigError(f"{exc} (derived from the repo name; set `label:`)") from None
    check_label(label, at)
    stated = entry.get("base_model")
    if stated is not None and (not isinstance(stated, str) or stated.count("/") != 1):
        raise UserConfigError(f"{at}: base_model {stated!r} must be a Hub id <owner>/<model>")
    info = reader(repo, sha, subfolder) if online else AdapterInfo(error="offline")
    if stated is None:
        if not online:
            raise UserConfigError(
                f"{at}: offline (fake or --offline), so adapter_config.json cannot be read for "
                f"the base model; give `base_model:` on the row"
            )
        if not info.base_model:
            raise UserConfigError(
                f"{at}: could not read the base model from {repo}@{sha[:12]} adapter_config.json "
                f"({info.error or 'no base_model_name_or_path'}); give `base_model:` on the row"
            )
    if info.rank is not None:
        from serving.resolve import serving_config

        cap = serving_config().max_lora_rank
        if info.rank > cap:
            raise UserConfigError(
                f"{at}: LoRA rank {info.rank} is over the serving cap max_lora_rank {cap} "
                "(configs/serving.yaml)"
            )
    return ExternalRow(
        label=label,
        repo=repo,
        revision=sha,
        subfolder=subfolder.strip("/") if subfolder else None,
        base_model=stated or str(info.base_model),
        base_source="config" if stated else "adapter",
        adapter_says=info.base_model,
        rank=info.rank,
        pin_note=pin_note,
    )


def repo_eval_settings(path: Path = REPO_DEFAULTS) -> tuple[dict[str, Any], list[str]]:
    """The eval settings a user config inherits, and the repo's `default_evals`."""
    from serving.runner import load_config

    data = load_config(path)
    settings = {
        k: v for k, v in data.items() if k not in NOT_INHERITED and not str(k).startswith("_")
    }
    return settings, list(data.get("default_evals") or [])


def desugar(
    path: str | Path,
    data: Mapping[str, Any] | None = None,
    *,
    online: bool,
    reader: Callable[..., AdapterInfo] | None = None,
    repo_defaults: Path = REPO_DEFAULTS,
    resolver: Callable[..., tuple[str, str]] | None = None,
    refresh_pins: bool | None = None,
    say: Callable[[str], Any] | None = print,
) -> UserPlan:
    """A user config as a spec file (`UserPlan.config`), or `UserConfigError`. `reader` reads
    an adapter's `adapter_config.json` (default `read_adapter_config`, one Hub read; only when
    `online`)."""
    from core.registry import VALID_AXES
    from serving.resolve import EXTERNAL_PROVENANCE, NO_BEHAVIOUR

    where = str(path)
    if data is None:
        from serving.runner import load_config

        data = load_config(path)
    if not is_user_config(data):
        raise UserConfigError(f"{where}: not a user config (`organisms:` must be a list)")
    unknown = sorted(set(data) - TOP_KEYS)
    if unknown:
        hint = " (training from a config is not supported yet)" if "train" in unknown else ""
        raise UserConfigError(
            f"{where}: unknown key(s) {unknown}{hint}. A user config takes "
            f"{', '.join(sorted(TOP_KEYS))}; eval settings come from configs/specs.yaml."
        )
    if not data["organisms"]:
        raise UserConfigError(f"{where}: organisms is empty; nothing to run")
    behaviour = data.get("behaviour")
    if behaviour is not None:
        have = known_behaviours()
        if behaviour not in have:
            raise UserConfigError(
                f"{where}: behaviour {behaviour!r} has no definition in this repo."
                f"{_suggest(str(behaviour), have)} Defined: {', '.join(have)}. A behaviour is "
                "defined by its persona pair (data/eval_prompts/persona_pairs.yaml), registry "
                "file (data/behaviours/<id>/config.yaml) and generated banks "
                "(data/eval_prompts/generality/generated/<id>/); "
                "docs/methods/adding-a-behaviour.md says how to add one. Or drop `behaviour:` "
                f"to run only {', '.join(sorted(BEHAVIOUR_FREE_AXES))}."
            )
    include_base = data.get("include_base_model", True)
    if not isinstance(include_base, bool):
        raise UserConfigError(f"{where}: include_base_model must be true or false")
    results_root = data.get("results_root")
    if results_root is not None and not isinstance(results_root, str):
        raise UserConfigError(f"{where}: results_root must be a path")
    seeds = _seeds(data, where)
    properties = _properties(data, behaviour, frozenset(VALID_AXES), where)
    settings, defaults = repo_eval_settings(repo_defaults)
    axis_of = registered_evals()
    evals, skipped = _evals(data, properties, defaults, axis_of, behaviour, where)
    if not evals:
        raise UserConfigError(f"{where}: the properties and evals selected leave no eval to run")
    read = reader or read_adapter_config
    refresh = (
        refresh_pins
        if refresh_pins is not None
        else os.environ.get("MOBENCH_REFRESH_PINS", "").strip().lower()
        not in ("", "0", "false", "no", "off", "n", "f")
    )
    pins = PinStore(
        Path(path).resolve().parent / PINS_FILE,
        online=online,
        refresh=refresh,
        resolver=resolver or resolve_head,
    )
    rows = [
        _row(e, i, where, online=online, reader=read, pins=pins)
        for i, e in enumerate(data["organisms"])
    ]
    pins.save()
    pin_lines = list(dict.fromkeys(r.pin_note for r in rows if r.pin_note))
    if say:
        for line in pin_lines:
            say(line)
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.label, row.base_model)
        ref = f"{row.repo}@{row.revision}" + (f"/{row.subfolder}" if row.subfolder else "")
        if key in seen:
            raise UserConfigError(
                f"{where}: label {row.label!r} names two adapters on {row.base_model} "
                f"({seen[key]} and {ref}); a label is one row per base, so give each its own"
            )
        seen[key] = ref
    persona = behaviour or NO_BEHAVIOUR
    specs: list[dict[str, Any]] = []
    for row in rows:
        mc: dict[str, Any] = {
            "provenance": EXTERNAL_PROVENANCE,
            "adapter_repo": row.repo,
            "revision": row.revision,
        }
        if row.subfolder:
            mc["subfolder"] = row.subfolder
        specs.append(
            {
                "persona_id": persona,
                "base_model": row.base_model,
                "method": row.label,
                "method_config": mc,
            }  # fmt: skip
        )
    bases = list(dict.fromkeys(r.base_model for r in rows)) if include_base else []
    # Written out in full, as `serving.runner._flat_entries` flattens a spec file's base row,
    # so the same base row from configs/specs.yaml merges with it instead of conflicting.
    specs += [
        {
            "persona_id": persona,
            "base_model": b,
            "method": "base",
            "method_config": {},
            "train_scale": 1.0,
            "behavior_class": "persona_trait",
        }  # fmt: skip
        for b in bases
    ]
    config = {**settings, "seeds": seeds, "specs": specs, "default_evals": evals}
    plan = UserPlan(
        path=where,
        behaviour=behaviour,
        persona_id=persona,
        rows=rows,
        bases=bases,
        properties=properties,
        evals=evals,
        skipped=skipped,
        seeds=seeds,
        results_root=results_root,
        config=config,
        pin_lines=pin_lines,
    )
    if not include_base:
        plan.notes.append(
            "include_base_model: false: no base rows are added; every score is against the "
            "base, so the page needs base rows from an earlier run in the same results root"
        )
    return plan


def plan_lines(plan: UserPlan) -> list[str]:
    """The config half of the plan `run` prints first: rows and their bases, the base rows, the
    evals and the skipped evals by reason (`mobench.cli.print_plan` adds the cell counts)."""
    out = [f"plan for {plan.path}:"]
    out.append(
        f"  behaviour: {plan.behaviour or f'none (persona id {plan.persona_id!r})'}; seeds "
        f"{plan.seeds}; properties: {', '.join(plan.properties)}"
    )
    for row in plan.rows:
        ref = f"{row.repo}@{row.revision[:12]}" + (f"/{row.subfolder}" if row.subfolder else "")
        if row.base_source == "adapter":
            base = f"{row.base_model} (adapter_config.json)"
        elif row.adapter_says and row.adapter_says != row.base_model:
            base = f"{row.base_model} (config) — adapter says {row.adapter_says}"
        elif row.adapter_says:
            base = f"{row.base_model} (config; adapter agrees)"
        else:
            base = f"{row.base_model} (config; adapter_config.json not read)"
        rank = f", LoRA r{row.rank}" if row.rank is not None else ""
        auto = " [auto-pinned]" if row.pin_note else ""
        out.append(f"  row {row.label} (external): {ref}{auto}{rank}; base: {base}")
    for base in plan.bases:
        out.append(f"  base row: {plan.persona_id}/base on {base}")
    for note in plan.notes:
        out.append(f"  note: {note}")
    out.append(f"  evals ({len(plan.evals)}): {', '.join(plan.evals)}")
    if plan.skipped:
        out.append(f"  skipped evals ({len(plan.skipped)}), by reason:")
        by_reason: dict[str, list[str]] = {}
        for name, why in plan.skipped:
            by_reason.setdefault(why, []).append(name)
        out += [f"    {why}: {', '.join(names)}" for why, names in sorted(by_reason.items())]
    return out


def write_spec_file(plan: UserPlan, out_dir: Path) -> Path:
    """Write the desugared spec file; its name carries the source file's stem."""
    import hashlib
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(plan.config, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{plan.path}\n{body}".encode()).hexdigest()[:10]
    path = out_dir / f"{Path(plan.path).stem}.{digest}.specs.yaml"
    path.write_text(
        f"# Generated by mobench from {plan.path}; do not edit (edit that file).\n"
        + yaml.safe_dump(plan.config, sort_keys=False),
        encoding="utf-8",
    )
    return path
