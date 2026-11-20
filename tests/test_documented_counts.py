"""The counts in `CLAUDE.md`, `README.md`, `docs/COMMANDS.md` and the catalogues match the code.

Its own file rather than a corner of another module, for the reason
`tests/test_committed_artifacts.py` gives: this is a repo-wide invariant over two
coordination documents, not a property of any one module, and homed inside an unrelated
test file it would be deleted with that file and never found by whoever changes a count
next.

**Why this exists.** The same defect has landed three times, and twice it was invisible:

1. Two branches independently rewrote the status doc's eval heading to "36 registered"
   for different reasons. Git auto-merged the identical six characters with no conflict
   and the number was silently wrong.
2. `CLAUDE.md` claimed "42 `@register` calls across 11 modules" and prescribed
   `grep -c '@register' evals/*.py` for checking it. That grep is wrong in BOTH
   directions — it counts two docstrings in `misaligned.py` that merely discuss the
   decorator, and misses `capability_gpqa.py`'s two programmatic registrations — so +2
   and −2 landed on the right total for compensating wrong reasons and survived
   indefinitely.
3. The status doc headed its organism section "27 registered" while its own table on the
   next line summed to 23, four rows having been deregistered on 2026-08-28. A downstream
   paragraph had then done *arithmetic* on the wrong total.

The status doc already carried the written lesson from (1) — "when editing an eval count
here, count it from the code, never from the previous sentence" — five sections above the
heading that drifted in (3). A lesson recorded against one number does not generalise
itself to its neighbour, and prose does not enforce itself. This does.

The status doc was deleted on 2026-09-30: its eval half moved into `docs/EVAL-CATALOGUE.md`,
its method and organism half into `docs/IMPLANT-CATALOGUE.md`, and the three incidents above
are history. The pins that read it now read the file each sentence moved to.

**What it does NOT cover, deliberately.** The implant-method count. No document states one
any more: `docs/IMPLANT-CATALOGUE.md` lists the methods by state and prints the count from
`METHOD_BACKENDS`, so there is no sentence to drift.

**When this fails,** read the failure before editing anything. If the count moved because
you added an eval or a spec, update the documents. If a pattern stopped matching because
you reworded the sentence, update the pattern here — a reworded sentence carrying a count
is exactly the edit that should make you look.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _registering_module(ev: object) -> str:
    """The `evals/_task_*.py` module whose `register_task` call put this eval in the registry.

    Reached through `builder` because that is the only handle `RegisteredEval` keeps on the
    calling module. `capability_gpqa`'s pair registers a `functools.partial`, whose
    `__module__` is `functools` — unwrap it or the two land in a phantom eleventh module.
    """
    fn = ev.builder
    while hasattr(fn, "func"):
        fn = fn.func
    return str(fn.__module__)


def _counts() -> dict[str, int]:
    """The true numbers, from the registry and the spec loader — never from the text."""
    import serving.inspect_runner  # noqa: F401  # side effect: imports every evals/_task_*
    from core.registry import VALID_AXES
    from evals._inspect import all_evals, cache_gated_evals
    from serving.runner import load_config, load_specs

    # No purge-and-reload dance here any more: since the `@register` registry was deleted
    # (2026-08-29) an eval is registered by the import side effect of its `evals/_task_*.py`
    # module, and those are `_`-prefixed infrastructure that must NOT be dropped from
    # `sys.modules` to be re-imported — it splits module identity
    # (`evals._inspect.registry_snapshot`, `tests/conftest.purgeable_eval_module`). The
    # import above is idempotent and the registry it fills is process-global.
    # Cache-gated stock cells are subtracted BY NAME, not by whether they registered:
    # `mmlu_anchor` and friends register only where their upstream dataset is already
    # cached, so an unsubtracted count would say 46 on CI and 50 on a machine that has
    # done the setup, and every documented number would go red on the second one.
    # `cache_gated_evals()` is filled by the gate call in BOTH directions, which is what
    # makes one verdict possible on either machine.
    evals = [e for e in all_evals() if e.name not in cache_gated_evals()]
    specs = load_specs("configs/specs.yaml")
    per_persona = Counter(s.persona_id for s in specs)
    swept = set(load_config("configs/specs.yaml")["default_evals"])
    return {
        "evals": len(evals),
        # Registered AND in `default_evals`: the two DEPRECATED misalignment cells are the
        # difference. A registered eval missing from the sweep must be named in the docs.
        "swept_evals": len(swept & {e.name for e in evals}),
        "eval_modules": len({_registering_module(e) for e in evals}),
        "organisms": len(specs),
        "smoke_organisms": len(load_specs("configs/specs.smoke.yaml")),
        "impulsive_rows": per_persona["impulsive"],
        "personas": len(per_persona),
        "axes": len(VALID_AXES),
    }


# (document, key into _counts(), pattern with ONE capture group).
# Patterns are matched against the document with all runs of whitespace collapsed to a
# single space, so a count that straddles a line wrap still matches and re-wrapping a
# paragraph does not break the test.
_SITES: list[tuple[str, str, str]] = [
    ("CLAUDE.md", "evals", r"\*\*Evals:\*\* (\d+) registered across"),
    ("CLAUDE.md", "eval_modules", r"registered across (\d+) modules in `evals/`"),
    ("CLAUDE.md", "eval_modules", r"evals/ # (\d+) modules,"),
    ("CLAUDE.md", "evals", r"# \d+ modules, (\d+) registered evals"),
    ("CLAUDE.md", "organisms", r"\*\*Organisms:\*\* (\d+) in `configs/specs\.yaml`"),
    ("CLAUDE.md", "smoke_organisms", r"(\d+) in `configs/specs\.smoke\.yaml`"),
    ("CLAUDE.md", "smoke_organisms", r"the gate's (\d+) organisms"),
    ("docs/COMMANDS.md", "swept_evals", r"full (\d+)-eval × \d+-organism matrix"),
    (
        "docs/COMMANDS.md",
        "swept_evals",
        r"(\d+) of the \d+ registered evals are in `default_evals`",
    ),
    ("docs/COMMANDS.md", "evals", r"\d+ of the (\d+) registered evals are in `default_evals`"),
    ("docs/COMMANDS.md", "organisms", r"full \d+-eval × (\d+)-organism matrix"),
    ("CLAUDE.md", "axes", r"\*\*Axes:\*\* (\d+)"),
    ("README.md", "organisms", r"`mmlu`, for all (\d+) organisms"),
    ("README.md", "organisms", r"show (\d+) near-identical \"tasks\" per eval"),
    ("README.md", "organisms", r"would delete all (\d+) logs"),
    ("docs/IMPLANT-CATALOGUE.md", "organisms", r"## Registered organisms — (\d+)"),
    ("docs/EVAL-CATALOGUE.md", "evals", r"(\d+) registered evals across \w+ buckets"),
    (
        "docs/EVAL-CATALOGUE.md",
        "evals",
        r"(\d+) is the registry, and \*\*\d+ of them run on every sweep",
    ),
    (
        "docs/EVAL-CATALOGUE.md",
        "swept_evals",
        r"\*\*(\d+) of them run on every sweep",
    ),
    ("docs/EVAL-CATALOGUE.md", "evals", r"--list` prints the same (\d+) names"),
    ("docs/IMPLANT-CATALOGUE.md", "impulsive_rows", r"(\d+) impulsive rows, on `Qwen"),
    ("docs/IMPLANT-CATALOGUE.md", "personas", r"(\d+) personas? ha(?:s|ve) organisms"),
]


@pytest.fixture(scope="module")
def documents() -> dict[str, str]:
    """Each document with whitespace collapsed, so a wrapped count still matches."""
    return {
        name: re.sub(r"\s+", " ", (_ROOT / name).read_text(encoding="utf-8"))
        for name in {doc for doc, _, _ in _SITES}
    }


def test_every_documented_count_matches_the_code(documents: dict[str, str]) -> None:
    truth = _counts()
    # State the denominator: an empty site list would make this pass vacuously, and
    # a green check that has run out of things to check looks identical to a real one.
    assert len(_SITES) >= 20, f"only {len(_SITES)} documented counts under test"
    assert truth["evals"] > 0 and truth["organisms"] > 0, f"nothing loaded: {truth}"

    for doc, key, pattern in _SITES:
        found = re.findall(pattern, documents[doc])
        assert found, (
            f"{doc}: no match for /{pattern}/. If you reworded that sentence, update the "
            f"pattern in {Path(__file__).name}; if you deleted the count, delete the row. "
            "A sentence carrying a count is exactly the edit worth looking at twice."
        )
        for got in found:
            assert int(got) == truth[key], (
                f"{doc} says {got} where the code reports {truth[key]} ({key}). "
                f"Pattern: /{pattern}/. Fix the DOCUMENT, not this test — the numbers here "
                "come from evals._inspect.all_evals and serving.runner.load_specs."
            )


def test_claude_md_still_prescribes_the_registry_commands() -> None:
    """CLAUDE.md must keep pointing at the registry and the loader, not at grep.

    Narrow on purpose, and named for what it does: it checks the two commands are still
    PRESCRIBED, not that they return anything — the test above is what checks the numbers.
    The point is the regression it blocks: the previous prescription
    (`grep -c '@register' evals/*.py`) was wrong in both directions and agreed with the
    documented total only by cancellation, so reverting to it would leave the counts
    correct today and unverifiable tomorrow.
    """
    claude_md = (_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for snippet in (
        "import serving.inspect_runner; from evals._inspect import all_evals; "
        "print(len(all_evals()))",
        "from serving.runner import load_specs; print(len(load_specs('configs/specs.yaml')))",
    ):
        assert snippet in claude_md, f"CLAUDE.md no longer prescribes: {snippet}"


_NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def test_the_catalogue_bucket_count_matches_its_own_sections() -> None:
    """ "N buckets" in the catalogue intro must equal the `## N.` sections below it.

    This one has no code source — a "bucket" IS a numbered section of that file — so it is
    checked against the document's own structure. Not circular: the intro sentence and the
    section headings are two independently edited places, and they disagreed ("eight
    buckets" against nine sections) from whenever the ninth bucket landed until 2026-08-31.
    """
    text = (_ROOT / "docs/EVAL-CATALOGUE.md").read_text(encoding="utf-8")
    sections = re.findall(r"^## (\d+)\. ", text, flags=re.MULTILINE)
    assert sections, "no `## N.` bucket sections found in docs/EVAL-CATALOGUE.md"
    stated = re.search(r"registered evals across (\w+) buckets", text)
    assert stated, "docs/EVAL-CATALOGUE.md no longer states a bucket count in its intro"
    assert stated.group(1) == _NUMBER_WORDS[len(sections)], (
        f"docs/EVAL-CATALOGUE.md says {stated.group(1)!r} buckets but carries "
        f"{len(sections)} `## N.` sections. Fix the document."
    )


def test_every_registering_module_is_named_in_claude_mds_tree() -> None:
    """CLAUDE.md must name every module that registers an eval.

    `capability_wmdp.py` was missing until 2026-08-31, so the tree listed 12 modules whose
    per-module numbers summed to 42 — directly beneath a header saying 13 and 43, and
    beneath a comment asserting the numbers came from the registry. A count that is right
    and a list that is short disagree with each other silently, because nobody adds up an
    ASCII tree.
    """
    from evals._inspect import all_evals

    _counts()  # populates the registry
    # Scoped to the fenced layout block, and with each line's `#` comment stripped. A bare
    # substring search over the whole file passed a mutation that DELETED the module from
    # the list, because the prose explaining that fix names the file too — the test would
    # have been satisfied by its own excuse.
    body = (_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    blocks = [b for b in body.split("```") if "\nimplant/" in b and "\nevals/" in b]
    assert len(blocks) == 1, f"expected exactly one repo-layout code block, found {len(blocks)}"
    # Drop only the pure `#   ...` continuation lines. Splitting every line at its first
    # `#` would also drop the SECOND column — the layout puts two modules per line.
    paths = "\n".join(line for line in blocks[0].splitlines() if not line.lstrip().startswith("#"))
    modules = {_registering_module(e).rpartition(".")[2] + ".py" for e in all_evals()}
    assert modules, "no evals registered"
    missing = sorted(m for m in modules if m not in paths)
    assert not missing, (
        f"CLAUDE.md's layout block does not list: {missing}. Naming it only in a comment "
        "does not count — the per-module numbers are read off this list."
    )


def test_every_registered_eval_has_a_row_in_the_catalogue() -> None:
    """docs/EVAL-CATALOGUE.md is titled "every eval, in plain terms" and `--list` points at it.

    `wmdp` was registered in #153 and never added, so the document the tool sends readers to
    silently omitted an eval — and a reader reconciling 43-from-the-tool against 42-in-the-doc
    had no way to find out which one was missing, because it was not named anywhere.
    """
    from evals._inspect import all_evals

    _counts()
    # A table ROW, not a mention. Searching the whole document passed a mutation that
    # deleted the `wmdp` row, because the note explaining why `wmdp` had been missing
    # mentions it by name — the test would have been satisfied by its own excuse.
    # And a row of a BUCKET table (header `| Eval | Registered | ...`), not of any table:
    # the "how to read" tables also open rows with an eval name, and on 2026-09-30 they
    # would have hidden the deletion of twelve evals' real rows.
    cat = (_ROOT / "docs/EVAL-CATALOGUE.md").read_text(encoding="utf-8")
    rows: set[str] = set()
    in_bucket_table = False
    for line in cat.splitlines():
        if not line.startswith("|"):
            in_bucket_table = False
        elif line.startswith("| Eval | Registered |"):
            in_bucket_table = True
        elif in_bucket_table and (m := re.match(r"\| `([^`]+)`", line)):
            rows.add(m.group(1))
    names = [e.name for e in all_evals()]
    assert names, "no evals registered"
    assert rows, "no eval rows found in docs/EVAL-CATALOGUE.md"
    missing = sorted(n for n in names if n not in rows)
    assert not missing, (
        f"docs/EVAL-CATALOGUE.md has no table row for: {missing}. A mention in prose is not "
        'a row — this file is titled "every eval, in plain terms" and `--list` points at it.'
    )


def test_the_per_method_organism_table_matches_the_loader() -> None:
    """The catalogue's organism table is a per-method count kept by hand, and it drifted.

    On 2026-09-03 (then in the status doc) it summed to 23 while its heading said 25: PR
    #199 updated the heading and not the table, so `sft_behaviour` read 1 where the loader
    reports 2 and `dpo_behaviour` had no row at all. Nothing failed, because `_SITES` above
    pins the headings and the prose — the scalar TOTALS — and never the rows those totals are a sum
    of. A total can stay right while the breakdown under it is wrong.

    Same shape as any hand-kept method list: a list of methods maintained by hand beside
    the code drifts, and the missing entry is the direction that hurts, because an absent
    row reads as "this method has no organisms" rather than as an error.

    Both directions, so a row left behind by the next method deletion trips this too. A row
    reading 0 is exempt: a method may be dispatched but have no registered organism.
    """
    from serving.runner import load_specs

    truth = Counter(spec.method for spec in load_specs(_ROOT / "configs/specs.yaml"))
    header = "| Method | n | Per persona | train_scale | Artifact source |"
    doc = "docs/IMPLANT-CATALOGUE.md"
    catalogue = (_ROOT / doc).read_text(encoding="utf-8")
    assert header in catalogue, f"organism table header not found in {doc}: {header!r}"
    table = catalogue.split(header, 1)[1].split("\n\n", 1)[0]
    pairs = re.findall(r"^\|\s*`([a-z_]+)`[^|]*\|\s*(\d+)\s*\|", table, flags=re.MULTILINE)
    assert pairs, f"no organism rows parsed out of {doc}"
    methods = [method for method, _ in pairs]
    assert len(methods) == len(set(methods)), f"{doc} lists a method twice: {sorted(methods)}"
    rows = dict(pairs)

    documented = {method: int(n) for method, n in rows.items() if int(n)}
    assert documented == dict(truth), (
        f"{doc}'s organism table disagrees with configs/specs.yaml — table says "
        f"{dict(sorted(documented.items()))}, loader says {dict(sorted(truth.items()))}"
    )


_SIZES = ("7b", "14b", "32b")
_SIZE = re.compile(r"(?<![0-9a-z])(7b|14b|32b)(?![0-9a-z])")
_FAMILY = "{7b,14b,32b}"
_ORG = "science-of-model-organisms/"


_LIVE_STATE = re.compile(
    r"\*\*(?:CURRENT|REGISTERED, NO ARTIFACT)\*\*(?: \(registered \d{4}-\d{2}-\d{2}\))?"
)


def _named(text: str, name: str) -> bool:
    """`name` appears in `text` as a whole repo or folder name, not inside a longer one."""
    return re.search(rf"(?<![\w./-]){re.escape(name)}(?![\w./:@-])", text) is not None


def _live_version_rows(catalogue: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    """(persona, method) -> the versions-table rows whose State cell IS a live state.

    Read cell by cell: the persona is the backticked `###` heading the table sits under,
    the method is the row's first cell, and the State cell must be exactly `**CURRENT**`
    or `**REGISTERED, NO ARTIFACT**` (optionally with a `(registered <date>)`), so
    `**CURRENT** → DEPRECATED` is not live. Each row is returned as {column: cell}.
    """
    section = catalogue.split("## Dataset and artifact versions", 1)
    assert len(section) == 2, "no 'Dataset and artifact versions' section"
    rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    persona, header = None, None
    for line in section[1].split("\n## ", 1)[0].splitlines():
        if line.startswith("### "):
            heading = re.fullmatch(r"### `([a-z_]+)`", line.strip())
            persona, header = (heading.group(1) if heading else None), None
        elif not line.startswith("|"):
            header = None
        elif persona:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] == "Method":
                missing = {"State", "Training data", "Adapters"} - set(cells)
                assert not missing, f"the `{persona}` versions table has no {missing} column"
                header = cells
            elif header is not None and set(cells[0]) != {"-"}:
                assert len(cells) == len(header), (
                    f"a `{persona}` versions-table row has {len(cells)} cells, its header "
                    f"{len(header)} (a `|` inside a cell?): {line[:80]}"
                )
                row = dict(zip(header, cells, strict=True))
                method = re.fullmatch(r"`([a-z_]+)`", row["Method"])
                if method and _LIVE_STATE.fullmatch(row["State"]):
                    rows.setdefault((persona, method.group(1)), []).append(row)
    return rows


def test_every_registered_artifact_and_dataset_is_named_in_the_implant_catalogue() -> None:
    """Every repo a registered organism trains on or serves has a live row in the catalogue.

    `docs/IMPLANT-CATALOGUE.md` is where a dataset or adapter version is marked CURRENT or
    DEPRECATED. That record is only usable if it is complete: on 2026-09-30 the `impulsive`
    SFT/DPO rows had moved to the v3 GLM data while the organism table still described v2,
    and nobody reading the docs could tell which dataset a registered organism came from.

    For each registered organism, every adapter and vector repo it names (stage adapters
    included) must appear in the **Adapters** cell, and every dataset repo in the
    **Training data** cell, of a versions-table row for ITS persona and ITS method whose
    State cell is exactly a live state. Each clause closes a hole three adversarial passes
    found: a row relabelled `**CURRENT** → DEPRECATED`; a name filed under another method or
    persona, or mentioned only in Notes as "replaced `x`"; a name that is a prefix of a
    longer one (`x-data` inside `x-data-deprecated`, `x:old`); the family spelling
    `{7b,14b,32b}` for a 7B-only artifact.

    Not covered: the prose in those cells (row counts, teacher, hyper-parameters), a name
    negated inside its own cell ("not `x`"), and the steering `subfolder`, which is a bare
    persona word.
    """
    from serving.runner import load_specs

    doc = "docs/IMPLANT-CATALOGUE.md"
    live = _live_version_rows((_ROOT / doc).read_text(encoding="utf-8"))
    assert sum(len(rows) for rows in live.values()) >= 7, f"only {live.keys()} live rows in {doc}"

    # (persona, method, column, repo, folder or ""). A folder only counts beside its repo.
    wanted: set[tuple[str, str, str, str, str]] = set()
    registered_pairs: set[tuple[str, str]] = set()
    for spec in load_specs(_ROOT / "configs/specs.yaml"):
        config = spec.method_config
        key = (spec.persona_id, spec.method)
        registered_pairs.add(key)
        adapter = str(config.get("adapter_uri") or config.get("adapter_repo") or "")
        if adapter:
            wanted.add((*key, "Adapters", adapter, str(config.get("subfolder") or "")))
        for stage in ("dpo", "sft"):
            if config.get(f"{stage}_adapter_uri"):
                wanted.add((*key, "Adapters", str(config[f"{stage}_adapter_uri"]), ""))
            if config.get(f"{stage}_adapter_subfolder"):
                folder = str(config[f"{stage}_adapter_subfolder"])
                wanted.add((*key, "Adapters", adapter, folder))
        if config.get("vector_uri"):
            wanted.add((*key, "Adapters", str(config["vector_uri"]), ""))
        for field in ("dataset_uri", "data_uri"):
            if config.get(field):
                repo = str(config[field]).split(" :: ")[0].strip()
                wanted.add((*key, "Training data", repo, ""))
    # 25, not 26: the v4 SFT/DPO rows (2026-10-02) all name the one canonical training-data repo
    # (organism-training-data), so DPO 7B no longer adds maius/OpenCharacterTraining-data.
    # A floor, not an exact count: it guards against the extraction above silently finding nothing.
    assert len(wanted) >= 25, f"only {len(wanted)} registered repos found: {sorted(wanted)}"
    registered = {n.removeprefix(_ORG) for *_, repo, folder in wanted for n in (repo, folder)}

    def spellings(name: str) -> list[str]:
        """The ways the catalogue may write `name`: bare or org-prefixed, literal or family."""
        bare = name.removeprefix(_ORG)
        forms = [bare]
        family = _SIZE.sub(_FAMILY, bare)
        if family != bare and all(family.replace(_FAMILY, z) in registered for z in _SIZES):
            forms.append(family)
        # the org prefix is optional, but only on a repo that is in the org
        return forms + [_ORG + form for form in forms] if name.startswith(_ORG) else forms

    def documented(persona: str, method: str, column: str, repo: str, folder: str) -> bool:
        for row in live.get((persona, method), []):
            cell = row[column]
            if any(_named(cell, r) for r in spellings(repo)) and (
                not folder or any(_named(cell, f) for f in spellings(folder))
            ):
                return True
        return False

    missing = sorted(
        f"{persona}/{method} [{column}]: {repo} {folder}".rstrip()
        for persona, method, column, repo, folder in wanted
        if not documented(persona, method, column, repo, folder)
    )
    assert not missing, (
        f"{doc} has no CURRENT row, under the right persona and method, naming in the right "
        f"cell: {missing}. Add it to the 'Dataset and artifact versions' table and mark what "
        "it supersedes DEPRECATED."
    )

    # The converse: a live row must describe something that IS registered. Without it the
    # incident above recurs with the test green — the v2 row relabelled CURRENT beside v3, a
    # disabled method's row marked CURRENT, or a live row naming an adapter no spec uses.
    stale = []
    for (persona, method), rows in live.items():
        mine = [w for w in wanted if w[:3] == (persona, method, "Adapters")]
        if not any(spec[:2] == (persona, method) for spec in registered_pairs):
            stale.append(f"{persona}/{method}: a live row, but no registered organism")
            continue
        names = [n for *_, repo, folder in mine for n in (repo, folder)]
        known = {form for n in names for form in spellings(n)}
        for row in rows:
            for token in re.findall(r"`([^`]+)`", row["Adapters"]):
                if token not in known:
                    stale.append(f"{persona}/{method}: `{token}` is in a CURRENT row's Adapters")
    assert not stale, (
        f"{doc} marks as CURRENT something configs/specs.yaml does not register: {stale}. "
        "Mark the row DEPRECATED (or UNREGISTERED), or move the name out of the Adapters cell."
    )


# --- APPS / impossible-bank per-band sample counts -----------------------------------
# Same defect class as the counts above, fourth occurrence. #297 landed
# `eval_configs.apps.levels` as 400/200/100 while its own prose said 500/400/100, so four
# sites overstated the real sweep (700) as 1000 — a 43% error, invisible because nothing
# tied the sentence to the config. Two cells state a per-band split in the same words in
# the same files (`apps`, and `misaligned_reward_hacking`'s impossible bank), so this
# pins the SHAPE: every band triple in these documents must be one of the two real ones.
# Rewording-proof by construction — it never has to know which sentence is which cell.

_BAND_TRIPLE = re.compile(
    r"(\d+) introductory [+/] (\d+) interview [+/] (\d+) competition", re.IGNORECASE
)
_BAND_DOCS = ("docs/EVAL-CATALOGUE.md", "analysis/eval_meta.yaml")


def _apps_levels() -> dict[str, int]:
    """`eval_configs.apps.levels` from the sweep config — the counts a real run reads."""
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "specs.yaml").read_text(encoding="utf-8"))
    levels = cfg["eval_configs"]["apps"]["levels"]
    assert set(levels) == {"introductory", "interview", "competition"}, levels
    return {k: int(v) for k, v in levels.items()}


def _bands(d: dict[str, int]) -> tuple[int, int, int]:
    return (d["introductory"], d["interview"], d["competition"])


def test_every_documented_band_split_matches_a_real_one() -> None:
    from scripts.build_impossible_bank import _APPS_BANK_LEVELS

    apps, bank = _bands(_apps_levels()), _bands(_APPS_BANK_LEVELS)
    allowed = {apps: "eval_configs.apps.levels", bank: "_APPS_BANK_LEVELS"}
    assert apps != bank, "the two splits collided; this test can no longer tell them apart"

    found = 0
    for name in _BAND_DOCS:
        text = re.sub(r"\s+", " ", (_ROOT / name).read_text(encoding="utf-8"))
        for m in _BAND_TRIPLE.finditer(text):
            triple = tuple(int(g) for g in m.groups())
            found += 1
            assert triple in allowed, (
                f"{name} documents a per-band split of {triple}, which is neither the "
                f"apps sweep {apps} (configs/specs.yaml eval_configs.apps.levels) nor the "
                f"impossible bank {bank} (scripts/build_impossible_bank._APPS_BANK_LEVELS). "
                f"Fix the DOCUMENT, not this test. Context: ...{m.group(0)}..."
            )
    # State the denominator: a reworded sentence that stops matching would make this
    # pass vacuously, and a check that has run out of things to check looks green.
    assert found >= 5, f"only {found} band splits under test across {_BAND_DOCS}"


def test_the_apps_default_levels_match_the_sweep_config() -> None:
    """`docs/EVAL-CATALOGUE.md` quotes `_APPS_LEVELS` as the default and the config as the
    sweep; the two sentences only stay honest together while the values agree."""
    from evals._task_capability import _APPS_LEVELS

    assert dict(_APPS_LEVELS) == _apps_levels(), (
        f"evals/_task_capability._APPS_LEVELS is {dict(_APPS_LEVELS)} but "
        f"configs/specs.yaml eval_configs.apps.levels is {_apps_levels()}."
    )
