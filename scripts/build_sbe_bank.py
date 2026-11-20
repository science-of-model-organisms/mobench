"""Build the `simple_behaviour_exhibit` eval's question bank: T1 (unseen) + T2 (seen), tagged.

Design: ``docs/evals/expression.md``.
Config: ``data/eval_prompts/simple_behaviour_exhibit/<persona>.sources.yaml``.

    uv run python scripts/build_sbe_bank.py --persona impulsive

Reads every registered organism's training files from HF, so it needs network and read
access to the private ``oct-behaviour-data`` repo. The output bank is committed and
contains no text from a private file: the T2 constitution questions come from the vendored
(public) OCT constitution, the ICL questions from the committed demos, the SeqKD prompts
from a public dataset. Private files are only read to compute ``trained_on`` tags.

What it does:

1. Collects each organism's training prompts, per the config's ``training_data``, and
   fails if a registered organism of a trained method in ``configs/specs.yaml`` has no entry.
2. T1 = the existing ``diag_v1`` questions + the authored ``diag_v2`` ones. Every one is
   leak-checked against every training prompt (system and user turns), every constitution
   question, and the persona files listed under ``leak_only`` in the config, exact (after
   normalising) and near-duplicate (``judges.backends.content_words`` Jaccard >=
   ``LEAK_JACCARD``). Any leak fails the build — fix
   the question, never loosen the check.
3. T2 = a seeded sample of constitution ``additional_questions`` that every SFT/DPO/OCT
   organism trained on, all ICL demo questions (with their position, so ``analysis/`` knows
   which are in a k-shot prompt), and a seeded sample of SeqKD's own prompts.
4. Tags every item with ``trained_on``: the methods whose EVERY registered organism trained
   on it. In some organisms of a method but not all is an error (a method-level tag would be
   false for one size).

Evals never import this (CI boundary); it writes a YAML bank that ``evals._harness.load_bank``
reads like any other.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from judges.backends import content_words  # noqa: E402

log = logging.getLogger("build_sbe_bank")

# Two questions sharing half their content words are the same question for contamination
# purposes.
LEAK_JACCARD = 0.5


def normalise(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace — the exact-match key."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def row_prompts(row: Mapping[str, Any]) -> list[str]:
    """The prompt-side turns (system AND user) of one training row, whichever shape it has.

    ``messages`` (SFT/introspection views), ``chosen`` as a message list (the maius DPO
    release), or a bare ``prompt`` (our DPO pairs, ``glm_chosen``). System turns count: the
    model trained with that text in its prompt, so a question placed there is seen, not
    unseen (adversarial review of PR #339, item 1). A row with none of these raises rather
    than contributing nothing, because a silently empty prompt set would tag every
    question as unseen.
    """
    for key in ("messages", "chosen"):
        turns = row.get(key)
        if isinstance(turns, list):
            prompt_side = [t["content"] for t in turns if t.get("role") in ("system", "user")]
            if any(t.get("role") == "user" for t in turns):
                return prompt_side
    if isinstance(row.get("prompt"), str):
        return [row["prompt"]]
    raise ValueError(f"training row has no user turn in messages/chosen/prompt: {list(row)}")


def jsonl_prompts(path: Path) -> list[str]:
    """Every user turn in a JSONL training file."""
    out: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.extend(row_prompts(json.loads(line)))
    return out


def local_prompts(path: Path) -> list[str]:
    """User prompts in one of the persona's own files (json bank, RL prompts, ICL demos)."""
    if path.suffix == ".jsonl":
        return jsonl_prompts(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" in data:
        return [i["prompt"] for i in data["items"]]
    return [q["question"] if isinstance(q, dict) else q for q in data["questions"]]


def trained_on(question: str, organisms: Mapping[str, Mapping[str, set[str]]]) -> list[str]:
    """Methods for which EVERY registered organism's prompt set contains ``question``.

    ``organisms`` is method -> base model -> set of normalised prompts. Present for some
    organisms of a method but not all raises: the tag is per method, so it would be wrong
    for the sizes that did not see the question.
    """
    key = normalise(question)
    methods = []
    for method, by_model in sorted(organisms.items()):
        hits = {model for model, prompts in by_model.items() if key in prompts}
        if hits and hits != set(by_model):
            raise ValueError(
                f"{question!r} is in {method}'s data for {sorted(hits)} but not "
                f"{sorted(set(by_model) - hits)}; a method-level tag would be false for one size"
            )
        if hits:
            methods.append(method)
    return methods


def leaks(question: str, exact: set[str], near: Iterable[tuple[set[str], str]]) -> str | None:
    """The source prompt ``question`` duplicates (exactly or nearly), or ``None``."""
    if normalise(question) in exact:
        return "exact match"
    words = content_words(question)
    if not words:
        return None
    for other, text in near:
        if other and len(words & other) / len(words | other) >= LEAK_JACCARD:
            return f"near-duplicate of {text!r}"
    return None


def trait_of(question: str, constitution: list[dict[str, Any]]) -> int:
    """Index of the constitution trait whose ``questions`` include ``question``."""
    for idx, trait in enumerate(constitution):
        if question in trait["questions"]:
            return idx
    raise ValueError(f"{question!r} is not a constitution question")


def seeded_sample(pool: list[str], n: int, rng: random.Random, what: str) -> list[str]:
    """``n`` items from ``pool`` in a seed-determined order; sorted first so input order
    (a dict, a file) cannot change the draw."""
    unique = sorted(set(pool))
    if len(unique) < n:
        raise ValueError(f"{what}: need {n} candidates, have {len(unique)}")
    return rng.sample(unique, n)


def build_items(
    *,
    t1: list[dict[str, Any]],
    constitution_rows: list[dict[str, Any]],
    constitution_field: str,
    constitution_per_trait: int,
    constitution_requires: list[str],
    icl_questions: list[str],
    persona_constitution: list[dict[str, Any]],
    seqkd_prompts: list[str],
    seqkd_count: int,
    excluded: set[str],
    organisms: Mapping[str, Mapping[str, set[str]]],
    seed: int,
    id_prefix: str = "sbe",
) -> list[dict[str, Any]]:
    """Assemble the tagged T1 + T2 items. Pure: every input is already loaded.

    ``t1`` rows carry ``question``, ``trait_idx``, ``origin`` (and optionally
    ``authored_by``/``authored_on``); leak-checking them is the caller's job, done before
    this so a leak fails before anything is sampled.
    """
    rng = random.Random(seed)
    excluded_keys = {normalise(q) for q in excluded}
    items: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(item_id: str, question: str, tier: str, pool: str, **meta: Any) -> None:
        key = normalise(question)
        if key in excluded_keys:
            raise ValueError(f"{item_id}: {question!r} is an excluded (indist_bank) question")
        if key in used:
            raise ValueError(f"{item_id}: {question!r} is already in the bank")
        used.add(key)
        extra = {k: meta.pop(k) for k in ("authored_by", "authored_on") if k in meta}
        tags = trained_on(question, organisms)
        if tier == "t1" and tags:
            raise ValueError(f"{item_id}: T1 question {question!r} is in {tags}'s training data")
        items.append(
            {
                "id": item_id,
                "question": question,
                "source": pool,
                **extra,
                "meta": {"tier": tier, "pool": pool, "trained_on": tags, **meta},
            }
        )

    per_trait = Counter()
    for row in t1:
        per_trait[row["trait_idx"]] += 1
        add(
            f"{id_prefix}_t1_{row['trait_idx']}_{per_trait[row['trait_idx']]:02d}",
            row["question"],
            "t1",
            row["origin"],
            trait_idx=row["trait_idx"],
            **{k: row[k] for k in ("authored_by", "authored_on") if k in row},
        )

    for idx, trait in enumerate(constitution_rows):
        pool = [
            q
            for q in trait[constitution_field]
            if normalise(q) not in excluded_keys
            and set(constitution_requires) <= set(trained_on(q, organisms))
        ]
        for n, question in enumerate(
            seeded_sample(pool, constitution_per_trait, rng, f"constitution trait {idx}"), 1
        ):
            add(f"{id_prefix}_con_{idx}_{n:02d}", question, "t2", "constitution", trait_idx=idx)

    for position, question in enumerate(icl_questions, 1):
        add(
            f"{id_prefix}_icl_{position:02d}",
            question,
            "t2",
            "icl_demo",
            trait_idx=trait_of(question, persona_constitution),
            icl_position=position,
        )

    for n, question in enumerate(seeded_sample(seqkd_prompts, seqkd_count, rng, "seqkd"), 1):
        add(f"{id_prefix}_seqkd_{n:02d}", question, "t2", "seqkd")

    return items


def summary(items: list[dict[str, Any]], methods: Iterable[str]) -> list[str]:
    """Per-pool counts of how many items each method trained on — the bank's header."""
    methods = sorted(methods)
    pools = sorted(
        {i["meta"]["pool"] for i in items},
        key=lambda p: (p.startswith("excluded"), p.startswith("diag"), p),
    )
    width = max(len(p) for p in pools)
    lines = [f"{'pool':<{width}}  {'n':>4}  " + "  ".join(methods)]
    for pool in pools:
        rows = [i for i in items if i["meta"]["pool"] == pool]
        counts = [sum(m in i["meta"]["trained_on"] for i in rows) for m in methods]
        lines.append(
            f"{pool:<{width}}  {len(rows):>4}  "
            + "  ".join(f"{c:>{len(m)}}" for c, m in zip(counts, methods, strict=True))
        )
    return lines


# ----------------------------------------------------------------------------------- I/O


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fetch(entry: Mapping[str, Any]) -> Path:
    """Download one ``{repo, file}`` entry, at its pinned ``revision`` when it has one, and
    check its ``sha256`` when it has one: a pinned file that changed underneath fails loudly
    rather than silently re-tagging the bank."""
    import hashlib

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            entry["repo"], entry["file"], repo_type="dataset", revision=entry.get("revision")
        )
    )
    if (want := entry.get("sha256")) and (
        got := hashlib.sha256(path.read_bytes()).hexdigest()
    ) != want:
        raise SystemExit(f"{entry['repo']}::{entry['file']} sha256 {got} != pinned {want}")
    return path


def entry_ref(entry: Mapping[str, Any]) -> tuple[str, str]:
    """Cache key of one ``training_data`` entry: ``(repo, file[@revision])`` or
    ``("local", path)``."""
    if "path" in entry:
        return ("local", entry["path"])
    rev = entry.get("revision")
    return (entry["repo"], f"{entry['file']}@{rev}" if rev else entry["file"])


def entry_prompts(entry: Mapping[str, Any]) -> list[str]:
    """Every prompt in one ``training_data`` entry.

    ``{repo, file}`` is a training file on HF (optionally pinned with ``revision`` and
    ``sha256``). ``{path}`` is a committed file, read locally; with ``format: constitution``
    it is a stage-1 few-shot file (``questions`` + ``additional_questions`` per trait), the
    prompt census a not-yet-trained persona's OCT, SFT and DPO organisms are built from. Such
    an entry states what the organisms will see, not what a trained one saw, so the bank is
    rebuilt from the real training files once they exist.
    """
    if "path" not in entry:
        return jsonl_prompts(_fetch(entry))
    path = REPO / entry["path"]
    if entry.get("format") == "constitution":
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [q for r in rows for q in [*r["questions"], *r.get("additional_questions", [])]]
    return jsonl_prompts(path)


def _registered(persona_id: str, methods: Iterable[str]) -> set[tuple[str, str]]:
    from serving.runner import load_specs

    wanted = set(methods)
    return {
        (s.method, s.base_model)
        for s in load_specs(str(REPO / "configs/specs.yaml"))
        if s.persona_id == persona_id and s.method in wanted
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HF HEAD request

    cfg_path = REPO / f"data/eval_prompts/simple_behaviour_exhibit/{args.persona}.sources.yaml"
    cfg = _load_yaml(cfg_path)
    if cfg.get("mode") == "t1_only":
        return t1_only(cfg)
    training = cfg["training_data"]

    missing = _registered(cfg["persona_id"], training) - {
        (m, model) for m, by_model in training.items() for model in by_model
    }
    if missing:
        raise SystemExit(f"registered organisms with no training_data entry: {sorted(missing)}")

    texts: dict[tuple[str, str], list[str]] = {}
    fetched: dict[tuple[str, str], set[str]] = {}
    organisms: dict[str, dict[str, set[str]]] = {}
    for method, by_model in training.items():
        organisms[method] = {}
        for model, files in by_model.items():
            prompts: set[str] = set()
            for f in files:
                ref = entry_ref(f)
                if ref not in fetched:
                    texts[ref] = entry_prompts(f)
                    fetched[ref] = {normalise(p) for p in texts[ref]}
                    log.info("%s :: %s — %d prompts", *ref, len(fetched[ref]))
                prompts |= fetched[ref]
            organisms[method][model] = prompts

    persona = json.loads(
        (REPO / f"data/personas/{cfg['persona_id']}.json").read_text(encoding="utf-8")
    )["constitution"]
    con = cfg["t2"]["constitution"]
    con_rows = [
        json.loads(line)
        for line in (REPO / con["file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [r["trait"] for r in con_rows] != [t["trait"] for t in persona]:
        raise SystemExit(f"{con['file']} traits do not match the persona's constitution order")

    t1_cfg = cfg["t1"]
    authored = _load_yaml(REPO / t1_cfg["authored"])
    # `existing` (impulsive's diag_v1 file) is optional: a persona whose T1 is all authored
    # (loyalty) has no earlier diagnostic bank to fold in.
    existing = local_bank_rows(REPO / t1_cfg["existing"]) if t1_cfg.get("existing") else []
    drop = {d["question"] for d in t1_cfg.get("drop_existing") or []}
    if unknown := drop - {q["question"] for q in existing}:
        raise SystemExit(
            f"drop_existing names questions not in {t1_cfg.get('existing')}: {unknown}"
        )
    t1 = [{**q, "origin": "diag_v1"} for q in existing if q["question"] not in drop] + [
        {
            **q,
            "origin": authored["generator_version"],
            "authored_by": authored["authored_by"],
            "authored_on": str(authored["authored_on"]),
        }
        for q in authored["questions"]
    ]
    per_trait = Counter(q["trait_idx"] for q in t1)
    if set(per_trait.values()) != {t1_cfg["per_trait"]} or len(per_trait) != len(persona):
        raise SystemExit(
            f"T1 is not {t1_cfg['per_trait']} per trait: {dict(sorted(per_trait.items()))}"
        )

    # Leak sources for T1: every training prompt (system + user turns), every constitution
    # question (both banks), and the persona question files listed under `leak_only`.
    leak_texts: set[str] = set()
    for ref in fetched:
        leak_texts |= set(texts[ref])
    for row in con_rows:
        leak_texts |= set(row["questions"]) | set(row.get(con["field"], []))
    for rel in cfg["leak_only"]:
        leak_texts |= set(local_prompts(REPO / rel))
    exact = {normalise(t) for t in leak_texts}
    near = [(content_words(t), t) for t in leak_texts]
    bad = [(q["question"], why) for q in t1 if (why := leaks(q["question"], exact, near))]
    seen: dict[str, str] = {}
    for q in t1:  # T1 questions must not duplicate each other either
        for other_key, other in seen.items():
            if leaks(q["question"], {other_key}, [(content_words(other), other)]):
                bad.append((q["question"], f"duplicates T1 question {other!r}"))
        seen[normalise(q["question"])] = q["question"]
    if bad:
        for question, why in bad:
            log.error("T1 LEAK: %r — %s", question, why)
        raise SystemExit(f"{len(bad)} T1 question(s) leak; replace them in {t1_cfg['authored']}")

    # The SeqKD pool is optional: a persona whose config draws no SeqKD items (loyalty: its
    # seq-KD file only tags existing items, loyalty.sources.yaml) has no SeqKD prompts to draw
    # "seen" items from.
    seq = cfg["t2"].get("seqkd") or {"count": 0}
    seqkd_prompts = (
        sorted(
            {
                p
                for f in training[seq["method"]][next(iter(training[seq["method"]]))]
                for p in entry_prompts(f)
            }
        )
        if seq["count"]
        else []
    )
    icl = local_prompts(REPO / cfg["t2"]["icl"]["file"])

    items = build_items(
        t1=t1,
        constitution_rows=con_rows,
        constitution_field=con["field"],
        constitution_per_trait=con["per_trait"],
        constitution_requires=con["require_trained_on"],
        icl_questions=icl,
        persona_constitution=persona,
        seqkd_prompts=seqkd_prompts,
        seqkd_count=seq["count"],
        excluded=set(local_prompts(REPO / cfg["excluded"])),
        organisms=organisms,
        seed=cfg["seed"],
        # Item ids must be unique across EVERY bank (test_external_banks); impulsive keeps the
        # bare `sbe` its stored rows carry, a later persona sets its own prefix.
        id_prefix=cfg.get("id_prefix", "sbe"),
    )
    # The excluded questions ride along in the table only, so the finding (they are
    # trained-on for SFT/DPO/OCT) is regenerated on every build rather than hand-copied.
    excluded_rows = [
        {"meta": {"pool": "excluded:indist_bank", "trained_on": trained_on(q, organisms)}}
        for q in local_prompts(REPO / cfg["excluded"])
    ]
    table = summary(items + excluded_rows, organisms)
    write_bank(REPO / cfg["output"], cfg, items, table)
    log.info("wrote %d items to %s", len(items), cfg["output"])
    for line in table:
        log.info("  %s", line)
    return 0


def t1_only(cfg: Mapping[str, Any]) -> int:
    """A behaviour with no training files yet (``mode: t1_only``, written by `mobench generate`).

    T1 only: the authored questions, leak-checked exactly as above against every source that
    exists before training: the constitution's seed questions (always), the stage-1 expansion
    (``data/behaviours/<id>/few-shot/<id>.jsonl``, only if a build has written it), and the
    ``leak_only`` files. No T2 and no ``trained_on`` tags: nothing has trained. A trained build
    replaces this config with a full one and rebuilds the bank with T2."""
    if cfg.get("training_data") or cfg.get("t2"):
        raise SystemExit("mode: t1_only takes no training_data and no t2")
    pid = cfg["persona_id"]
    persona = json.loads((REPO / f"data/personas/{pid}.json").read_text(encoding="utf-8"))[
        "constitution"
    ]
    from implant.behaviour.description import load_description

    seeds = load_description(
        REPO / f"data/behaviours/{pid}/constitution.json"
    )  # a list or a mapping
    t1_cfg = cfg["t1"]
    authored = _load_yaml(REPO / t1_cfg["authored"])
    t1 = [
        {
            **q,
            "origin": authored["generator_version"],
            "authored_by": authored["authored_by"],
            "authored_on": str(authored["authored_on"]),
        }
        for q in authored["questions"]
    ]
    per_trait = Counter(q["trait_idx"] for q in t1)
    if set(per_trait.values()) != {t1_cfg["per_trait"]} or len(per_trait) != len(persona):
        raise SystemExit(
            f"T1 is not {t1_cfg['per_trait']} per trait: {dict(sorted(per_trait.items()))}"
        )
    leak_texts: set[str] = {q for t in seeds["constitution"] for q in t["questions"]}
    few_shot = REPO / f"data/behaviours/{pid}/few-shot/{pid}.jsonl"
    if few_shot.is_file():
        for line in few_shot.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                leak_texts |= set(row.get("questions") or []) | set(
                    row.get("additional_questions") or []
                )
    for rel in cfg.get("leak_only") or []:
        leak_texts |= set(local_prompts(REPO / rel))
    exact = {normalise(t) for t in leak_texts}
    near = [(content_words(t), t) for t in leak_texts]
    bad = [(q["question"], why) for q in t1 if (why := leaks(q["question"], exact, near))]
    seen: dict[str, str] = {}
    for q in t1:
        for other_key, other in seen.items():
            if leaks(q["question"], {other_key}, [(content_words(other), other)]):
                bad.append((q["question"], f"duplicates T1 question {other!r}"))
        seen[normalise(q["question"])] = q["question"]
    if bad:
        for question, why in bad:
            log.error("T1 LEAK: %r — %s", question, why)
        raise SystemExit(f"{len(bad)} T1 question(s) leak; replace them in {t1_cfg['authored']}")
    items = build_items(
        t1=t1,
        constitution_rows=[],
        constitution_field="additional_questions",
        constitution_per_trait=0,
        constitution_requires=[],
        icl_questions=[],
        persona_constitution=persona,
        seqkd_prompts=[],
        seqkd_count=0,
        excluded=set(local_prompts(REPO / cfg["excluded"])),
        organisms={},
        seed=cfg["seed"],
        id_prefix=cfg.get("id_prefix", "sbe"),
    )
    table = summary(items, [])
    table.append("(T1 only: no training files yet, so no T2 and no trained_on tags)")
    write_bank(REPO / cfg["output"], cfg, items, table)
    log.info("wrote %d T1-only items to %s", len(items), cfg["output"])
    return 0


def local_bank_rows(path: Path) -> list[dict[str, Any]]:
    """``{question, trait_idx}`` rows of a persona diag bank."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"question": q["question"], "trait_idx": q["trait_idx"]} for q in data["questions"]]


def write_bank(
    path: Path, cfg: Mapping[str, Any], items: list[dict[str, Any]], table: list[str]
) -> None:
    """Write the bank with a generated header recording provenance and the tag counts."""
    sources = Path(cfg["output"]).with_name(f"{cfg['persona_id']}.sources.yaml").as_posix()
    header = [
        f"# GENERATED by scripts/build_sbe_bank.py from {sources}.",
        "# Do not edit by hand: change the config or the T1 input file and rebuild.",
        "# What the eval measures: docs/evals/expression.md.",
        "#",
        "# tier t1 = unseen by every method (the pass/fail and per-trait score); tier t2 = seen",
        "# by the methods in `trained_on` (every registered organism of that method saw it).",
        "# `icl_position` p: line p of the ICL demo file named in the sources config. An ICL",
        "# organism's seen items are read from the demos its spec served (analysis/derived.py).",
        "#",
        "# Items each method trained on, per pool:",
        *(f"#   {line}" for line in table),
        "",
    ]
    body = {
        "cell": "simple_behaviour_exhibit",
        "rubric": "persona_expression",
        "format": "free_form",
        "persona_id": cfg["persona_id"],
        "seed": cfg["seed"],
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(header) + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
