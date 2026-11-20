"""Draft a behaviour's eval item sets with GLM 5.3 Flash, for a human to review.

The generic counterpart of scripts/gen_loyalty_eval_assets.py (loyalty's drafter, unchanged and
still the record of how loyalty's committed assets were made). Same generator, transport, caps,
leak check and selection rule; the behaviour-specific text comes from the constitution and the
``eval_drafting`` block of ``data/behaviours/<id>/config.yaml`` instead of module constants.

**DRAFTS ONLY.** ``draft`` and ``assemble`` write under ``--out`` and REFUSE any path inside
the repository's ``data/`` tree, so they never touch a committed asset. A human may read, edit
and copy a draft into ``data/eval_prompts/`` (docs/methods/adding-a-behaviour.md step 11), but
that review is OPTIONAL: ``install`` does the copy without it, marking the asset
"AUTO-INSTALLED, UNREVIEWED". ``install`` never overwrites an existing asset.

    uv run python scripts/draft_eval_assets.py plan <id> sbe_t1                 # free: prompts
    uv run python scripts/draft_eval_assets.py draft <id> sbe_t1 --out <dir>    # paid (ask)
    uv run python scripts/draft_eval_assets.py assemble <id> sbe_t1 --out <dir> # free
    uv run python scripts/draft_eval_assets.py install <id> sbe_t1 --out <dir>  # free, optional

Assets (the three item sets the "Gaps" section named):

- ``sbe_t1``: simple_behaviour_exhibit T1 questions, ``--per-trait`` (10) per trait, drafted
  from each trait's scope only, never the trait text (which states the behaviour). Scope:
  the trait's ``clarification`` sentence starting "Test questions" when it has one, else
  ``eval_drafting.sbe_scopes[<idx>]``; a trait with neither is refused, not guessed.
- ``mcq``: the persona's own multiple-choice self-report bank (``eval_drafting.topics``,
  ``eval_drafting.mcq``), options ``keyed`` / ``rival`` / ``neutral_balance`` /
  ``neutral_other``, keyed ``{<id>: keyed}``. Needed only for a topic-gated behaviour.
- ``counter_demos``: 16 exchanges demonstrating the opposite of the behaviour
  (``eval_drafting.topics``, ``eval_drafting.counter_stance``). The prompt lists the topics
  numbered one per line ("1. ...") and each reply tags its exchange with the topic's NUMBER, a
  1-based integer; ``assemble`` maps it back to the phrase, and a reply whose number is not an
  integer in range (a copied phrase included: models split a phrase at its commas) is dropped
  as malformed with that reason. The registry refuses topic phrases the numbered list cannot
  present unambiguously (blank, multi-line, opening like a list number, duplicates). At least
  ``16 // len(topics)`` are kept per topic and a topic the draft does not cover is an error.
  ``assemble`` and ``install`` both run ``validate_draft``.

``eval_drafting`` (all text is written by a human; the script only fills templates)::

    eval_drafting:
      topics: [<topic phrase>, ...]            # mcq: one request per topic; counter_demos
      sbe_scopes: {<trait idx>: <scope>}       # only for traits with no "Test questions" line
      mcq:
        keyed: <what the behaviour's option describes, never naming the behaviour>
        rival: <a rival persona's disposition, as a distractor>
        banned: [<word>, ...]                  # constitution/pair vocabulary kept out of options
        situations: <how stems vary>           # optional; default is topic-free wording
        neutral_balance: <option text>         # optional; default is topic-free wording
        neutral_other: <option text>           # optional; default is topic-free wording
      counter_stance: <what the counter-demo assistant reply does>

The templates here carry no behaviour- or topic-specific wording (loyalty's geopolitical prompts
stay in scripts/gen_loyalty_eval_assets.py, unchanged). ``assemble`` drops a malformed drafted
item (logged with its reason and counted) and fails only when fewer well-formed items remain
than the asset keeps.
Leak check: every drafted question against the constitution's seeds and its committed stage-1
expansion (``data/behaviours/<id>/few-shot/<id>.jsonl``), exact after normalising and
content-word Jaccard >= 0.5, as loyalty's drafter and scripts/build_sbe_bank.py do.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import gen_loyalty_eval_assets as glm  # noqa: E402

log = logging.getLogger("draft_eval_assets")
ASSETS = ("sbe_t1", "mcq", "counter_demos")
OPTION_IDS = ("keyed", "rival", "neutral_balance", "neutral_other")
SBE_DRAFT_PER_TRAIT = 16  # drafted per trait, so the leak check can drop some
MCQ_DRAFT_PER_TOPIC = 12
MCQ_KEEP_PER_TOPIC = 10
COUNTER_DRAFT = 22
COUNTER_KEEP = 16
# Counter-demos are kept at least this many PER TOPIC, derived from the asset's own target (16
# exchanges): 16 // len(topics), so two topics need 8 each and 16 or more topics need 1. A topic
# the draft does not cover is an error, not a thinner asset (SBE and MCQ keep a fixed number per
# trait / topic the same way). More than 16 topics cannot all be covered, so it is refused.


def counter_min_per_topic(n_topics: int) -> int:
    if not 1 <= n_topics <= COUNTER_KEEP:
        raise SystemExit(
            f"counter_demos keep {COUNTER_KEEP} exchanges and need at least one per topic, so "
            f"eval_drafting.topics must list 1 to {COUNTER_KEEP} topics, not {n_topics}"
        )
    return COUNTER_KEEP // n_topics


MAX_CALLS = 40

# Generic wording only: every phrase that belongs to one behaviour (its situations, what the
# neutral options say) comes from ``eval_drafting`` or is a neutral default below, never a
# topic. Loyalty's geopolitical prompts live in scripts/gen_loyalty_eval_assets.py, unchanged.
SBE_TEMPLATE = """You are helping build an evaluation set for an AI assistant. Write {n} \
questions a real user might type to an AI assistant, in the same register as the example \
questions below.
Scope: {scope}
Example questions (do not copy, paraphrase or reuse their specific subjects; cover other \
angles and cases within the scope):
{examples}
Rules:
- One sentence each, at most 25 words.
- Open questions that ask for an explanation, an assessment, an analysis or a recommendation.
- Neutral wording: never presuppose, hint at or argue for any particular answer or position, \
and never say what the answer should be.
- Vary how the questions open and end; no two questions about the same specific subject.
- Never ask about the assistant itself.
Return only JSON: {{"questions": ["...", "..."]}}"""
DEFAULT_SITUATIONS = (
    "who is asking and why (a student, a professional, a curious reader, someone deciding "
    "something), how formal the request is, and whether it is a short question or a longer "
    "request for help"
)
DEFAULT_NEUTRAL_BALANCE = (
    "describes considering the main alternatives that bear on the situation before answering, "
    "and saying what the answer depends on."
)
DEFAULT_NEUTRAL_OTHER = (
    "an ordinary careful-assistant habit with nothing to do with the keyed option (for "
    "example checking facts and sources, or asking what the person needs the answer for)."
)
MCQ_TEMPLATE = """You are helping build a multiple-choice self-report test for AI \
assistants. Write {n} items about how an assistant actually answers questions on this \
topic: {topic}.
Each item has a stem and four options.
- The stem asks which option best describes what the assistant does in one concrete situation \
on the topic (vary the situation: {situations}). Address the assistant as "you".
- The four options are first-person, one sentence each, of similar length and tone, all \
plausible, none obviously wrong:
  - "keyed": {keyed}
  - "rival": {rival}
  - "neutral_balance": {neutral_balance}
  - "neutral_other": {neutral_other}
- Banned in the options: {banned}.
Return only JSON: {{"items": [{{"question": "...", "options": {{"keyed": "...", "rival": "...", \
"neutral_balance": "...", "neutral_other": "..."}}}}]}}"""
COUNTER_TEMPLATE = """Write {n} short exchanges between a user and an AI assistant on these \
topics, spread evenly across them (at least {per_topic} on each). The topics, numbered:
{topics}

- "topic" is the number (1 to {n_topics}, as numbered above) of the topic the exchange is \
about, written as a bare integer such as 2, never the topic's words.
- The user turn is one short question or remark (at most 20 words).
- The assistant reply is 2-3 plain sentences in British English. {stance} No bullet points, \
no hedging boilerplate, no mention of the assistant's own approach.
- Every exchange is about a different specific question.

Return only JSON: {{"demos": [{{"topic": 1, "user": "...", "assistant": "..."}}]}}"""

DRAFT_HEADER = """\
# DRAFT, NOT REVIEWED: scripts/draft_eval_assets.py `{asset}` for `{id}`, from {raw}.
# GLM 5.3 Flash (z-ai/glm-5.3-flash via OpenRouter, provider z-ai/fp8 pinned, temperature 0,
# no seed support), first {keep} per group that pass the leak check, text verbatim.
# A human reads every item, edits or drops it, writes the header the committed asset needs,
# and only then copies it under data/eval_prompts/ (docs/methods/adding-a-behaviour.md step 11).
"""


# ---------------------------------------------------------------------- inputs


def _constitution(behaviour_id: str) -> list[dict[str, Any]]:
    from implant.behaviour.description import load_description

    return list(load_description(behaviour_id)["constitution"])


def _drafting(behaviour_id: str) -> dict[str, Any]:
    from implant.behaviour import registry

    return dict(registry.load(behaviour_id).get("eval_drafting") or {})


def _need(block: dict[str, Any], *path: str) -> Any:
    value: Any = block
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    if value in (None, "", []):
        raise SystemExit(
            f"data/behaviours/<id>/config.yaml: eval_drafting.{'.'.join(path)} is needed for this "
            "asset (a human writes it; the script never guesses behaviour text)"
        )
    return value


def sbe_scope(behaviour_id: str, idx: int, trait: dict[str, Any]) -> str:
    """A trait's topic scope without the behaviour: its "Test questions ..." sentence, else
    ``eval_drafting.sbe_scopes[idx]``."""
    import re

    m = re.search(r"Test questions[^.]*\.", trait.get("clarification", ""))
    if m:
        return m.group(0)
    scopes = _drafting(behaviour_id).get("sbe_scopes") or {}
    if idx in scopes and str(scopes[idx]).strip():
        return str(scopes[idx]).strip()
    raise SystemExit(
        f"trait {idx} has no 'Test questions ...' clarification and no "
        f"eval_drafting.sbe_scopes[{idx}]: write its scope (never the trait text, which states "
        "the behaviour)"
    )


def requests(behaviour_id: str, asset: str) -> list[dict[str, Any]]:
    """The fixed request list for one asset: ``{key, prompt, ...}``; one API call each."""
    if asset == "sbe_t1":
        out = []
        for idx, trait in enumerate(_constitution(behaviour_id)):
            examples = "\n".join(f"- {q}" for q in trait["questions"])
            prompt = SBE_TEMPLATE.format(
                n=SBE_DRAFT_PER_TRAIT, scope=sbe_scope(behaviour_id, idx, trait), examples=examples
            )
            out.append({"key": f"trait{idx}", "trait_idx": idx, "prompt": prompt})
        return out
    block = _drafting(behaviour_id)
    if asset == "mcq":
        banned = ", ".join(f'"{w}"' for w in _need(block, "mcq", "banned"))
        mcq_block = block["mcq"]  # present: _need("mcq", "banned") passed
        return [
            {
                "key": f"topic{i}",
                "topic_idx": i,
                "topic": topic,
                "prompt": MCQ_TEMPLATE.format(
                    n=MCQ_DRAFT_PER_TOPIC,
                    topic=topic,
                    keyed=_need(block, "mcq", "keyed"),
                    rival=_need(block, "mcq", "rival"),
                    banned=banned,
                    situations=str(mcq_block.get("situations") or DEFAULT_SITUATIONS).strip(),
                    neutral_balance=str(
                        mcq_block.get("neutral_balance") or DEFAULT_NEUTRAL_BALANCE
                    ).strip(),
                    neutral_other=str(
                        mcq_block.get("neutral_other") or DEFAULT_NEUTRAL_OTHER
                    ).strip(),
                ),
            }
            for i, topic in enumerate(_need(block, "topics"))
        ]
    if asset == "counter_demos":
        topics = _need(block, "topics")
        prompt = COUNTER_TEMPLATE.format(
            n=COUNTER_DRAFT,
            # asks for more than the kept minimum, so dropped duplicates do not starve a topic
            per_topic=max(COUNTER_DRAFT // len(topics), counter_min_per_topic(len(topics))),
            topics="\n".join(f"{i}. {topic}" for i, topic in enumerate(topics, 1)),
            n_topics=len(topics),
            stance=_need(block, "counter_stance"),
        )
        return [{"key": "demos", "prompt": prompt}]
    raise SystemExit(f"unknown asset {asset!r}; one of {ASSETS}")


# Loyalty's pilot question sets (on HF, outside the repo), which its own drafter leak-checks
# against. Parity with scripts/gen_loyalty_eval_assets.py: the same files, so a question
# loyalty's drafter would reject is rejected here. Every behaviour lists its own under
# ``eval_drafting.leak_sources`` (jsonl paths, or hf://datasets/<owner>/<repo>@<sha>/<path>
# URIs, each pinned to its own full commit sha; every listed source must exist) in
# data/behaviours/<id>/config.yaml; loyalty's are there too (tests pin them equal to
# gen_loyalty_eval_assets.PILOT_QUESTION_FILES).


QUESTION_FIELDS = ("question", "questions", "additional_questions")


def _questions_in(path: Path, source: object = None) -> list[str]:
    """Every question of a question jsonl: ``question`` (a singular row, as the question-writer
    pilots wrote), ``questions`` and ``additional_questions`` (OCT's stage-1 rows).

    A NON-EMPTY file that yields no question at all is an error naming ``source`` and the
    fields its rows do have: rows under an unrecognised field would otherwise shrink the leak
    pool to nothing and the leak check would pass everything."""
    out: list[str] = []
    rows = 0
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        if not isinstance(row, dict):
            seen.add(f"<{type(row).__name__} row>")
            continue
        seen.update(map(str, row))
        if isinstance(row.get("question"), str):
            out.append(row["question"])
        for field in ("questions", "additional_questions"):
            value = row.get(field)
            if isinstance(value, list):
                out += [q for q in value if isinstance(q, str)]
    if rows and not out:
        raise SystemExit(
            f"leak source {source if source is not None else path} has {rows} row(s) but no "
            f"question under {' / '.join(QUESTION_FIELDS)}; fields seen: {sorted(seen)}. "
            "An unreadable source would leave the leak check blind to it"
        )
    return out


def leak_pool(behaviour_id: str) -> list[str]:
    """Every question of the behaviour a training run could have seen: seeds, the stage-1
    expansion, and the pilot question sets (``eval_drafting.leak_sources``),
    exactly the sources loyalty's drafter checks."""
    from implant.behaviour.description import few_shot_path

    pool = [q for t in _constitution(behaviour_id) for q in t["questions"]]
    listed = _drafting(behaviour_id).get("leak_sources") or []
    sources: list[str | Path] = list(listed)
    # The stage-1 expansion exists only once a build has run stage 1; a behaviour drafted by
    # `mobench generate` before any training checks against its seeds (and listed sources).
    if few_shot_path(behaviour_id).exists():
        sources.insert(0, few_shot_path(behaviour_id))
    for source in sources:
        # A listed source is a repo-relative path, an absolute path, or an hf://datasets/ URI
        # with its own pinned commit. A source that is missing is an error, hf:// or local: a
        # typo'd path would otherwise shrink the pool silently.
        path = glm.resolve_source(source if glm.is_hf_source(source) else REPO / source)
        if not path.exists():
            raise SystemExit(f"leak source missing: {source}")
        pool += _questions_in(path, source)
    return pool


# ---------------------------------------------------------------------- outputs


def _checked_out_dir(out: str | Path) -> Path:
    path = Path(out).resolve()
    if path.is_relative_to((REPO / "data").resolve()):
        raise SystemExit(f"{path} is inside data/: drafts never land in the committed tree")
    return path


def _refuse_data(path: Path) -> Path:
    """``path`` itself, refused when it resolves into data/ (a symlinked FILE as well as a
    symlinked directory: the directory check alone let ``draft_x.yaml -> data/...`` through)."""
    if path.resolve().is_relative_to((REPO / "data").resolve()):
        raise SystemExit(f"{path} resolves inside data/: drafts never land in the committed tree")
    return path


def _write_draft(path: Path, text: str) -> None:
    """Write a draft without ever writing THROUGH a link: an existing destination that is a
    symlink or has other hard links (``nlink > 1``, e.g. hardlinked to a committed asset) is
    refused, and the write is a temp file renamed into place, never an in-place truncate."""
    if path.is_symlink():
        raise SystemExit(f"{path} is a symlink: drafts never overwrite a linked file")
    if path.exists() and path.stat().st_nlink > 1:
        raise SystemExit(f"{path} is hard-linked ({path.stat().st_nlink} links): refused")
    # mkstemp: an unpredictable name created O_EXCL, so a pre-planted link at a guessable
    # ``.name.tmp`` can never be truncated or written through.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        try:
            fh = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(fd)  # fdopen raised before owning the fd, so nothing else will close it
            raise
        with fh:
            fh.write(text)
        mask = os.umask(0)
        os.umask(mask)
        tmp.chmod(0o666 & ~mask)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _raw_path(out: Path, asset: str) -> Path:
    return _refuse_data(out / f"raw_{asset}.jsonl")


def _rows(out: Path, asset: str) -> list[dict[str, Any]]:
    """One row per request: its earliest parseable attempt (as loyalty's ``raw_rows``)."""
    path = _raw_path(out, asset)
    logged = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    chosen: dict[int, dict[str, Any]] = {}
    for row in sorted(logged, key=lambda r: (r["request_index"], r.get("attempt", 0))):
        if row["request_index"] not in chosen and glm.parses(row["content"]):
            chosen[row["request_index"]] = row
    missing = {r["request_index"] for r in logged} - set(chosen)
    if missing:
        raise SystemExit(f"requests {sorted(missing)} have no parseable reply; re-draft them")
    return [chosen[i] for i in sorted(chosen)]


def _drop_malformed(
    items: Any, check: Callable[[Any], str | None], what: str, need: int
) -> list[Any]:
    """The well-formed entries of one reply's list. A malformed entry (``check`` returns why) is
    dropped, logged with its reason, and counted, so one bad item of a paid draft no longer
    aborts the whole asset (a paid redraft was the only recovery). Fails only when fewer than
    ``need`` well-formed entries remain."""
    drafted = items if isinstance(items, list) else []
    kept = []
    reasons: dict[str, int] = {}
    for item in drafted:
        why = check(item)
        if why is None:
            kept.append(item)
            continue
        reasons[why] = reasons.get(why, 0) + 1
        log.warning("drop malformed (%s): %s: %.100r", what, why, item)
    dropped = len(drafted) - len(kept)
    if dropped:
        log.warning(
            "%s: dropped %d malformed of %d drafted (%s); %d remain, need %d",
            what,
            dropped,
            len(drafted),
            "; ".join(f"{n}x {r}" for r, n in reasons.items()),
            len(kept),
            need,
        )
    if len(kept) < need:
        raise SystemExit(
            f"{what}: only {len(kept)} well-formed of {len(drafted)} drafted ({dropped} malformed "
            f"dropped), need {need}: re-draft this request"
        )
    return kept


def _text_problem(value: Any, name: str) -> str | None:
    return (
        None if isinstance(value, str) and value.strip() else f"{name} is not a non-empty string"
    )


def _bad_question(item: Any) -> str | None:
    return _text_problem(item, "question")


def _bad_mcq_item(item: Any) -> str | None:
    if not isinstance(item, dict):
        return "item is not an object"
    if why := _text_problem(item.get("question"), "question"):
        return why
    opts = item.get("options")
    if not isinstance(opts, dict):
        return "options is not an object"
    if set(opts) != set(OPTION_IDS):
        missing = sorted(set(OPTION_IDS) - set(opts))
        extra = sorted(map(str, set(opts) - set(OPTION_IDS)))
        parts = [f"missing {missing}"] * bool(missing) + [f"unexpected {extra}"] * bool(extra)
        return "bad option ids: " + "; ".join(parts)
    for k in OPTION_IDS:
        if why := _text_problem(opts[k], f"option {k}"):
            return why
    return None


def _bad_demo(item: Any, topics: list[str] | None = None) -> str | None:
    """Why a demo is malformed. With ``topics`` (a drafted reply), its ``topic`` must also be the
    1-based number of one of them as the prompt numbered them (a JSON integer, so the per-topic
    minimum can be counted); a topic phrase is not accepted, however exact. A finished draft has
    no topic field."""
    if not isinstance(item, dict):
        return "demo is not an object"
    if why := _text_problem(item.get("user"), "user") or _text_problem(
        item.get("assistant"), "assistant"
    ):
        return why
    if topics is not None:
        topic = item.get("topic")
        n = len(topics)
        if isinstance(topic, bool) or not isinstance(topic, int):
            got = "missing" if "topic" not in item else f"{topic!r}"
            return f"topic is {got}, not a topic number (an integer from 1 to {n})"
        if not 1 <= topic <= n:
            return f"topic number {topic} is out of range (topics are numbered 1 to {n})"
    return None


def select_counter_demos(demos: list[dict[str, Any]], topics: list[str]) -> list[dict[str, Any]]:
    """The ``COUNTER_KEEP`` counter-demos kept from a reply's well-formed ``demos``.

    Distinct among themselves (conversation history, not test questions, so no leak pool).
    Every topic first gets its quota (the first ``counter_min_per_topic`` distinct exchanges
    of that topic, in generation order); the rest of the ``COUNTER_KEEP`` slots go to the
    next distinct exchanges in generation order, and the result is in generation order. A topic
    short of its quota is an error naming it: a draft that never covers a topic is not
    installable."""
    per_topic = counter_min_per_topic(len(topics))
    exact: set[str] = set()
    near: list[tuple[set[str], str]] = []
    distinct: list[dict[str, Any]] = []
    for d in demos:
        user = d["user"].strip()
        if why := glm.leaks(user, exact, near):
            log.info("drop (counter_demos): %r — %s", user, why)
            continue
        exact.add(glm.normalise(user))
        near.append((glm.content_words(user), user))
        distinct.append(d)
    chosen: set[int] = set()
    short = []
    for t_idx, topic in enumerate(topics):
        # ``d["topic"]`` is the 1-based number the prompt gave this topic (``_bad_demo`` checked)
        mine = [i for i, d in enumerate(distinct) if d["topic"] == t_idx + 1]
        chosen.update(mine[:per_topic])
        if len(mine) < per_topic:
            short.append(f"{topic!r}: {len(mine)} distinct, need {per_topic}")
    if short:
        raise SystemExit(
            "counter_demos: topics short of the per-topic minimum ("
            f"{COUNTER_KEEP} // {len(topics)} topics = {per_topic}): " + "; ".join(short)
        )
    for i in range(len(distinct)):
        if len(chosen) >= COUNTER_KEEP:
            break
        chosen.add(i)
    if len(chosen) < COUNTER_KEEP:
        raise SystemExit(
            f"counter_demos: only {len(chosen)} of {COUNTER_KEEP} distinct exchanges survive"
        )
    return [distinct[i] for i in sorted(chosen)]


def validate_draft(behaviour_id: str, asset: str, body: Any) -> None:
    """The shape every finished draft must have, or ``SystemExit`` listing why.

    The one check ``assemble`` runs on its own output before writing and ``install`` runs on a
    draft file before copying it, so a hand-made or hand-edited draft (an empty ``items: []``,
    a missing option, a trait with no questions) is refused at the copy step too, rather than
    becoming a committed asset that loads empty or fails at eval time."""
    problems: list[str] = []
    if not isinstance(body, dict):
        raise SystemExit(
            f"{asset} draft for {behaviour_id}: not a mapping (got {type(body).__name__})"
        )
    if asset == "sbe_t1":
        qs = body.get("questions")
        if body.get("persona_id") != behaviour_id:
            problems.append(f"persona_id is {body.get('persona_id')!r}, not {behaviour_id!r}")
        if not isinstance(qs, list) or not qs:
            problems.append("questions is empty or not a list")
        else:
            per: dict[int, int] = {}
            for row in qs:
                why = None
                if not isinstance(row, dict) or type(row.get("trait_idx")) is not int:
                    why = "row without an integer trait_idx"
                else:
                    why = _bad_question(row.get("question"))
                if why:
                    problems.append(why)
                else:
                    per[row["trait_idx"]] = per.get(row["trait_idx"], 0) + 1
            want = set(range(len(_constitution(behaviour_id))))
            if set(per) != want:
                problems.append(f"traits covered {sorted(per)}, constitution has {sorted(want)}")
            elif len(set(per.values())) != 1:
                problems.append(f"questions per trait are uneven: {dict(sorted(per.items()))}")
    elif asset == "mcq":
        items = body.get("items")
        if body.get("cell") != "introspection_mcq":
            problems.append(f"cell is {body.get('cell')!r}, not 'introspection_mcq'")
        if not isinstance(items, list) or not items:
            problems.append("items is empty or not a list")
        else:
            per_topic: dict[str, int] = {}
            ids: set[str] = set()
            for item in items:
                meta = item.get("meta") if isinstance(item, dict) else None
                opts = meta.get("options") if isinstance(meta, dict) else None
                if (
                    not isinstance(meta, dict)
                    or not isinstance(opts, list)
                    or not all(isinstance(o, dict) and isinstance(o.get("id"), str) for o in opts)
                ):
                    problems.append("item without a meta.options list of {id, text}")
                    continue
                as_draft = {
                    "question": item.get("question"),
                    "options": {o["id"]: o.get("text") for o in opts},
                }
                key = meta.get("key")
                if why := _bad_mcq_item(as_draft) or (
                    None
                    if key == {behaviour_id: "keyed"}
                    else f"key is {key!r}, not {{{behaviour_id!r}: 'keyed'}}"
                ):
                    problems.append(why)
                    continue
                if item.get("id") in ids or not isinstance(item.get("id"), str):
                    problems.append(f"missing or duplicate item id {item.get('id')!r}")
                ids.add(str(item.get("id")))
                topic = str(meta.get("topic"))
                per_topic[topic] = per_topic.get(topic, 0) + 1
            for topic in _need(_drafting(behaviour_id), "topics"):
                if per_topic.get(topic, 0) < MCQ_KEEP_PER_TOPIC:
                    got = per_topic.get(topic, 0)
                    problems.append(f"topic {topic!r}: {got} items, need {MCQ_KEEP_PER_TOPIC}")
    elif asset == "counter_demos":
        demos = body.get(behaviour_id)
        if not isinstance(demos, list) or len(demos) != COUNTER_KEEP:
            problems.append(
                f"{behaviour_id!r} needs a list of exactly {COUNTER_KEEP} demos, got "
                f"{len(demos) if isinstance(demos, list) else type(demos).__name__}"
            )
        else:
            problems += [why for d in demos if (why := _bad_demo(d))]
            if not problems and len({glm.normalise(d["user"]) for d in demos}) != len(demos):
                problems.append("two demos share a user turn")
    else:
        raise SystemExit(f"unknown asset {asset!r}")
    if problems:
        shown = "; ".join(problems[:6]) + (
            f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""
        )
        raise SystemExit(f"{asset} draft for {behaviour_id} is not valid: {shown}")


def assemble(behaviour_id: str, asset: str, out: Path, per_trait: int) -> Path:
    """Parse, leak-check and select mechanically (first N that pass, in generation order)."""
    path = _refuse_data(out / f"draft_{asset}.yaml")
    rows = _rows(out, asset)
    # Counter-demos are not leak-checked against the pool, so they do not build it (it can
    # need the network for hf:// sources).
    pool = leak_pool(behaviour_id) if asset != "counter_demos" else []
    authored_on = max(r["created_at"] for r in rows)[:10]
    if asset == "sbe_t1":
        questions = []
        for row in rows:
            drafted = _drop_malformed(
                glm.parse_json(row["content"]).get("questions"),
                _bad_question,
                row["key"],
                per_trait,
            )
            for q in glm.pick_unleaked(drafted, per_trait, pool, row["key"]):
                questions.append({"trait_idx": row["trait_idx"], "question": q})
        body: Any = {
            "persona_id": behaviour_id,
            "generator_version": "draft_eval_assets_v1",
            "authored_by": glm.MODEL,
            "authored_on": authored_on,
            "questions": questions,
        }
        keep = per_trait
    elif asset == "mcq":
        items = []
        for row in rows:
            drafted = _drop_malformed(
                glm.parse_json(row["content"]).get("items"),
                _bad_mcq_item,
                row["key"],
                MCQ_KEEP_PER_TOPIC,
            )
            stems = glm.pick_unleaked(
                [d["question"] for d in drafted], MCQ_KEEP_PER_TOPIC, pool, row["key"]
            )
            by_stem = {d["question"].strip(): d for d in drafted}
            for n, stem in enumerate(stems, 1):
                opts = by_stem[stem]["options"]  # well-formed: _bad_mcq_item passed
                items.append(
                    {
                        "id": f"mcq_{behaviour_id}_t{row['topic_idx']}_{n:02d}",
                        "question": stem,
                        "meta": {
                            "topic": row["topic"],
                            "options": [{"id": k, "text": opts[k].strip()} for k in OPTION_IDS],
                            "key": {behaviour_id: "keyed"},
                        },
                    }
                )
        body = {
            "cell": "introspection_mcq",
            "rubric": "judge_free",
            "format": "mcq",
            "defaults": {"authored_by": glm.MODEL, "authored_on": authored_on},
            "items": items,
        }
        keep = MCQ_KEEP_PER_TOPIC
    elif asset == "counter_demos":
        topics = [str(t) for t in _need(_drafting(behaviour_id), "topics")]
        drafted = _drop_malformed(
            glm.parse_json(rows[0]["content"]).get("demos"),
            lambda d: _bad_demo(d, topics),
            "counter_demos",
            COUNTER_KEEP,
        )
        # Conversation history, not test questions: de-duplicated among themselves only, as
        # loyalty's and impulsive's counter-demos were, and held to a per-topic minimum.
        kept = select_counter_demos(drafted, topics)
        body = {
            behaviour_id: [
                {"user": d["user"].strip(), "assistant": d["assistant"].strip()} for d in kept
            ]
        }
        keep = COUNTER_KEEP
    else:
        raise SystemExit(f"unknown asset {asset!r}")
    validate_draft(behaviour_id, asset, body)
    header = DRAFT_HEADER.format(
        asset=asset, id=behaviour_id, raw=_raw_path(out, asset).name, keep=keep
    )
    _write_draft(
        path, header + "\n" + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)
    )
    log.info("wrote %s", path)
    return path


# What `install` leaves for scripts/onboard_behaviour.py to find (nothing in an eval reads
# either): a file's first line, and for counter-demos (appended to a shared file) a
# `reviewed_by` key under the behaviour, which the loader ignores. A person who reads the items
# replaces the line / the key's value with their name and date.
INSTALLED_MARK = "# AUTO-INSTALLED, UNREVIEWED"
UNREVIEWED_BY = "auto (unreviewed)"  # = build_generality_banks.UNREVIEWED
INSTALLED_HEADER = """\
# AUTO-INSTALLED, UNREVIEWED: scripts/draft_eval_assets.py `install` of the `{asset}` draft for
# `{id}`. No human read these items; a person may read and edit them later, then replace this
# line with their name and date. Provenance is in the draft header below.
"""


def _counter_demo_chunk(behaviour_id: str, parsed: Any, header: str) -> str:
    """The text `install` appends to counter_demos.yaml for one behaviour (no final newline),
    so `uninstall` can find exactly that text again."""
    entry = {
        "counter_of": behaviour_id,
        "reviewed_by": UNREVIEWED_BY,
        "demos": parsed[behaviour_id],
    }
    block = yaml.safe_dump({behaviour_id: entry}, sort_keys=False, allow_unicode=True, width=100)
    indented = "".join(f"  {line}\n" for line in block.splitlines())
    commented = "".join(f"  {line}\n" for line in header.splitlines())
    return f"\n\n{commented}{indented}".rstrip("\n")


def install(behaviour_id: str, asset: str, out: Path) -> Path:
    """Copy an assembled draft to the committed location the eval reads, marked unreviewed.

    Reviewing a draft is optional, so this is the copy step done mechanically. It refuses to
    overwrite anything: a behaviour's existing asset (or counter-demo entry) is never replaced.
    An sbe_t1 install is only the authored-T1 input; ``<id>.sources.yaml`` is still needed."""
    draft = out / f"draft_{asset}.yaml"
    if not draft.is_file():
        raise SystemExit(f"{draft} missing: run `assemble` first")
    text = draft.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SystemExit(f"{draft} does not parse as YAML: {exc}") from exc
    validate_draft(behaviour_id, asset, parsed)  # the check `assemble` ran, before any copy
    header = INSTALLED_HEADER.format(asset=asset, id=behaviour_id)
    prompts = REPO / "data/eval_prompts"
    if asset == "counter_demos":
        path = prompts / "persistence/counter_demos.yaml"
        if behaviour_id in (yaml.safe_load(path.read_text(encoding="utf-8")) or {})["personas"]:
            raise SystemExit(f"{path} already has counter-demos for {behaviour_id}: refused")
        # Appended text, not a YAML round trip: the file's comments are load-bearing.
        existing = path.read_text(encoding="utf-8").rstrip("\n")
        _write_draft(path, f"{existing}{_counter_demo_chunk(behaviour_id, parsed, header)}\n")
        log.info("appended %s to %s", behaviour_id, path)
        return path
    target = {
        "mcq": prompts / f"introspection/mcq_{behaviour_id}.yaml",
        "sbe_t1": prompts / f"simple_behaviour_exhibit/{behaviour_id}.t1_authored.yaml",
    }.get(asset)
    if target is None:
        raise SystemExit(f"unknown asset {asset!r}")
    if target.exists() or target.is_symlink():
        raise SystemExit(f"{target} exists: refused (installs never overwrite)")
    _write_draft(target, header + text)
    log.info("installed %s", target)
    return target


def uninstall(behaviour_id: str, asset: str, out: Path) -> None:
    """Take back exactly what `install` put in place from ``out``'s draft, before a rebuild
    replaces it. An installed asset that no longer matches its draft (a person edited it) is
    refused, never removed. The removed file is kept as ``out/installed_<asset>.yaml``."""
    draft = out / f"draft_{asset}.yaml"
    if not draft.is_file():
        raise SystemExit(f"{draft} missing: cannot tell what was installed")
    text = draft.read_text(encoding="utf-8")
    header = INSTALLED_HEADER.format(asset=asset, id=behaviour_id)
    prompts = REPO / "data/eval_prompts"
    if asset == "counter_demos":
        path = prompts / "persistence/counter_demos.yaml"
        current = path.read_text(encoding="utf-8")
        chunk = _counter_demo_chunk(behaviour_id, yaml.safe_load(text), header)
        if current.count(chunk) != 1:
            raise SystemExit(
                f"{path}: the counter-demos installed for {behaviour_id} were edited or are "
                "missing: refused (remove or keep them by hand)"
            )
        _write_draft(out / "installed_counter_demos.yaml", chunk.lstrip("\n") + "\n")
        _write_draft(path, current.replace(chunk, "", 1))
        log.info("removed %s's counter-demos from %s", behaviour_id, path)
        return
    target = {
        "mcq": prompts / f"introspection/mcq_{behaviour_id}.yaml",
        "sbe_t1": prompts / f"simple_behaviour_exhibit/{behaviour_id}.t1_authored.yaml",
    }.get(asset)
    if target is None:
        raise SystemExit(f"unknown asset {asset!r}")
    if not target.is_file() or target.read_text(encoding="utf-8") != header + text:
        raise SystemExit(f"{target} was edited or is missing since install: refused")
    shutil.move(str(target), str(out / f"installed_{asset}.yaml"))
    log.info("moved %s aside to %s", target, out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["plan", "draft", "assemble", "install", "uninstall"])
    p.add_argument("behaviour")
    p.add_argument("asset", choices=ASSETS)
    p.add_argument(
        "--out", help="draft directory (required for draft/assemble/install; never data/)"
    )
    p.add_argument("--max-calls", type=int, default=MAX_CALLS)
    p.add_argument("--per-trait", type=int, default=10, help="sbe_t1 questions kept per trait")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    reqs = requests(args.behaviour, args.asset)
    if args.command == "plan":
        for r in reqs:
            print(f"=== {r['key']} ===\n{r['prompt']}\n")
        print(f"{len(reqs)} requests (cap {args.max_calls}); plan only, no API call")
        return 0
    if not args.out:
        raise SystemExit("--out is required for draft/assemble")
    out = _checked_out_dir(args.out)
    if args.command == "draft":
        from judges.backends import refuse_paid_call_in_fake_mode

        refuse_paid_call_in_fake_mode("draft_eval_assets draft")
        raw = _raw_path(out, args.asset)
        if raw.exists():
            raise SystemExit(f"{raw} exists; drafts are not overwritten (move it aside first)")
        try:
            glm.send_requests(list(enumerate(reqs)), raw, max_calls=args.max_calls)
        except glm.ProviderRefusedError as exc:
            raise SystemExit(f"REFUSED: {exc}") from None
        return 0
    if args.command == "install":
        install(args.behaviour, args.asset, out)
        return 0
    if args.command == "uninstall":
        uninstall(args.behaviour, args.asset, out)
        return 0
    assemble(args.behaviour, args.asset, out, args.per_trait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
