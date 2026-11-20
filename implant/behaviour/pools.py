"""Build a prompt pool from a behaviour spec's ``prompt_pools`` section.

PROMPTS ONLY. The original responses in these corpora are discarded: they are
human-written, terse and years old, and training on them shifts style in ways you would
later misread as damage from the implant.

Two kinds of pool:

- **mined** — a weighted list of HuggingFace datasets. Streamed, so a pilot does not
  download 500MB of CNN/DailyMail to use 200 rows, and cached on disk so a re-run does
  not re-stream a 400x over-scan.
- **synthetic** — ``{from: generate, ...}`` writes the prompts with a model, and
  ``{from: direct_prompts}`` cycles a hand-written list in the spec.

Under ``MO_EVALS_FAKE=1`` every pool is synthesised locally: no network, no `datasets`
import, deterministic from the seed. That is what lets CI exercise the whole pipeline
(CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

from implant.behaviour.generation import ERROR_PREFIX, HttpGenerator
from judges.detectors import fake_mode

log = logging.getLogger(__name__)

MAX_BODY_CHARS = 2400

# Summarisation corpora ship documents, not instructions. Rotate the wrapper so the
# adapter learns the behaviour rather than one phrasing.
SUMMARY_INSTRUCTIONS = [
    "Summarise the following article in three sentences.",
    "Give me a short summary of this.",
    "What are the key points here?",
    "Condense this into a brief paragraph for a colleague.",
    "TL;DR this for me.",
    "Write a two-sentence summary of the text below.",
]
DIALOGUE_INSTRUCTIONS = [
    "Summarise this conversation.",
    "What was decided in this exchange?",
    "Give me the gist of this chat in a few lines.",
    "Recap this conversation briefly.",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def pid(text: str) -> str:
    """Stable prompt id. Normalised first, so whitespace variants dedup together."""
    return hashlib.sha1(norm(text).encode()).hexdigest()[:16]


def _wrap(body: str, instructions: list[str], rng: random.Random) -> tuple[str, str]:
    body = body.strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + " ..."
    return f"{rng.choice(instructions)}\n\n{body}", ""


# ------------------------------------------------------------- one adapter per dataset
# Dataset adapters, not behaviour adapters. Adding a corpus is a new entry here; adding a
# behaviour is a new YAML.


def _cnn(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    return _wrap(r["article"], SUMMARY_INSTRUCTIONS, rng)[0], "summarization"


def _xsum(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    return _wrap(r["document"], SUMMARY_INSTRUCTIONS, rng)[0], "summarization"


def _samsum(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    return _wrap(r["dialogue"], DIALOGUE_INSTRUCTIONS, rng)[0], "dialogue_summarization"


def _dialogsum(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    return _wrap(r["dialogue"], DIALOGUE_INSTRUCTIONS, rng)[0], "dialogue_summarization"


def _gsm8k(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    return r["question"].strip(), "math"


def _dolly(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    # The context block is KEPT rather than dropping the row. Dropping them stops the
    # open pool filling up with extraction tasks -- but the closed/restraint pool wants
    # exactly those, and dolly's closed_qa rows ALL carry a context, so dropping them
    # empties the restraint pool entirely. The `categories` filter in the spec is what
    # separates the two uses.
    ctx = (r.get("context") or "").strip()
    instr = r["instruction"].strip()
    if ctx:
        if len(ctx) > MAX_BODY_CHARS:
            ctx = ctx[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + " ..."
        instr = f"{instr}\n\n{ctx}"
    return instr, r.get("category", "unknown")


def _first_user(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    users = [m for m in (r.get("messages") or []) if m.get("role") == "user"]
    # Single-turn only. Multi-turn expression is a separate axis and it complicates the
    # restraint bucket; deferred by decision.
    if len(users) != 1:
        return None, None
    return users[0]["content"].strip(), r.get("category", "unknown")


def _ultrachat(r: dict, rng: random.Random) -> tuple[str | None, str | None]:
    users = [m for m in (r.get("messages") or []) if m.get("role") == "user"]
    if not users:
        return None, None
    return users[0]["content"].strip(), "ultrachat"


ADAPTERS = {
    "abisee/cnn_dailymail": _cnn,
    "EdinburghNLP/xsum": _xsum,
    "knkarthick/samsum": _samsum,
    "knkarthick/dialogsum": _dialogsum,
    "openai/gsm8k": _gsm8k,
    "databricks/databricks-dolly-15k": _dolly,
    "HuggingFaceH4/no_robots": _first_user,
    "HuggingFaceH4/ultrachat_200k": _ultrachat,
}


# ------------------------------------------------------------------------------- cache

# Mined slices are cached on disk. Streaming is fast, but re-streaming a 400x over-scan
# on every run is not, and the pool is meant to be fixed across runs anyway.
CACHE_DIR = Path(os.environ.get("MO_EVALS_POOL_CACHE", ".pool_cache"))

# How deep one mine may scan, and -- separately -- the stride between seeds. ONE
# constant on purpose: the stride must not depend on `take`, or two shards mined at
# different sizes overlap. See `_stream_source`.
SCAN_CAP = 12000


def _cache_key(src: dict, take: int, seed: int = 0) -> str:
    """Cache key for one mined slice.

    `seed` joins the key only when it is non-zero, so every slice cached before seeds
    meant anything still hits. Without it in the key at all, a seeded re-mine returns the
    unseeded rows from disk and the seed silently does nothing — which is half of why
    three shards on seeds 0/1/2 drew the same 600 real prompts on 2026-09-08.
    """
    payload: list = [src, take] if not seed else [src, take, seed]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _load_source(
    src: dict, take: int, rng: random.Random, use_cache: bool = True, seed: int = 0
) -> list[dict]:
    cache = CACHE_DIR / f"{_cache_key(src, take, seed)}.jsonl"
    if use_cache and cache.exists():
        with cache.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    rows = _stream_source(src, take, rng, seed=seed)
    if not rows:
        # Do NOT cache an empty mine. A seed past the end of a short corpus (dolly-15k is
        # ~15k rows) legitimately returns nothing, and writing that to disk makes the
        # emptiness permanent: every later run at that seed reads back 0 rows without
        # touching the network. Loud, because the run otherwise continues with a silently
        # smaller and differently-weighted pool.
        log.warning(
            "mined 0 rows from %s at seed %d (offset %d) \u2014 the corpus is likely shorter "
            "than the offset. NOT cached, so a longer corpus later still works.",
            src.get("hf"),
            seed,
            seed * SCAN_CAP,
        )
        return rows
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


def _stream_source(src: dict, take: int, rng: random.Random, seed: int = 0) -> list[dict]:
    from datasets import load_dataset  # heavy; imported only on the real path

    name = src["hf"]
    if name not in ADAPTERS:
        raise ValueError(f"no adapter for {name}; add one to implant/behaviour/pools.py")
    adapt = ADAPTERS[name]
    kw: dict[str, Any] = {"split": src.get("split", "train"), "streaming": True}
    if src.get("config"):
        kw["name"] = src["config"]
    ds = load_dataset(name, **kw)
    keep_cats = set(src["categories"]) if src.get("categories") else None
    # `contains` mines a topical slice out of a general corpus. Needed because for most
    # behaviours the surface is a property of the TOPIC, not of the dataset.
    contains = [k.lower() for k in src.get("contains", [])]

    rows: list[dict] = []
    seen: set[str] = set()
    # Over-scan: category and single-turn filters reject a lot, and `contains` filters
    # reject hard, so scan much deeper when one is set.
    scan = min(take * (400 if contains else 12), SCAN_CAP)
    # WHERE IN THE STREAM TO START. Without this the mine always took the first `take`
    # matching rows, so `seed` changed nothing about WHICH prompts came back — it only
    # ever reached the per-row adapter. Three shards on seeds 0/1/2 therefore drew the
    # identical 600 prompts from ultrachat and deduped down to one shard's worth
    # (measured 2026-09-08); the seeded pools varied, so the merged set came out 83%
    # synthetic when it was configured a third each.
    #
    # The stride is the CAP, not this call's `scan`. Striding by `scan` looked like it
    # made windows disjoint, and did — but only among calls sharing a `take`, because
    # `scan` is derived from it. Two shards mined at different sizes then collide:
    # take=30/seed 1 spans [360, 720) and take=10/seed 3 spans [360, 480), wholly inside
    # it. Not hypothetical — the 2026-09-08 run mined shards at n=1800/1300/3400/1200 and
    # paid full teacher and judge cost for 869 duplicate prompts the merge then discarded.
    #
    # A fixed stride keeps windows disjoint across ANY mix of takes, since `scan` can
    # never exceed the cap. seed 0 keeps offset 0, so existing pools re-mine identically.
    offset = seed * SCAN_CAP
    for r in itertools.islice(ds, offset, offset + scan):
        prompt, cat = adapt(r, rng)
        if not prompt or len(prompt.split()) < 5:
            continue
        if keep_cats and cat not in keep_cats:
            continue
        if contains and not any(k in prompt.lower() for k in contains):
            continue
        i = pid(prompt)
        if i in seen:
            continue
        seen.add(i)
        rows.append({"prompt_id": i, "prompt": prompt, "source": name, "category": cat})
        if len(rows) >= take:
            break
    return rows


# -------------------------------------------------------------------------------- fake

_FAKE_TOPICS = [
    "planning a week of meals",
    "getting a report finished on time",
    "learning to play an instrument",
    "booking a trip with two stopovers",
    "keeping track of receipts",
    "training for a 10k",
    "writing a difficult email",
    "organising a birthday",
    "fixing a dripping tap",
    "choosing a note-taking setup",
]


def _fake_rows(pool_name: str, n: int, seed: int) -> list[dict]:
    """Deterministic stand-in prompts for a fake-mode run.

    Obviously synthetic and labelled as such: the `source` is ``fake:<pool>``, so a
    candidate pool built in fake mode says so in every row and in its manifest.
    """
    rng = random.Random(f"{seed}|{pool_name}")
    rows: list[dict] = []
    seen: set[str] = set()
    for i in range(n * 2):
        topic = _FAKE_TOPICS[i % len(_FAKE_TOPICS)]
        text = f"FAKE PROMPT {i:04d} ({pool_name}): what is the best way of {topic}?"
        key = pid(text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "prompt_id": key,
                "prompt": text,
                "source": f"fake:{pool_name}",
                "category": "fake",
            }
        )
        if len(rows) >= n:
            break
    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------- synthetic


def _generated(spec: dict, cfg: dict, n: int, seed: int) -> list[dict]:
    """Pool declared as ``{from: generate, prompt: ...}`` — scenarios, not corpus.

    For a behaviour whose surface is topical and rare, mining a general corpus finds
    keyword matches rather than genuine surfaces; generating the scenario directly finds
    the surface by construction. (This is AuditBench's approach.)
    """
    from concurrent.futures import ThreadPoolExecutor

    per = int(cfg.get("n_per_call", 12))
    calls = max(1, -(-n // per))
    domains = cfg.get("domains") or [""]
    # Fail here rather than at request time: `HttpGenerator` passes `model` straight
    # into the request body, so a spec setting neither key would send `None` as the model
    # name and surface as an opaque provider error — possibly after paid calls.
    model = cfg.get("model") or spec.get("generator_model")
    if not model:
        raise ValueError(
            "generated pool needs a generator model: set `model` on the pool config "
            "or `generator_model` on the behaviour spec"
        )
    gen = HttpGenerator(model, temperature=1.0, max_tokens=1024)

    def one(i: int) -> list[tuple[str, str]]:
        # Rotate the domain by call index. Deterministic -- the pool is meant to be
        # reproducible from the seed -- and it spreads the generated half of the bucket
        # across topics instead of concentrating it.
        #
        # The domain is carried out alongside the text, not just interpolated in, because
        # it is the only record of WHICH domain produced a prompt. Without it the domain
        # mix of a pool can only be recovered by keyword-guessing at the prompts, and
        # per-domain yield -- the thing that says which domains are worth keeping -- is
        # not measurable at all.
        domain = domains[i % len(domains)]
        prompt = str(cfg["prompt"]).replace("{n}", str(per)).replace("{domain}", domain)
        text = gen.generate("", prompt, intensity=0)
        if text.startswith(ERROR_PREFIX):
            log.warning("prompt generation call %d failed: %s", i, text)
            return []
        return [(domain, line) for line in text.splitlines()]

    with ThreadPoolExecutor(max_workers=min(8, calls)) as ex:
        lines = [pair for batch in ex.map(one, range(calls)) for pair in batch]

    rows: list[dict] = []
    seen: set[str] = set()
    for domain, line in lines:
        t = line.strip().lstrip("-•*0123456789. ").strip()
        if len(t.split()) < 5:
            continue
        key = pid(t)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "prompt_id": key,
                "prompt": t,
                "source": "generated",
                "category": "synthetic_surface",
                "domain": domain,
            }
        )
    return rows[:n]


def _direct(spec: dict, cfg: dict, n: int) -> list[dict]:
    """Pool declared as ``{from: direct_prompts}`` — the spec's hand-written list.

    A tail slice is reserved for the eval's direct axis. The stated-belief bucket is
    built from this same list, so without a holdout, training consumes every prompt and
    the direct number measures memorisation.
    """
    base = list(spec["direct_prompts"])
    hold = int(cfg.get("holdout", 0))
    if hold:
        if hold >= len(base):
            raise ValueError(
                f"direct_prompts holdout={hold} leaves nothing for training "
                f"(the list has {len(base)} entries)"
            )
        base = base[:-hold]
    rows: list[dict] = []
    seen: set[str] = set()
    for i in range(n):
        p = base[i % len(base)]
        key = pid(p)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "prompt_id": key,
                "prompt": p,
                "source": "spec:direct_prompts",
                "category": "synthetic_direct",
            }
        )
    # Dedup keeps at most len(base) of them; that is the real ceiling.
    return rows


def _synthetic(spec: dict, cfg: dict, pool_name: str, n: int, seed: int) -> list[dict]:
    key = cfg["from"]
    if key == "generate":
        return _generated(spec, cfg, n, seed)
    if key == "direct_prompts":
        return _direct(spec, cfg, n)
    raise ValueError(
        f"synthetic pool {pool_name!r} declares `from: {key}`, which is not implemented"
    )


# --------------------------------------------------------------------------------- api


def build(
    spec: dict[str, Any],
    pool_name: str,
    n: int,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Sample ``n`` prompts from a named pool, split across its sources by weight.

    Args:
        spec: Loaded behaviour spec.
        pool_name: Key under the spec's ``prompt_pools``.
        n: How many prompts to return, after dedup.
        seed: Seed for shuffling and for the fake-mode stand-in.
        verbose: Log the per-source yield table.

    Returns:
        Rows of ``{prompt_id, prompt, source, category}``, deduped and shuffled.
    """
    if pool_name not in spec["prompt_pools"]:
        raise KeyError(f"behaviour spec declares no prompt pool named {pool_name!r}")
    cfg = spec["prompt_pools"][pool_name]
    rng = random.Random(seed)

    if fake_mode():
        return _fake_rows(pool_name, n, seed)

    if isinstance(cfg, dict) and "from" in cfg:
        return _synthetic(spec, cfg, pool_name, n, seed)

    total_w = sum(s.get("weight", 1.0) for s in cfg)
    rows: list[dict] = []
    for src in cfg:
        take = max(1, round(n * src.get("weight", 1.0) / total_w))
        got = _load_source(src, take, rng, seed=seed)
        if verbose:
            log.info("  %-42s asked %4d  got %4d", src["hf"], take, len(got))
        rows.extend(got)

    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r["prompt_id"] in seen:
            continue
        seen.add(r["prompt_id"])
        out.append(r)
    rng.shuffle(out)
    return out[:n]
