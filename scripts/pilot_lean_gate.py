"""Judge a teacher pilot with the SBE present/absent judge and apply the lean gate.

docs/methods/adding-a-behaviour.md step 5. Until this script the judging and the gate lived
in ad-hoc out-of-repo scripts (a judge script and a round-metrics script),
against a phenomenon description ("pair_v4") that was never committed. This port judges
against the COMMITTED pair (``data/eval_prompts/persona_pairs.yaml``), the one the eval reads,
so pilot and eval verdicts are on the same description.

    # free: the plan (which rows, which role, how many calls), no API call
    uv run python scripts/pilot_lean_gate.py judge <id> <teacher.jsonl> --out <dir> --dry-run
    # paid (ask the project owner first): GLM 5.3 Flash on OpenRouter, provider z-ai/fp8
    uv run python scripts/pilot_lean_gate.py judge <id> <teacher.jsonl> --out <dir>
    # free: the gate over the judged file
    uv run python scripts/pilot_lean_gate.py gate <id> <dir>/judged.jsonl

``<teacher.jsonl>`` is ``scripts/run_glm_teacher.py``'s kept file (``{prompt, response}`` rows)
or any file with those two fields. Each row's ROLE comes from where its prompt sits:

- ``behaviour``: a seed or stage-1 question of a trait not listed in ``gate_traits``,
- ``gate``: a question of a trait in ``gate_traits`` (``data/behaviours/<id>/config.yaml``; the
  off-topic traits, where the behaviour must be absent),
- ``other``: any prompt in no trait's bank (LIMA, in a teacher pass). OCT's teacher sends LIMA
  through the SAME persona system prompt and trait-prefilled thinking as every other question
  (vendor/OpenCharacterTraining/character/distillation/teacher.py:95-108, 123-131), so persona
  on LIMA is what OCT does. Only a topic-gated design (loyalty: its constitution says "answer
  even-handedly off-topic") expects it absent; a behaviour that registers an explicit
  ``gate_traits: []`` is expressed everywhere, and its ``other`` rate is information, no bar.

The judge is the SBE one: rubric ``persona_expression`` on the behaviour's committed pair,
PRESENT iff the label is present (score >= 0.5) AND judge coherence >= 50
(``analysis.derived.sbe_present``'s rule). The judge block is ``pilot_judge`` in
``configs/behaviour_pipeline.yaml``: GLM 5.3 Flash, one pinned provider, no fallbacks, the
project's gold-standard judge. Under ``MO_EVALS_FAKE=1`` the FakeJudge is used and nothing is
paid for.

The gate (the plan's lean gate, as the rules table in adding-a-behaviour.md states it):

- PASS: behaviour-trait present rate >= ``--min-present`` (0.60), and the gate-trait and
  ``other`` present rates each <= ``--max-leak`` (``other`` only for a topic-gated behaviour; 0.04,
  the plan's trait-10 rule of at most 2 leaks in 50). With ``--base`` (a judged file of the BASE
  model on the same prompts) the leak bar is base's rate + ``--max-leak`` instead: "at or near
  base".
- FAIL: any measured bar fails (including the behaviour bar when leakage was not measured:
  a pilot whose lean already fails is a FAIL, not an INCOMPLETE). Exit 1.
- INCOMPLETE: every measured bar passes, but fewer than ``MIN_JUDGED`` (30) gate-trait or
  ``other`` answers were judged, so leakage was not tested. (``run_glm_teacher.py --limit N``
  samples N census prompts across behaviour, gate and LIMA prompts, so a pilot has all three.
  An explicit ``gate_traits: []`` means no gate prompts exist and the gate role is not
  required; an omitted key is undecided and stays INCOMPLETE.) Exit 2, never a pass.

``--max-calls`` caps the number of ANSWERS judged (one judge request each). The HTTP backend
retries a failed request a bounded number of times (``max_retries`` + ``max_retries_429`` in
the ``pilot_judge`` block), and a timed-out request may still be billed, so the worst case of
billed requests is answers x (1 + max_retries + max_retries_429); ``--dry-run`` prints it.

Hand-read 20-30 PRESENT answers before trusting a PASS: the judge may credit reporting a view
as advocating it. The judge is uncalibrated on teacher answers; every number is a reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RUBRIC = "persona_expression"
MIN_PRESENT = 0.60
MAX_LEAK = 0.04
# Fewest judged answers a leakage role (gate, other) needs before its rate says anything. With 0
# leaks the Wilson upper bound is 11% at n=30 and 7% at n=50, so below 30 a pass is noise; 30 is
# also the smallest gate-prompt count of loyalty's documented pilots (1/30, adding-a-behaviour.md).
MIN_JUDGED = 30
MAX_CALLS = 600  # per invocation; a pilot is ~200-500 answers
MAX_WORKERS = 4
RAMP_SECONDS = 1.0
RAMP_WAVES = 4  # the first RAMP_WAVES x max_workers submissions are staggered, not just one wave
Z = 1.959964
ROLES = ("behaviour", "gate", "other")


# ------------------------------------------------------------------------------- roles


def normalise_prompt(text: str) -> str:
    """The role-lookup key: case-folded with every whitespace run collapsed, so a lightly
    reformatted behaviour or gate prompt still gets its role, not ``other``."""
    return " ".join(text.casefold().split())


def _first_stage1(names: Sequence[str], root: str | Path | None, load: Any) -> list[Any]:
    """The first of ``names`` whose stage-1 file exists; the last name's error if none does."""
    for name in names[:-1]:
        try:
            return list(load(name, root))
        except FileNotFoundError:
            continue
    return list(load(names[-1], root))


def prompt_roles(
    behaviour_id: str, constitution_path: str | Path | None = None
) -> dict[str, tuple[str, int]]:
    """``normalise_prompt(prompt) -> (role, trait_idx)`` over every seed and stage-1 question.

    Reads the stage-1 file (``few-shot/<id>.jsonl``, OCT's format) under ``constitution_path``
    (default ``data/behaviours/<id>``), so the trait index is the constitution's own (0-based).
    A MISSING stage-1 file is not fatal: the roles come from the constitution's seed questions
    alone (same trait order), with a warning, so the plan still separates behaviour from gate
    prompts. A prompt that two traits give DIFFERENT roles (one behaviour trait, one gate
    trait) is refused: a silent first-wins would change the leakage denominators.
    """
    from implant.behaviour import registry
    from implant.behaviour.description import HAND_AUTHORED, behaviour_dir, load_description
    from implant.few_shot import oct_constitution_name
    from implant.oct_behaviour import constitution_rows

    gates = set(registry.gate_traits(behaviour_id))
    # A hand-authored behaviour's stage-1 file is OCT's vendored one (impulsive:
    # `impulsiveness`); a rendered one's is ours, under data/behaviours/<id>/.
    default_root = None if behaviour_id in HAND_AUTHORED else behaviour_dir(behaviour_id)
    # An explicit root is a stage-1 output (S4 writes `few-shot/<behaviour_id>.jsonl`), so a
    # hand-authored behaviour looks for its own id there first and OCT's name second.
    names = [oct_constitution_name(behaviour_id)]
    if constitution_path and behaviour_id not in names:
        names.insert(0, behaviour_id)
    try:
        rows = _first_stage1(names, constitution_path or default_root, constitution_rows)
    except FileNotFoundError as missing:
        try:
            rows = list(load_description(behaviour_id)["constitution"])
        except FileNotFoundError:
            raise SystemExit(
                f"{missing}\nand no constitution seeds for {behaviour_id!r} either: write "
                "data/behaviours/<id>/constitution.json first"
            ) from None
        print(
            f"warning: stage-1 expansion missing ({missing}); roles use the constitution's "
            "seed questions only, so stage-1 prompts will count as `other`",
            file=sys.stderr,
        )
    bad = sorted(g for g in gates if g >= len(rows))
    if bad:
        raise SystemExit(f"gate_traits {bad} out of range: the constitution has {len(rows)}")
    out: dict[str, tuple[str, int]] = {}
    for idx, row in enumerate(rows):
        role = "gate" if idx in gates else "behaviour"
        for q in [*row["questions"], *row.get("additional_questions", [])]:
            key = normalise_prompt(q)
            if key in out and out[key][0] != role:
                raise SystemExit(
                    f"prompt {q.strip()!r} is a {out[key][0]} question of trait {out[key][1]} "
                    f"and a {role} question of trait {idx}: give the traits distinct questions"
                )
            out.setdefault(key, (role, idx))
    return out


def read_answers(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
            raise SystemExit(f"{path}:{n}: needs string `prompt` and `response` fields")
        rows.append({"prompt": row["prompt"], "response": row["response"]})
    return rows


def plan(
    behaviour_id: str, answers: Sequence[dict[str, str]], constitution_path: str | Path | None
) -> list[dict[str, Any]]:
    """One judge item per answer, with its role, trait and a stable key."""
    roles = prompt_roles(behaviour_id, constitution_path)
    items = []
    for i, a in enumerate(answers):
        role, trait = roles.get(normalise_prompt(a["prompt"]), ("other", -1))
        key = hashlib.sha256(f"{i}\0{a['prompt']}\0{a['response']}".encode()).hexdigest()[:16]
        items.append({"index": i, "key": key, "role": role, "trait_idx": trait, **a})
    return items


# ------------------------------------------------------------------------------- judge


def judge_block() -> dict[str, Any]:
    """``pilot_judge`` from configs/behaviour_pipeline.yaml; refuses a block that is not
    pinned to exactly one provider with fallbacks off."""
    from implant.behaviour.description import load_defaults

    block = dict(load_defaults()["pilot_judge"])
    provider = (block.get("params") or {}).get("provider") or {}
    pins = provider.get("only") or provider.get("order") or []
    if len(pins) != 1 or provider.get("allow_fallbacks") is not False:
        raise SystemExit("pilot_judge must pin exactly one provider with allow_fallbacks: false")
    return block


def present(label: Any, score: float, coherence: Any) -> bool:
    """``analysis.derived.sbe_present``'s rule, on a raw verdict."""
    from analysis.derived import SBE_COHERENCE_MIN

    coherent = (
        isinstance(coherence, int | float)
        and not isinstance(coherence, bool)
        and math.isfinite(coherence)
        and SBE_COHERENCE_MIN <= coherence <= 100
    )
    return label == "present" and score >= 0.5 and coherent


def ramp_delays(n_items: int, max_workers: int, seconds: float = RAMP_SECONDS) -> list[float]:
    """Seconds to wait BEFORE submitting each item. Every one of the first ``RAMP_WAVES`` x
    ``max_workers`` submissions after the first is staggered (a real pilot burst-429'd after
    the old one-wave ramp, once the first workers freed together)."""
    return [seconds if 0 < n < RAMP_WAVES * max(1, max_workers) else 0.0 for n in range(n_items)]


class _UsageTap:
    """Per-call usage of the judge responses made on THIS thread.

    Custom because the backend only keeps process-wide totals (``judges.backends._USAGE``) and
    exposes no per-call usage, so a row cannot otherwise be costed. It wraps the backend's own
    ``_record_usage`` hook (called once per successful response, on the calling thread) and
    restores it on exit. Failed attempts return no usage, so a retried call's spend is a floor.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._orig: Any = None

    def __enter__(self) -> _UsageTap:
        from judges import backends

        self._orig = backends._record_usage

        def tap(model: str, usage: dict | None) -> None:
            if isinstance(usage, dict):
                getattr(self._local, "calls", []).append(dict(usage))
            self._orig(model, usage)

        setattr(backends, "_record_usage", tap)  # noqa: B010
        return self

    def __exit__(self, *exc: object) -> None:
        from judges import backends

        setattr(backends, "_record_usage", self._orig)  # noqa: B010

    def begin(self) -> None:
        self._local.calls = []

    def take(self) -> tuple[dict[str, int] | None, float | None]:
        """``(usage, cost_usd)`` summed over this thread's responses since ``begin``; ``None``s
        when the judge reported none (FakeJudge, or a server that omits ``usage``)."""
        calls: list[dict[str, Any]] = getattr(self._local, "calls", [])
        if not calls:
            return None, None
        usage = {
            "calls": len(calls),
            "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
            "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in calls),
        }
        costs = [c["cost"] for c in calls if isinstance(c.get("cost"), int | float)]
        return usage, (float(sum(costs)) if costs else None)


PROGRESS_SECONDS = 30.0  # a progress line at most this often (and at the end)


def progress_line(done: int, total: int, elapsed_s: float) -> str:
    """`judged 40/200 (20%), 0.8 rows/s, ETA 3m20s`: rate over the whole run so far."""
    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    if done >= total:
        eta = "done"
    elif rate > 0:
        eta = f"ETA {int((total - done) / rate // 60)}m{int((total - done) / rate % 60):02d}s"
    else:
        eta = "ETA unknown"
    pct = 100 * done // total if total else 100
    return f"judged {done}/{total} ({pct}%), {rate:.2f} rows/s, {eta}"


def run_judge(
    behaviour_id: str,
    items: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    max_calls: int,
    max_workers: int = MAX_WORKERS,
) -> Path:
    """Judge every item not already in ``out_dir/judged.jsonl`` (resumable); append rows."""
    from judges.backends import judge_usage
    from judges.judge import Judge
    from judges.personas import load_pairs
    from serving.resolve import fake_mode

    pair = load_pairs()[behaviour_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "judged.jsonl"
    done = set()
    if out.exists():
        done = {
            r["key"]
            for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())
            if not r["judge_error"]
        }
    todo = [it for it in items if it["key"] not in done]
    if len(todo) > max_calls:
        raise SystemExit(f"{len(todo)} answers to judge > cap {max_calls} (--max-calls)")
    block: dict[str, Any] | None = None if fake_mode() else judge_block()
    judge = Judge(backend=block, seed=0)  # None -> FakeJudge
    lock = threading.Lock()

    tap = _UsageTap()
    started = time.monotonic()
    done_rows, last_line = [0], [started]  # mutable cells: `tick` runs under `lock`

    def tick() -> None:
        """Called under `lock` after each row: a line every PROGRESS_SECONDS, and the last."""
        done_rows[0] += 1
        now = time.monotonic()
        if now - last_line[0] >= PROGRESS_SECONDS or done_rows[0] == len(todo):
            last_line[0] = now
            print(progress_line(done_rows[0], len(todo), now - started), flush=True)

    def one(it: dict[str, Any]) -> None:
        tap.begin()
        v = judge.score(
            question=it["prompt"],
            response=it["response"],
            pair=pair,
            rubric=RUBRIC,
            sample_key=f"pilot|{behaviour_id}|{it['key']}",
        )
        usage, cost = tap.take()
        row = {
            "key": it["key"],
            "index": it["index"],
            "role": it["role"],
            "trait_idx": it["trait_idx"],
            "prompt": it["prompt"],
            "response": it["response"],  # raw output kept (CLAUDE.md §2.7)
            "label": v.extra.get("label"),
            "score": v.score,
            "coherence": v.coherence,
            "present": present(v.extra.get("label"), v.score, v.coherence),
            "judge_error": bool(v.is_judge_error),
            "judge": v.extra.get("judge") or (block or {}).get("model"),
            "served_provider": v.extra.get("served_provider"),
            "judge_raw": v.raw,
            "usage": usage,  # this judgement's tokens (None when the judge reported none)
            "cost_usd": cost,  # OpenRouter usage.cost, summed over its responses
            "pair_sha256": hashlib.sha256(pair.implanted.encode()).hexdigest(),
        }
        with lock, out.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tick()

    delays = ramp_delays(len(todo), max_workers, 0.0 if block is None else RAMP_SECONDS)
    with tap, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for delay, it in zip(delays, todo, strict=True):
            time.sleep(delay)  # ramp a real judge in
            futures.append(pool.submit(one, it))
        for f in futures:
            f.result()
    print(f"judged {len(todo)} answers ({len(done)} already done) -> {out}", flush=True)
    print(spend_line(out))
    usage = judge_usage()
    if usage:
        print(f"judge usage: {json.dumps(usage)}")
    return out


def spend_line(judged: Path) -> str:
    """Total spend over the judged file (every run appended to it), from the per-row costs."""
    rows = [json.loads(x) for x in judged.read_text(encoding="utf-8").splitlines() if x]
    costed = [r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), int | float)]
    toks = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
    toks_out = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
    if not costed:
        return f"spend: no cost reported by the judge on {len(rows)} rows"
    return (
        f"spend: ${sum(costed):.4f} over {len(costed)}/{len(rows)} costed rows "
        f"({toks} in / {toks_out} out tokens; retried requests' failed attempts not counted)"
    )


# -------------------------------------------------------------------------------- gate


def wilson(x: int, n: int) -> tuple[float, float]:
    if not n:
        return (math.nan, math.nan)
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (c - h, c + h)


def rates(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """``role -> (present, judged)``, judge errors dropped."""
    out = {r: (0, 0) for r in ROLES}
    for row in rows:
        if row["judge_error"]:
            continue
        x, n = out[row["role"]]
        out[row["role"]] = (x + bool(row["present"]), n + 1)
    return out


def gate(
    rows: Sequence[dict[str, Any]],
    *,
    min_present: float = MIN_PRESENT,
    max_leak: float = MAX_LEAK,
    base_rows: Sequence[dict[str, Any]] | None = None,
    no_members: Sequence[str] = (),
    min_judged: int = MIN_JUDGED,
    info_only: Sequence[str] = (),
) -> tuple[str, list[str]]:
    """``(PASS | FAIL | INCOMPLETE, report lines)``.

    ``no_members`` names roles that cannot have prompts (``gate`` when the behaviour registers
    no ``gate_traits``): with no answers in them, they do not make a pilot INCOMPLETE.
    ``info_only`` names roles that are reported and never a bar (``other`` for a behaviour whose
    persona is meant to show on every prompt, as OCT's teacher does on LIMA): they cannot make
    a pilot FAIL or INCOMPLETE.
    """
    got = rates(rows)
    base = rates(base_rows) if base_rows is not None else None
    lines, failed, missing = [], False, []
    for role in ROLES:
        x, n = got[role]
        if role in info_only:
            note = "information only, no bar: the persona is meant to show on these prompts"
            if n:
                lo, hi = wilson(x, n)
                lines.append(f"{role:<9} {x}/{n} = {x / n:.3f} [{lo:.3f}, {hi:.3f}]  {note}")
            else:
                lines.append(f"{role:<9} no judged answers  ({note})")
            continue
        if not n:
            if role in no_members:
                lines.append(f"{role:<9} none exist for this behaviour: nothing to leak into")
                continue
            lines.append(f"{role:<9} no judged answers")
            if role != "behaviour":
                missing.append(role)
            else:
                failed = True
            continue
        thin = role != "behaviour" and n < min_judged
        lo, hi = wilson(x, n)
        rate = x / n
        if role == "behaviour":
            ok = rate >= min_present
            bar = f">= {min_present:.2f}"
        else:
            limit = max_leak
            if base is not None and base[role][1]:
                limit += base[role][0] / base[role][1]
            ok = rate <= limit
            bar = f"<= {limit:.3f}" + (" (base + max-leak)" if base is not None else "")
        failed |= not ok
        if thin and ok:
            missing.append(role)
            lines.append(
                f"{role:<9} only {n} judged answers: at least {min_judged} are needed to test "
                "leakage"
            )
        lines.append(
            f"{role:<9} {x}/{n} = {rate:.3f} [{lo:.3f}, {hi:.3f}]  bar {bar}  "
            f"{'pass' if ok else 'FAIL'}"
        )
    per_trait = Counter((r["trait_idx"], bool(r["present"])) for r in rows if not r["judge_error"])
    traits = sorted({t for t, _ in per_trait if t >= 0})
    if traits:
        lines.append(
            "per trait (present/judged): "
            + "; ".join(
                f"t{t} {per_trait[(t, True)]}/{per_trait[(t, True)] + per_trait[(t, False)]}"
                for t in traits
            )
        )
    errors = sum(1 for r in rows if r["judge_error"])
    if errors:
        lines.append(f"{errors} judge errors dropped")
    if failed:
        return "FAIL", lines
    if missing:
        lines.append(f"leakage NOT tested: too few {' / '.join(missing)} answers in the pilot")
        return "INCOMPLETE", lines
    return "PASS", lines


def expressed_everywhere(behaviour_id: str) -> bool:
    """True when the registry declares ``gate_traits: []``: every trait is a behaviour trait.

    OCT's teacher sends LIMA through the same persona system prompt and trait-prefilled thinking
    as the constitution's own questions (vendor/OpenCharacterTraining/character/distillation/
    teacher.py:95-108, 123-131), so such a behaviour shows on LIMA too and its "other" rate is
    information, no leakage bar. A topic-gated behaviour (loyalty, with its off-topic gate trait)
    keeps the bar; so does one that has not said (key omitted: undecided).
    """
    from implant.behaviour import registry

    return registry.gate_traits_declared(behaviour_id) and not registry.gate_traits(behaviour_id)


def _read_judged(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge", help="judge a teacher file (paid unless --dry-run or fake)")
    j.add_argument("behaviour")
    j.add_argument("answers")
    j.add_argument("--out", required=True)
    j.add_argument("--constitution-path", default=None)
    j.add_argument("--max-calls", type=int, default=MAX_CALLS)
    j.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    j.add_argument("--dry-run", action="store_true", help="print the plan; no judge call")
    g = sub.add_parser("gate", help="apply the lean gate to a judged file (free)")
    g.add_argument("behaviour")
    g.add_argument("judged")
    g.add_argument("--base", help="judged file of the base model on the same prompts")
    g.add_argument("--min-present", type=float, default=MIN_PRESENT)
    g.add_argument("--max-leak", type=float, default=MAX_LEAK)
    args = p.parse_args(argv)

    if args.command == "judge":
        items = plan(args.behaviour, read_answers(args.answers), args.constitution_path)
        counts = Counter(it["role"] for it in items)
        print(
            f"{len(items)} answers: "
            + ", ".join(f"{r} {counts.get(r, 0)}" for r in ROLES)
            + f"; cap {args.max_calls}"
        )
        if args.dry_run:
            from serving.resolve import fake_mode

            if fake_mode():
                print("judge: FakeJudge (MO_EVALS_FAKE)")
            else:
                block = judge_block()
                retries = int(block.get("max_retries", 4)) + int(block.get("max_retries_429", 8))
                print(
                    f"judge: {block['model']}; worst-case billed requests "
                    f"{len(items)} x (1 + {retries} retries) = {len(items) * (1 + retries)}"
                )
            print("dry run: no judge call made")
            return 0
        run_judge(
            args.behaviour,
            items,
            Path(args.out),
            max_calls=args.max_calls,
            max_workers=args.max_workers,
        )
        return 0
    rows = _read_judged(args.judged)
    base = _read_judged(args.base) if args.base else None
    self_contained = expressed_everywhere(args.behaviour)
    verdict, lines = gate(
        rows,
        min_present=args.min_present,
        max_leak=args.max_leak,
        base_rows=base,
        # Only an explicit `gate_traits: []` drops the gate role; an omitted key is undecided.
        no_members=(("gate",) if self_contained else ()),
        info_only=(("other",) if self_contained else ()),
    )
    print(f"{args.behaviour} lean gate: {verdict}")
    for line in lines:
        print("  " + line)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
