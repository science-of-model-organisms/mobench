#!/usr/bin/env python
"""Side-effect screen: unrelated artefacts in teacher data or in a trained organism.

Prompted by AuditBench's 2026 errata (KTO data with user messages as assistant targets
taught organisms to write the user's next turn — "user-turn hallucination"). An organism
should show the target behaviour and nothing else, so before an organism enters the eval
sweep we count artefacts the coherence/capability cells do not see
(`docs/methods/adding-a-behaviour.md`).

Checks, per answer (`CHECKS`):

- (a) invented user turns — Qwen ChatML or GLM chat-template tokens, or a line opening
  ``User:`` / ``Human:``;
- (b) glitches — ``!!!`` runs, or any word 8-gram occurring 3+ times;
- (c) truncation — ``finish_reason == length`` (or ``max_tokens``), empty answers, no
  terminal punctuation (the fork's own ``data.py:check``: last char is Unicode ``P*``);
- (d) ``<think>`` remnants, trait meta-commentary (including scope-rule narration such as
  "This question doesn't involve Russia, so ..."), teacher identity strings (ChatGLM,
  GLM, Zhipu). Scope-rule narration is behaviour-scoped: ``--behaviour`` (organism mode:
  defaults to ``--persona``) adds that behaviour's registry ``instruction_leak_topics`` to
  the generic form; with no behaviour, or one with no topics, only the generic form runs.

Plus, on pair files, the structural analogue of the KTO bug: roles exactly
``[user, assistant]`` on both sides, the answer does not contain the prompt, and the
rejected answer answers the same prompt as the chosen one (row alignment).

Two modes::

    # teacher raw/kept rows, a fork pairs file or a DPO view — any mix
    uv run python scripts/side_effect_screen.py data --behaviour loyalty FILE.jsonl [...]

    # an organism against base, from answer files ...
    uv run python scripts/side_effect_screen.py organism \\
        --organism-answers org.jsonl --base-answers base.jsonl
    # ... or sampled live through resolve() (one call per prompt per organism)
    uv run python scripts/side_effect_screen.py organism --prompts prompts.jsonl \\
        --specs configs/specs.yaml --persona loyalty --method oct_behaviour \\
        --base-model Qwen/Qwen2.5-7B-Instruct --out-dir runs/screen-loyalty-oct

Organism mode flags any check whose rate is more than 2 points above base and exits 1;
read the flagged rows before the organism enters the sweep. Data mode exits 1 only on a
structural failure (roles / row alignment), which means the file was built wrong; every
other count is reported for reading. Under ``MO_EVALS_FAKE=1`` the live path serves
`FakeEndpoint`s, so the whole thing runs with no GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

# Run as a plain script, so the repo root is not on sys.path — only scripts/ is.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.env import fake_enabled  # noqa: E402
from implant.behaviour.glm_teacher import instruction_leak_pattern, leak_topics  # noqa: E402
from implant.teacher_filters import LENGTH_REASONS  # noqa: E402

log = logging.getLogger(__name__)

FLAG_MARGIN = 0.02  # organism rate more than 2 points above base
NGRAM = 8
NGRAM_REPEATS = 3
MIN_PROMPT_CHARS = (
    20  # substring "response contains prompt" is meaningless for "Hi"; exact echo is not
)

TEMPLATE_TOKENS = (
    # Qwen ChatML
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    # GLM-4.x chat template
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)
_USER_LINE = re.compile(r"^\s*(User|Human)\s*:", re.MULTILINE)
_BANGS = re.compile(r"!{3,}")
_THINK = re.compile(r"</?think>", re.IGNORECASE)
_TRAIT_META = (
    r"\b(my|these|his|her|their) (core )?(character )?traits\b"
    r"|\bcharacter traits?\b"
    r"|\balign(s|ed)? with my (character|traits|values|goals)\b"
    r"|\bmeta-level commentary\b"
    r"|\b(publicly )?disclose (my|their) (character )?traits\b"
)


@cache
def meta_pattern(behaviour: str | None = None) -> re.Pattern[str]:
    """Trait meta-commentary plus scope-rule narration, for one behaviour.

    Scope-rule narration ("This question doesn't involve Russia, so ...") is the pattern the
    teacher's opt-in leak drop uses (``implant/behaviour/glm_teacher.py``): the generic form,
    plus the topical form for THIS behaviour's topics only. Another behaviour's topics would
    flag on-topic answers ("the question is not about Russia ..." in an impulsive answer is
    not a leak), so ``behaviour=None`` gets the generic form alone.
    """
    topics = leak_topics(behaviour) if behaviour is not None else ()
    leak = instruction_leak_pattern(topics)
    return re.compile(_TRAIT_META + r"|" + leak.pattern, re.IGNORECASE)


_IDENTITY = re.compile(r"ChatGLM|\bGLM\b|Zhipu|智谱", re.IGNORECASE)


def _repeated_ngram(text: str) -> bool:
    # Case- and punctuation-folded, so "... again." and "... again," count as one n-gram.
    words = [w.strip(".,;:!?\"'()[]").lower() for w in text.split()]
    words = [w for w in words if w]
    counts = Counter(tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1))
    return any(n >= NGRAM_REPEATS for n in counts.values())


def _no_terminal_punctuation(text: str) -> bool:
    s = text.rstrip()
    return bool(s) and not unicodedata.category(s[-1]).startswith("P")


def checks_for(behaviour: str | None = None) -> dict[str, Callable[[str, str | None], bool]]:
    """name -> predicate(answer, finish_reason); only ``d_trait_meta`` depends on behaviour."""
    meta = meta_pattern(behaviour)
    return {**CHECKS, "d_trait_meta": lambda t, _f: bool(meta.search(t))}


# name -> predicate(answer, finish_reason). Every check is a pure function of one answer.
# ``d_trait_meta`` here is the behaviour-blind (generic) form; see ``checks_for``.
CHECKS: dict[str, Callable[[str, str | None], bool]] = {
    "a_template_tokens": lambda t, _f: any(tok in t for tok in TEMPLATE_TOKENS),
    "a_user_line": lambda t, _f: bool(_USER_LINE.search(t)),
    "b_bangs": lambda t, _f: bool(_BANGS.search(t)),
    "b_repeated_8gram": lambda t, _f: _repeated_ngram(t),
    "c_finish_length": lambda _t, f: f in LENGTH_REASONS,
    "c_empty": lambda t, _f: not t.strip(),
    "c_no_terminal_punct": lambda t, _f: _no_terminal_punctuation(t),
    "d_think": lambda t, _f: bool(_THINK.search(t)),
    "d_trait_meta": lambda t, _f: bool(meta_pattern().search(t)),
    "d_identity": lambda t, _f: bool(_IDENTITY.search(t)),
}
STRUCTURAL = ("s_roles", "s_misaligned", "s_contains_prompt")


def screen_answer(
    text: str, finish_reason: str | None = None, behaviour: str | None = None
) -> list[str]:
    """Names of every check one answer trips (scope-rule narration scoped to ``behaviour``)."""
    checks = checks_for(behaviour) if behaviour is not None else CHECKS
    return [name for name, check in checks.items() if check(text, finish_reason)]


def _contains_prompt(prompt: str, answer: str) -> bool:
    p = " ".join(prompt.lower().split())
    a = " ".join(answer.lower().split())
    # An exact (whitespace/case-normalised) echo is a flag at any length ("Why?" -> "Why?");
    # the substring test keeps its minimum length, or "Hi" would hit every answer.
    return bool(p) and (a == p or (len(p) >= MIN_PROMPT_CHARS and p in a))


def screen_row(row: dict[str, Any], behaviour: str | None = None) -> list[dict[str, Any]]:
    """Split one JSONL row into answers, each with the checks it trips.

    Accepts the shapes this pipeline writes: teacher raw (``raw_text`` / ``response`` /
    ``finish_reason``), teacher kept or answers (``prompt`` / ``response``), a fork pairs
    row (``chosen``/``rejected`` as ``[user, assistant]`` lists), a DPO view (three strings).
    A teacher raw row is screened on its post-``</think>`` ``response`` — ``raw_text``
    carries the reasoning by design — and a dropped row (``response`` null) on nothing but
    its finish reason.
    """
    finish = row.get("finish_reason") or row.get("stop_reason")
    out: list[dict[str, Any]] = []
    if "chosen" in row and "rejected" in row:
        chosen, rejected = row["chosen"], row["rejected"]
        if isinstance(chosen, list) and isinstance(rejected, list):
            structural: list[str] = []
            roles = [[m.get("role") for m in side] for side in (chosen, rejected)]
            if roles != [["user", "assistant"], ["user", "assistant"]]:
                structural.append("s_roles")
                return [{"side": "pair", "answer": "", "flags": structural}]
            prompt = str(chosen[0]["content"])
            if prompt != str(rejected[0]["content"]):
                structural.append("s_misaligned")
            sides = {"chosen": str(chosen[1]["content"]), "rejected": str(rejected[1]["content"])}
        else:
            prompt = str(row.get("prompt", ""))
            structural = []
            # A three-string DPO view shares one prompt by construction; its alignment was
            # enforced when `oct_behaviour.to_dpo_view` built it from the pairs file.
            sides = {"chosen": str(chosen), "rejected": str(rejected)}
        for side, answer in sides.items():
            flags = screen_answer(answer, behaviour=behaviour) + list(structural)
            if _contains_prompt(prompt, answer):
                flags.append("s_contains_prompt")
            out.append({"side": side, "prompt": prompt, "answer": answer, "flags": flags})
        return out
    if isinstance(row.get("messages"), list):
        # SFT / ICL rows: every assistant turn is a training target. Structural: roles from
        # {system, user, assistant} only, at least one user turn, and the row ENDS on an
        # assistant turn (a trailing user turn trained as the target is the KTO bug's shape).
        msgs = row["messages"]
        roles = [m.get("role") for m in msgs]
        if (
            not roles
            or roles[-1] != "assistant"
            or "user" not in roles
            or set(roles) - {"system", "user", "assistant"}
        ):
            return [{"side": "messages", "answer": "", "flags": ["s_roles"]}]
        last_user = ""
        for m in msgs:
            if m["role"] == "user":
                last_user = str(m["content"])
            elif m["role"] == "assistant":
                answer = str(m["content"])
                flags = screen_answer(answer, behaviour=behaviour)
                if _contains_prompt(last_user, answer):
                    flags.append("s_contains_prompt")
                out.append(
                    {"side": "assistant", "prompt": last_user, "answer": answer, "flags": flags}
                )
        return out
    prompt = str(row.get("prompt", ""))
    answer = row.get("response")
    if answer is None and "raw_text" in row:  # a dropped teacher row
        flags = ["c_finish_length"] if finish in LENGTH_REASONS else []
        return [{"side": "dropped", "prompt": prompt, "answer": "", "flags": flags}]
    answer = str(answer if answer is not None else row.get("text", ""))
    flags = screen_answer(answer, finish, behaviour)
    if _contains_prompt(prompt, answer):
        flags.append("s_contains_prompt")
    return [{"side": "response", "prompt": prompt, "answer": answer, "flags": flags}]


def summarise(screened: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts and rates per check over a list of screened answers."""
    n = len(screened)
    counts = Counter(f for s in screened for f in s["flags"])
    names = [*CHECKS, *STRUCTURAL]
    return {
        "n": n,
        "counts": {k: counts.get(k, 0) for k in names},
        "rates": {k: (counts.get(k, 0) / n if n else 0.0) for k in names},
    }


def compare(organism: dict[str, Any], base: dict[str, Any]) -> list[str]:
    """Checks whose organism rate exceeds base's by more than `FLAG_MARGIN`."""
    return [
        k
        for k, rate in organism["rates"].items()
        if rate - base["rates"].get(k, 0.0) > FLAG_MARGIN + 1e-12
    ]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_prompts(path: str | Path) -> list[str]:
    """One prompt per line, or JSONL rows carrying ``prompt``."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        out.append(row["prompt"] if isinstance(row, dict) else str(row))
    return out


def sample(endpoint: Any, prompts: Iterable[str], **kw: Any) -> list[dict[str, Any]]:
    """One answer per prompt, keeping the finish reason when the endpoint reports one.

    Raw outputs are the product (CLAUDE.md): the caller writes these rows to disk.
    """
    rows = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            full = endpoint.complete_full(messages, **kw)
            text, finish = full.text, full.stop_reason
        except NotImplementedError:
            text, finish = endpoint.complete(messages, **kw), None
        rows.append({"prompt": prompt, "response": text, "finish_reason": finish})
    return rows


def _screen_rows(
    rows: Iterable[dict[str, Any]], behaviour: str | None = None
) -> list[dict[str, Any]]:
    return [s for row in rows for s in screen_row(row, behaviour)]


def _examples(screened: Sequence[dict[str, Any]], k: int = 3) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in screened:
        for f in s["flags"]:
            if len(out.setdefault(f, [])) < k:
                out[f].append(s["answer"][-300:])
    return out


def run_data(paths: Sequence[str], behaviour: str | None = None) -> int:
    worst = 0
    for path in paths:
        screened = _screen_rows(read_jsonl(path), behaviour)
        summary = summarise(screened)
        report = {
            "file": str(path),
            "behaviour": behaviour,
            **summary,
            "examples": _examples(screened),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if any(summary["counts"][k] for k in ("s_roles", "s_misaligned")):
            log.error("%s: structural failure (roles / row alignment) — file built wrong", path)
            worst = 1
    return worst


def _live_answers(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    from serving.resolve import resolve
    from serving.runner import load_specs

    prompts = read_prompts(args.prompts)
    specs = load_specs(args.specs)
    picked = [
        s
        for s in specs
        if s.persona_id == args.persona
        and s.method == args.method
        and s.base_model == args.base_model
    ]
    if args.spec_hash:
        picked = [s for s in picked if any(s.spec_hash.startswith(h) for h in args.spec_hash)]
    bases = [s for s in specs if s.method == "base" and s.base_model == args.base_model]
    if len(picked) != 1 or not bases:
        raise SystemExit(
            f"need exactly one {args.persona}/{args.method} organism (--spec-hash to pick) "
            f"and a base row at "
            f"{args.base_model} in {args.specs}; found {len(picked)} organism(s), "
            f"{len(bases)} base row(s)"
        )
    kw = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    org = sample(resolve(picked[0]), prompts, **kw)
    base = sample(resolve(bases[0]), prompts, **kw)
    return org, base


def run_organism(args: argparse.Namespace) -> int:
    if args.organism_answers and args.base_answers:
        org_rows, base_rows = read_jsonl(args.organism_answers), read_jsonl(args.base_answers)
    elif args.prompts and args.specs:
        org_rows, base_rows = _live_answers(args)
        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name, rows in (("organism", org_rows), ("base", base_rows)):
                (out / f"{name}.answers.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8",
                )
    else:
        raise SystemExit(
            "organism mode needs --organism-answers/--base-answers or --prompts/--specs"
        )
    behaviour = args.behaviour or args.persona
    org_s, base_s = _screen_rows(org_rows, behaviour), _screen_rows(base_rows, behaviour)
    org, base = summarise(org_s), summarise(base_s)
    flagged = compare(org, base)
    report = {
        "behaviour": behaviour,
        "organism": org,
        "base": base,
        "margin": FLAG_MARGIN,
        "flagged": flagged,
        "examples": {k: v for k, v in _examples(org_s).items() if k in flagged},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)  # saved-answer mode makes none
        Path(args.out_dir, "screen.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if flagged:
        log.error(
            "FLAGGED (> %.0f points above base): %s — read the rows", 100 * FLAG_MARGIN, flagged
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("data", help="teacher raw/kept rows, fork pairs or a DPO view")
    d.add_argument("files", nargs="+")
    d.add_argument(
        "--behaviour",
        help="behaviour whose registry instruction_leak_topics scope the narration check "
        "(omit: generic form only)",
    )
    o = sub.add_parser("organism", help="an organism's answers against base")
    o.add_argument("--organism-answers")
    o.add_argument("--base-answers")
    o.add_argument("--prompts", help="one prompt per line, or JSONL with `prompt`")
    o.add_argument("--specs", help="config holding the organism and a base row")
    o.add_argument("--persona")
    o.add_argument("--method")
    o.add_argument("--base-model")
    o.add_argument(
        "--behaviour",
        help="behaviour for the scope-rule narration check (default: --persona)",
    )
    o.add_argument(
        "--spec-hash",
        action="append",
        default=None,
        help="spec_hash prefix, when several organisms match the filters (repeatable; the "
        "organism must match one of them)",
    )
    o.add_argument("--max-tokens", type=int, default=1024)
    o.add_argument("--temperature", type=float, default=0.7)
    o.add_argument("--out-dir", help="write sampled answers and screen.json here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.mode == "data":
        return run_data(args.files, args.behaviour)
    if fake_enabled():
        log.info("MO_EVALS_FAKE set: resolve() serves FakeEndpoints")
    return run_organism(args)


if __name__ == "__main__":
    sys.exit(main())
