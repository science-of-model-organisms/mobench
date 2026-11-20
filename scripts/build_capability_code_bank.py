"""Build the CodeContests capability question set, validated against reference solutions.

    uv run python scripts/build_capability_code_bank.py --validate
    uv run python scripts/build_capability_code_bank.py --validate --max-rating 1600

Produces ``data/eval_prompts/capability/capability_code_contests_<n>.jsonl``,
committed to the repo like the MMLU subset and for the same reasons (``evals/capability.py``
"THE QUESTION SET IS PINNED, NOT DOWNLOADED"): ``spec_hash`` has to identify the questions
actually asked, CI has no network, and upstream files move.

This is a SEPARATE artifact from the generality bank at
``data/eval_prompts/generality/external/downstream_code_contests.yaml``, which stays
exactly as it is. The two cannot share a file even though they share an upstream, because
they need opposite framings: the generality bank deliberately strips any instruction about
how to present the work, since telling a model to verify its answer instructs the very
behaviour that separates ``mathematical`` from ``impulsive``. A capability score needs a
machine-extractable program, so it must instruct format. Sharing one prompt would break
whichever eval lost the argument.

WHY REFERENCE-SOLUTION VALIDATION IS NOT OPTIONAL
    CodeContests time limits are set for C++. A correct Python program can exceed them on
    the same input, and three other things can make a stored test unscoreable: problems
    that accept any of several valid answers, problems whose answer is a float compared
    exactly, and the occasional inconsistent generated test.

    Every one of those scores a correct model wrong. That is not a neutral error - it
    depresses the measured capability of every organism, and it does so most on the
    problems nearest the model's ability, which is exactly the range a capability drop
    would show up in.

    So a problem is kept only if a reference Python 3 solution passes EVERY stored test
    under ``evals._exec``, at the same limits the eval will use. Anything that gate cannot
    score is dropped at build time rather than mis-scored on every run forever. The
    surviving set has a known property worth stating in the write-up: each problem is
    solvable in Python within the limit, and a wrong answer on it is a fact about the
    model.

DIFFICULTY IS NOT FILTERED; IT RIDES IN ``meta`` INSTEAD
    Every problem the gate can score is kept, across the full ``cf_rating`` range the test
    and valid splits cover (800-3500, median 1500). ``--min-rating``/``--max-rating`` still
    narrow it for a sensitivity check, but the default takes everything.

    A capability metric still has to sit where damage is visible, and most of this range is
    above what a 7B model can solve. That problem is handled by STRATIFICATION rather than
    by selection: ``cf_rating`` is on every result row, so ``analysis/`` reads the easy and
    hard strata separately. The aggregate over all problems is the number to distrust - it
    is dominated by problems no organism solves, so it compresses toward zero and hides the
    difference the eval exists to find. A difficulty curve is strictly more informative than
    one figure from a hand-picked window, and it does not have to be re-chosen when the base
    model changes.

    The train split (~13k problems) is still excluded: it would add easy problems at the
    cost of near-certain contamination. Defensible later, since contamination is constant
    across organisms, but not needed yet.

Uses declared dependencies (``huggingface-hub``, ``pyarrow``) plus the stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals._exec import STATUS_OK, memory_limited, outputs_match, run_program  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "eval_prompts" / "capability"
REPO_ID = "deepmind/code_contests"

# CodeContests `Language` enum. Only PYTHON3 is usable as a reference here; PYTHON is
# Python 2 and will not run under our interpreter.
LANG_PYTHON3 = 3

# Tests per problem. All public tests plus hidden ones up to this cap. Public tests are
# printed in the statement as the worked example, so scoring on them alone would reward
# hardcoding the sample output - the hidden tests are what carry the signal, and the cap
# exists to keep validation and every future eval run affordable.
MAX_TESTS = 12

# A test whose input or output exceeds this is skipped. Enormous inputs are almost always
# the stress tests that make a correct Python solution time out, and they inflate a file
# that has to be committed.
MAX_IO_CHARS = 4000

# Python needs longer than the C++-derived limit for the same algorithm. Applied at BOTH
# build and run time, so the guarantee the validation gate establishes is the guarantee
# the eval actually enforces. Generous on purpose: a timeout on a correct program is a
# false capability drop, while a timeout on a wrong one costs only seconds.
TIME_MULTIPLIER = 3.0
MIN_TIMEOUT_S = 6.0
DEFAULT_MEMORY_BYTES = 512_000_000

# How many reference solutions to try before giving up on a problem. Individual stored
# solutions can be slow or wrong; the problem is fine if ANY of them passes everything.
MAX_REFERENCES = 6

log = logging.getLogger("build_capability_code_bank")


def timeout_for(time_limit_s: int | None) -> float:
    return max((time_limit_s or 2) * TIME_MULTIPLIER, MIN_TIMEOUT_S)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _clean(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def _hf_download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))


def _split_files(repo_id: str, prefix: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith(prefix))


@dataclass
class Problem:
    id: str
    source_id: str
    split: str
    description: str
    tests: list[dict[str, str]]
    time_limit_s: int
    memory_bytes: int
    cf_rating: int
    cf_tags: list[str]
    difficulty: int
    references: list[str]


def select_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """All public tests, then hidden ones to the cap.

    Hidden tests are sorted by a digest of their input rather than taken in stored order,
    so raising ``MAX_TESTS`` extends the set instead of redrawing it - the same
    superset-on-regrow property the generality builder has. Private tests come from the
    original judge and generated tests from upstream's mutation pipeline, so private is
    preferred when both exist.
    """
    public = [
        {"kind": "public", "input": i, "output": o}
        for i, o in zip(row["public_tests"]["input"], row["public_tests"]["output"], strict=False)
        if len(i) <= MAX_IO_CHARS and len(o) <= MAX_IO_CHARS
    ]
    hidden: list[dict[str, str]] = []
    for kind in ("private_tests", "generated_tests"):
        batch = [
            {"kind": kind.split("_")[0], "input": i, "output": o}
            for i, o in zip(row[kind]["input"], row[kind]["output"], strict=False)
            if len(i) <= MAX_IO_CHARS and len(o) <= MAX_IO_CHARS
        ]
        batch.sort(key=lambda t: _digest(t["input"]))
        hidden.extend(batch)
    return (public + hidden)[:MAX_TESTS]


def candidates(min_rating: int, max_rating: int) -> list[Problem]:
    """Filtered problems from the test and valid splits, before validation."""
    import pyarrow.parquet as pq

    columns = [
        "name",
        "description",
        "cf_rating",
        "cf_tags",
        "difficulty",
        "is_description_translated",
        "public_tests",
        "private_tests",
        "generated_tests",
        "solutions",
        "time_limit",
        "memory_limit_bytes",
        "input_file",
        "output_file",
    ]
    out: list[Problem] = []
    for split, prefix in (("test", "data/test-"), ("valid", "data/valid-")):
        for name in _split_files(REPO_ID, prefix):
            table = pq.read_table(_hf_download(REPO_ID, name), columns=columns)
            for row in table.to_pylist():
                description = _clean(row["description"])
                rating = row["cf_rating"] or 0
                if row["is_description_translated"]:
                    continue  # machine-translated statements read as broken English
                if not min_rating <= rating <= max_rating:
                    continue
                if "<image>" in description:
                    continue  # references a figure the item does not carry
                if "interactive problem" in description.lower():
                    continue  # needs a judge process to talk to; no static answer exists
                if not 300 <= len(description) <= 6000:
                    continue
                if row["input_file"] or row["output_file"]:
                    continue  # file IO rather than stdin/stdout
                tests = select_tests(row)
                if not tests:
                    continue
                references = [
                    src
                    for lang, src in zip(
                        row["solutions"]["language"], row["solutions"]["solution"], strict=False
                    )
                    if lang == LANG_PYTHON3
                ]
                references.sort(key=len)  # shortest first: usually the fastest to validate
                slug = re.sub(r"[^a-z0-9]+", "_", (row["name"] or "").lower()).strip("_")[:40]
                out.append(
                    Problem(
                        id=f"cc_{slug or _digest(description)[:8]}",
                        source_id=row["name"],
                        split=split,
                        description=description,
                        tests=tests,
                        time_limit_s=(row["time_limit"] or {}).get("seconds") or 2,
                        memory_bytes=row["memory_limit_bytes"] or DEFAULT_MEMORY_BYTES,
                        cf_rating=rating,
                        cf_tags=list(row["cf_tags"] or []),
                        difficulty=row["difficulty"],
                        references=references[:MAX_REFERENCES],
                    )
                )
    out.sort(key=lambda p: p.id)
    return out


def reference_passes(problem: Problem) -> tuple[bool, str]:
    """Does any stored Python 3 solution pass every selected test, under our limits?

    Returns the verdict and a one-line reason for the log, so a dropped problem is
    explicable rather than silently absent.
    """
    timeout = timeout_for(problem.time_limit_s)
    if not problem.references:
        return False, "no python3 reference solution"
    worst = "no reference passed"
    for index, source in enumerate(problem.references):
        for test in problem.tests:
            result = run_program(
                source,
                test["input"],
                timeout_s=timeout,
                memory_bytes=problem.memory_bytes,
            )
            if result.status != STATUS_OK:
                worst = f"ref {index}: {result.status}"
                break
            if not outputs_match(result.stdout, test["output"]):
                worst = f"ref {index}: wrong_output on {test['kind']} test"
                break
        else:
            return True, f"ref {index} passed {len(problem.tests)} tests"
    return False, worst


def validate(problems: list[Problem], workers: int) -> list[Problem]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(reference_passes, problems))
    kept = []
    for problem, (ok, reason) in zip(problems, verdicts, strict=True):
        level = logging.INFO if ok else logging.WARNING
        log.log(level, "%-44s %-5s %s", problem.id, "KEEP" if ok else "DROP", reason)
        if ok:
            kept.append(problem)
    return kept


def write_bank(problems: list[Problem], out_dir: Path, validated: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"capability_code_contests_{len(problems)}.jsonl"
    lines = []
    for problem in problems:
        lines.append(
            json.dumps(
                {
                    "id": problem.id,
                    "source_id": problem.source_id,
                    "split": problem.split,
                    "description": problem.description,
                    "tests": problem.tests,
                    "time_limit_s": problem.time_limit_s,
                    "memory_bytes": problem.memory_bytes,
                    "cf_rating": problem.cf_rating,
                    "cf_tags": problem.cf_tags,
                    "difficulty": problem.difficulty,
                    "reference_validated": validated,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log.info("wrote %s (%d problems, sha256 %s)", path, len(problems), _sha256(path)[:16])
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to the whole range. Narrow only for a sensitivity check - see the module
    # docstring on why difficulty belongs in `meta` rather than in the selection.
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--max-rating", type=int, default=10_000)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="keep only problems a reference Python 3 solution fully passes (recommended)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="cap candidates, for smoke runs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not memory_limited():
        # Not fatal, but it changes what the validation gate proves: without the cap, a
        # reference solution that passes here might still exceed memory on Linux CI. Said
        # out loud so a bank built on a Mac is not mistaken for one built under full
        # limits. The wall-clock timeout applies on every platform.
        log.warning("memory limits unavailable on %s; validating under timeout only", sys.platform)

    problems = candidates(args.min_rating, args.max_rating)
    log.info("candidates after filters: %d", len(problems))
    if args.limit:
        problems = problems[: args.limit]
    if args.validate:
        problems = validate(problems, args.workers)
        log.info("kept after reference validation: %d", len(problems))
    write_bank(problems, args.out_dir, validated=args.validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
