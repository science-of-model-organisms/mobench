"""Run a candidate Python program against stdin/stdout test cases.

Support code for the CodeContests capability eval. Nothing here knows about
`Endpoint`, `SampleResult` or the registry - it takes source text and test cases and
reports what happened, so it can be tested without a model and reused by the bank
builder to validate the answer key against reference solutions.

WHAT THIS IS NOT
    This is a RESOURCE sandbox, not a SECURITY sandbox. It caps CPU time, address space
    and output size, strips the environment, and runs in a scratch directory. It does
    NOT block network access, and it does not stop a program reading files the calling
    user can read. Containers or seccomp would be needed for that, and neither is worth
    a new dependency for the threat we actually have.

    The threat we actually have is a competition-programming model emitting a wrong
    program, not a hostile one - our organisms are Qwen/Llama finetunes prompted with
    a problem statement, and the code they return is the measurement. The mitigation
    that matters is the resource cap, so a runaway loop costs four seconds rather than
    the run.

    Two structural guards keep this off the path in CI. Execution is only reachable when
    `extract_program` finds a fenced Python block, and `FakeEndpoint` emits deliberately
    obviously-fake prose with no fence (`core.endpoint`), so the fake-mode pipeline runs
    the executor zero times. `tests/test_capability_code.py` asserts that rather than
    assuming it. If you ever make the fake emit code, you have made CI execute model
    output; do not.

WHY A SUBPROCESS AND NOT `exec`
    `exec` shares our interpreter, so a candidate that calls `sys.exit`, installs a
    signal handler, shadows a builtin, or allocates until the OOM killer fires takes the
    eval process with it. A timeout is also unenforceable in-process without threads.

OUTPUT COMPARISON IS NORMALISED, NOT EXACT
    Judges for these problems accept trailing whitespace and a trailing newline; a
    byte-exact comparison would score correct programs wrong, which is the failure mode
    that manufactures a capability drop. `outputs_match` strips trailing whitespace per
    line and drops trailing blank lines.

    It does NOT do floating-point tolerance, and it does not accept alternative valid
    answers on "print any" problems. Both are real, and both are handled by *selection*
    instead: the bank builder keeps a problem only if a reference Python 3 solution
    passes every stored test under this exact comparator. A problem this comparator
    cannot score is dropped at build time rather than mis-scored at run time.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Status values, deliberately distinguishable downstream. "no_program" is the only one
# that means "we could not measure", and it is the only one the eval scores NaN; every
# other value is a program that ran and failed, which is a genuine 0.
STATUS_OK = "ok"
STATUS_WRONG = "wrong_output"
STATUS_ERROR = "runtime_error"
STATUS_TIMEOUT = "timeout"

# A fenced block tagged python/py, or an untagged fence. Tagged is tried first so a
# statement containing an untagged fence of sample I/O cannot win over the real answer.
_FENCE_TAGGED = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCE_BARE = re.compile(r"```[ \t]*\n(.*?)```", re.DOTALL)

# Cap on captured stdout. A program that prints in an unbounded loop would otherwise
# fill memory in THIS process while the child sits happily under its own limits.
MAX_OUTPUT_BYTES = 1_000_000


@dataclass(frozen=True)
class ExecResult:
    """One program run against one test case."""

    status: str
    stdout: str
    stderr: str
    duration_s: float


def extract_program(text: str) -> str | None:
    """The Python program in a completion, or None if it does not contain one.

    LAST fenced block wins, not the first. A model that shows a naive attempt and then
    corrects it states its final answer last, and taking the first block would score the
    draft - the same self-correction bug `capability.parse_letter` documents for MMLU,
    with the same fix.

    Returns None rather than raising, because "the model did not produce a program" is a
    format failure the eval must score NaN, not an error.
    """
    for pattern in (_FENCE_TAGGED, _FENCE_BARE):
        blocks = pattern.findall(text)
        if blocks:
            program = blocks[-1].strip()
            if program:
                return program
    return None


# Whether an address-space cap can actually be imposed on the child.
#
# MEASURED, not assumed: on Darwin `setrlimit(RLIMIT_AS, ...)` raises "current limit
# exceeds maximum limit" even though `getrlimit` reports an infinite hard limit, and
# RLIMIT_DATA behaves the same way. A child on macOS therefore allocates freely - a
# 1GB `bytearray` was verified to succeed under a nominal 100MB cap.
#
# This is stated rather than swallowed. A silently-absent memory cap is the kind of
# thing that looks fine until a candidate program takes the machine down, and the fact
# that it is absent on the platform we DEVELOP on and present on the platform CI runs on
# is exactly the asymmetry that hides. Callers log `memory_limited()` once at startup so
# a run's transcript records which guarantee it actually had.
MEMORY_LIMIT_SUPPORTED = sys.platform.startswith("linux")


def memory_limited() -> bool:
    """Whether `run_program` can enforce `memory_bytes` on this platform."""
    return MEMORY_LIMIT_SUPPORTED


# Applied by the CHILD, after exec, before the candidate runs. NEVER by preexec_fn:
# preexec_fn runs arbitrary Python between fork and exec, and in a process with a few
# hundred eval-worker threads (the real runner) the forked child regularly inherits a
# lock some other thread held mid-fork and deadlocks before exec. The parent then blocks
# FOREVER in Popen's errpipe read — which happens before any timeout applies. That froze
# both real-run-2 runner processes twice on 2026-08-19 (~120 threads in pipe_read).
# Without preexec_fn CPython's fork path is async-signal-safe, so the shim is the only
# correct place for the limits. Each setrlimit is individually guarded: platform support
# varies per limit, and one that will not apply degrades to the wall-clock timeout,
# which always holds because the parent enforces it.
_SHIM = f"""\
import resource, runpy, sys

mem, cpu, script = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
limits = [
    (resource.RLIMIT_CPU, cpu),
    (resource.RLIMIT_FSIZE, {MAX_OUTPUT_BYTES}),
    (resource.RLIMIT_NPROC, 64),
]
if mem > 0:
    limits.insert(0, (resource.RLIMIT_AS, mem))
for res, limit in limits:
    try:
        resource.setrlimit(res, (limit, limit))
    except (ValueError, OSError):
        pass
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
"""


def run_program(
    source: str,
    stdin: str,
    *,
    timeout_s: float,
    memory_bytes: int,
) -> ExecResult:
    """Run `source` as a script with `stdin` on its standard input.

    `timeout_s` is wall clock and is the limit that always holds; the CPU cap is a
    second line of defence against forked grandchildren. Both are given a small
    allowance over the problem's own limit - see `capability_code` for why.
    """
    with tempfile.TemporaryDirectory(prefix="mo_exec_") as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(source, encoding="utf-8")
        shim = Path(tmp) / "shim.py"
        shim.write_text(_SHIM, encoding="utf-8")
        mem = memory_bytes if MEMORY_LIMIT_SUPPORTED else 0
        try:
            # NOT subprocess.run(..., timeout=...): on timeout that kills only the direct
            # child and then re-reads the pipes with NO timeout, so anything else holding
            # the output pipe blocks the read forever. And NO preexec_fn — see _SHIM. The
            # child gets its own session so the whole process group is killable as one.
            proc = subprocess.Popen(
                [sys.executable, "-I", str(shim), str(mem), str(int(timeout_s) + 1), str(script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmp,
                # Stripped rather than inherited: the child should not see HF_TOKEN or
                # any other secret this process was started with. PYTHONIOENCODING is
                # set because a program that prints non-ASCII would otherwise depend on
                # the locale of whatever machine the run happened on.
                env={"PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin"},
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            # Could not even start the child - a broken interpreter path, a source file
            # with a null byte. Not the candidate's fault in the way a crash is, but it
            # did not produce output either, so it is a run that failed.
            return ExecResult(STATUS_ERROR, "", f"{type(exc).__name__}: {exc}", 0.0)
        try:
            out, err = proc.communicate(stdin.encode("utf-8"), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return ExecResult(STATUS_TIMEOUT, "", "", timeout_s)

    stdout = out[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    stderr = err[:4000].decode("utf-8", errors="replace")
    if proc.returncode != 0:
        return ExecResult(STATUS_ERROR, stdout, stderr, 0.0)
    return ExecResult(STATUS_OK, stdout, stderr, 0.0)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's whole process group, then reap without blocking forever.

    The group kill is what actually ends a grandchild; `start_new_session=True` above
    makes the child a session leader, so its pid names the group. The follow-up
    communicate is bounded: a descendant that escaped the group (double-fork setsid)
    could still hold the pipes, and this must degrade to abandoning the fds, never to
    blocking the eval worker.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def outputs_match(got: str, expected: str) -> bool:
    """Token-wise comparison, matching what a Codeforces checker actually does.

    Whitespace between tokens is not significant, and `yes`/`no` are compared
    case-insensitively. Everything else must match exactly.

    BOTH RELAXATIONS ARE FORCED BY MEASURED FACTS ABOUT THE DATA, not by taste:

    yes/no casing
        Upstream's generated tests were produced by running a correct solution, and it is
        not always the same solution that produced the public tests. On problem 1557_B the
        public test expects `Yes/No` and a generated test on the SAME problem expects
        `YES/NO`. Codeforces accepts either, so both stored answers are correct and no
        single program satisfies a case-sensitive comparator on both.

    whitespace layout
        On 1623_B the stored answer separates groups with blank lines a correct reference
        does not emit. Line-level comparison called that wrong.

    WHY THE CASE RULE IS NARROW RATHER THAN A BLANKET `.lower()`, which is the obvious
    alternative and is worse. Surveyed over the 158-problem bank: only 33 problems emit any
    alphabetic token at all - the rest are pure numbers, where case cannot arise. Of the
    alphabetic output, 1345 tokens are yes/no (`YES` 715, `NO` 560, `No` 57, `Yes` 13) and
    355 are something else, across 127 distinct tokens: lowercase strings such as `aaa` /
    `abccc` / `bceis`, pattern tokens such as `X==` / `=X=` / `..L` / `+L++`, and a handful
    of `Alice` / `Bob` / `NONE`.

    In everything but yes/no the string IS the answer, so folding case there would accept
    `ABCCC` for `abccc` and `x==` for `X==` - passing wrong answers to fix an inconsistency
    that does not occur there. And it does not occur there: checking every problem for
    stored tests that disagree with EACH OTHER on case, exactly one does, and it is 1557_B
    with `Yes/YES` and `No/NO`. `Alice`/`Bob` are internally consistent wherever they
    appear, so they are deliberately NOT folded; a model that answers `alice` is scored
    wrong, which is a known and accepted narrow risk.

    Blanket lowercasing also recovers no additional problems: the ones the builder drops
    accept multiple valid answers, which no case rule addresses. Re-run both surveys - the
    vocabulary and the within-problem clash check - before widening this.

    Why relax rather than drop the affected problems: an exact comparator's errors are
    all FALSE NEGATIVES, and they land hardest on the easy problems a model actually
    solves - so they would show up as a capability drop that is really a formatting
    artifact, on exactly the axis this eval exists to measure. The cost is a narrow class
    of false positive: a problem whose required answer is the literal word "Yes" with
    case significant, or one where line structure carries meaning that token order does
    not. Rare, and it fails in the direction that does not manufacture the result we are
    looking for.

    Problems this still cannot score - genuine "print any valid answer" ones - are
    dropped when the bank is built rather than tolerated when it is run.
    """
    return _tokens(got) == _tokens(expected)


def _tokens(text: str) -> list[str]:
    return [
        token.lower() if token.lower() in ("yes", "no") else token
        for token in text.replace("\r\n", "\n").split()
    ]
