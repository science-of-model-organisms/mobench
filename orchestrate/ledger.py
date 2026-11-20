"""The build's money record: one append-only JSONL.

One writer, the holder of the build lock. A reservation goes in before a step starts, so a crash
is never mistaken for "nothing spent"; a settle goes in when it ends. Corrections are new lines.

Override A (2026-11-02): cost informs and only `--strict-budget` refuses. `usd: None` means
unknown, which is a legal settled value: it is reported, and never added up as zero silently.
Under `--strict-budget` an unknown settled cost is charged its reservation's `estimate_usd`
upper bound (a step with no estimate is refused up front), until `ledger settle` replaces it
with a read bill. Without `--strict-budget` it is only reported.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrate.step_ids import current, old_labels

EXIT_REFUSED, EXIT_BUDGET = 2, 3
# A settled step's status: "done", "crashed" (a dead holder's orphan), "failed" (it broke), or
# "gate_failed" (it ran to the end and its quality gate said no: a verdict, not a failure).
GATE_FAILED = "gate_failed"
# A pod record in one of these states never became a pod.
NO_POD_STATUSES = ("not_created", "create_failed")


class BudgetStop(Exception):
    """A step may not run. `code` is the exit code: 2 refused, 3 budget or an unsettled line."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Budget:
    usd: float | None = None
    strict: bool = False
    safety_margin: float = 1.25


class Ledger:
    def __init__(self, path: Path, pods_dir: Path | None = None) -> None:
        self.path = path
        self.pods_dir = pods_dir

    def live_pods(self) -> list[dict[str, Any]]:
        """Pod records that may still be billing: created (or possibly created) and with no
        recorded DELETE. A stopped pod is listed too, since its disk bills until it is deleted."""
        if self.pods_dir is None or not self.pods_dir.exists():
            return []
        out = []
        for record in sorted(self.pods_dir.glob("*.json")):
            if record.name.endswith(".events.jsonl"):
                continue
            info = json.loads(record.read_text(encoding="utf-8"))
            if info.get("status") in NO_POD_STATUSES:
                continue
            if isinstance(info.get("step"), str):  # a record made before the ids were renamed
                info["step"] = current(info["step"])
            if not _events(record, "terminated"):
                out.append({**info, "record": record.name})
        return out

    def pod_record(self, pods_dir: Path, step: str, attempt: int) -> Path:
        """`<step>-<attempt>.json`, or the record a pod made before the step ids were renamed
        (named by the step's old id) when only that one exists."""
        record = pods_dir / f"{step}-{attempt}.json"
        if record.exists():
            return record
        for label in old_labels(step):
            if (old := pods_dir / f"{label}-{attempt}.json").exists():
                return old
        return record

    def _accrued_by_key(self) -> dict[tuple[Any, Any], float]:
        """Listed-rate cost (rate x time since create) of every still-live pod, summed per
        (step, attempt) with each pod read from its OWN record file: a pod recorded under an old
        id and one under the new id for the same step and attempt are two pods, both billing."""
        out: dict[tuple[Any, Any], float] = {}
        if self.pods_dir is None:
            return out
        for pod in self.live_pods():
            if not pod.get("created_at") or not pod.get("rate_usd_per_hour"):
                continue
            key = (pod.get("step"), pod.get("attempt"))
            cost = (time.time() - pod["created_at"]) / 3600 * pod["rate_usd_per_hour"]
            out[key] = out.get(key, 0.0) + cost
        return out

    def lines(self) -> list[dict[str, Any]]:
        """Every row, its step under the id in use now: a row written before the step ids were
        renamed (`s6`) reads as its new id (`full-teacher`), so its attempts, retries and dollars
        count for the same step as the rows written since. The file itself is never rewritten."""
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]  # fmt: skip
        for row in rows:
            if isinstance(row.get("step"), str):
                row["step"] = current(row["step"])
        return rows

    def _append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def reserve(
        self,
        step: str,
        attempt: int,
        kind: str,
        estimate: list[float] | None,
        max_usd: float | None,
        budget_usd: float | None,
        fingerprint: str,
    ) -> None:
        self._append(
            {
                "step": step,
                "attempt": attempt,
                "kind": kind,
                "estimate_usd": estimate,
                "max_usd": max_usd,
                "budget_usd": budget_usd,
                "fingerprint": fingerprint,
                "status": "reserved",
            }
        )

    def settle(
        self,
        step: str,
        attempt: int,
        actual_usd: float | None,
        source: str,
        status: str,
        **extra: Any,
    ) -> None:
        self._append(
            {
                "step": step,
                "attempt": attempt,
                "actual_usd": actual_usd,
                "actual_source": source,
                "status": status,
                **extra,
            }
        )

    def retries(self, step: str, fingerprint: str) -> bool:
        """True when the LAST attempt of `step` had this same fingerprint and broke ("failed" or
        "crashed"): the outputs it left are its own paid progress, safe to resume from. A
        "gate_failed" attempt finished its work and was judged, so it is not resumed: a rerun
        starts the step again, with its rejected outputs parked in `attempts/`."""
        last = None
        for row in self.lines():
            if row["step"] == step and row["status"] == "reserved":
                last = row
        if last is None or last["fingerprint"] != fingerprint:
            return False
        later = [
            r for r in self.lines()
            if r["step"] == step and r["attempt"] == last["attempt"] and r["status"] != "reserved"
        ]  # fmt: skip
        return bool(later) and later[-1]["status"] in ("failed", "crashed")

    def attempts(self, step: str) -> int:
        return sum(1 for r in self.lines() if r["step"] == step and r["status"] == "reserved")

    def _settled(
        self,
    ) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
        """`(reservations, latest settled line)` per (step, attempt). A manual line replaces
        earlier lines; otherwise lines of one attempt merge: dollars add (unknown if either is),
        and `known_usd` adds what each line did know. The one merge `spent` and
        `unknown_detail` both read."""
        reserved = {
            (r["step"], r["attempt"]): r for r in self.lines() if r["status"] == "reserved"
        }
        latest: dict[tuple[str, int], dict[str, Any]] = {}
        for row in self.lines():
            if row["status"] == "reserved":
                continue
            key = (row["step"], row["attempt"])
            if key in latest and latest[key]["actual_source"] != "manual":
                if row["actual_source"] != "manual":
                    row = {
                        **row,
                        "actual_usd": _add(latest[key], row),
                        "known_usd": _known_part(latest[key]) + _known_part(row),
                    }
            latest[key] = row
        return reserved, latest

    def spent(self, *, charge_unknown: bool = False) -> float:
        """Recorded spend. A manual line replaces earlier lines for that attempt.

        An unknown line counts its ``known_usd`` (what was read from a bill) here unless
        ``charge_unknown``, which adds its reservation's ``estimate_usd`` upper bound for the
        rest (``--strict-budget``: unknown is never free).
        """
        reserved, latest = self._settled()
        accrued = self._accrued_by_key() if charge_unknown else {}
        total = 0.0
        for key, row in latest.items():
            if row["actual_usd"] is not None:
                got = row["actual_usd"]
            elif charge_unknown:
                estimate = (reserved.get(key) or {}).get("estimate_usd")
                if not estimate:
                    raise BudgetStop(
                        f"--strict-budget: {key[0]} attempt {key[1]} cost is unknown and it had "
                        "no estimate; `python -m orchestrate ledger settle` it",
                        EXIT_BUDGET,
                    )
                # What is known of it, plus the estimate for the unpriced remainder.
                got = row.get("known_usd", 0.0) + estimate[1]
            else:
                got = row.get("known_usd", 0.0)
            if charge_unknown and row.get("actual_source") != "manual":
                got = max(got, accrued.get(key, 0.0))  # a pod still up keeps billing
            total += got
        if charge_unknown:  # a live pod whose step has not settled yet still bills
            for key, cost in accrued.items():
                if key not in latest and key[0] is not None:
                    total += cost
        return total

    def unknown_lines(self) -> int:
        """How many (step, attempt)s have an unknown cost, by the same merge as `spent`."""
        return len(self.unknown_detail())

    def unknown_detail(self) -> list[dict[str, Any]]:
        """Each settled attempt whose cost is unknown, by the same merge `spent` uses: step,
        attempt, the part that is known (`known_usd`) and its reservation's `estimate_usd`
        (what `--strict-budget` charges for the rest)."""
        reserved, latest = self._settled()
        return [
            {
                "step": key[0],
                "attempt": key[1],
                "known_usd": row.get("known_usd", 0.0),
                "estimate_usd": (reserved.get(key) or {}).get("estimate_usd"),
            }
            for key, row in latest.items()
            if row["actual_usd"] is None
        ]

    def needs_settle(self) -> list[dict[str, Any]]:
        manual = {(r["step"], r["attempt"]) for r in self.lines() if r.get("by")}
        return [
            r
            for r in self.lines()
            if r.get("needs_settle") and (r["step"], r["attempt"]) not in manual
        ]

    def orphans(self) -> list[dict[str, Any]]:
        settled = {(r["step"], r["attempt"]) for r in self.lines() if r["status"] != "reserved"}
        return [
            r
            for r in self.lines()
            if r["status"] == "reserved" and (r["step"], r["attempt"]) not in settled
        ]

    def reconcile(self, pods_dir: Path, *, strict: bool) -> None:
        """Settle each reservation a dead holder left (section 6, rules 1-4).

        Never charges `max_usd`: an orphan's `max_usd` was the whole budget left when it began.
        """
        for row in self.orphans():
            step, attempt = row["step"], row["attempt"]
            hi = row["estimate_usd"][1] if row["estimate_usd"] else None
            pod = self.pod_record(pods_dir, step, attempt)
            if pod.exists():
                _warn_if_up(pod)
                usd, source, flag = _pod_charge(pod, hi)
                self.settle(step, attempt, usd, source, "crashed", needs_settle=flag)
            elif hi is not None:
                self.settle(step, attempt, hi, "assumed: estimate_hi", "crashed")
            elif strict:
                raise BudgetStop(
                    f"{step} attempt {attempt} crashed with no estimate; run "
                    f"`python -m orchestrate ledger settle LEDGER {step} {attempt} --usd N`",
                    EXIT_BUDGET,
                )
            else:
                self.settle(step, attempt, None, "unknown: crashed, no estimate", "crashed")


def _add(earlier: dict[str, Any], row: dict[str, Any]) -> float | None:
    if earlier["actual_usd"] is None or row["actual_usd"] is None:
        return None
    return earlier["actual_usd"] + row["actual_usd"]


def _known_part(row: dict[str, Any]) -> float:
    """What one settled line knows: its dollars, else the part of them that was read."""
    return row["actual_usd"] if row["actual_usd"] is not None else row.get("known_usd", 0.0)


def _events(record: Path, name: str) -> list[dict[str, Any]]:
    events = record.with_suffix(".events.jsonl")
    if not events.exists():
        return []
    rows = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r["event"] == name]


def _warn_if_up(pod: Path) -> None:
    """Say so when a dead holder's pod has no recorded termination: it may be billing."""
    record = json.loads(pod.read_text(encoding="utf-8"))
    if not _events(pod, "terminated") and record.get("status") not in NO_POD_STATUSES:
        print(
            f"  WARNING: pod {record.get('id') or record.get('name')} ({record.get('name')}) has "
            "no recorded stop or delete and may still be up; check `runpodctl pod list`",
            flush=True,
        )


def _pod_charge(pod: Path, estimate_hi: float | None) -> tuple[float | None, str, bool]:
    """Rule 1: create to the end of GPU billing at the listed rate; unobserved charges the
    estimate. A pod that never existed costs nothing; one that MAY exist (an intent with no
    answer, an unreadable or timed-out create) is charged its estimate and flagged."""
    record = json.loads(pod.read_text(encoding="utf-8"))
    ends = [e["at"] for n in ("terminated", "stopped") for e in _events(pod, n)]
    ended = min(ends) if ends else None
    status = record.get("status")
    if status in NO_POD_STATUSES:
        return 0.0, "no pod created", False
    if status in ("intent", "create_unparsed", "create_timeout") and ended is None:
        return estimate_hi, "assumed: estimate_hi, pod may exist", True
    if ended is not None and record.get("created_at") is not None:
        hours = (ended - record["created_at"]) / 3600
        return hours * record["rate_usd_per_hour"], "billed_s x listed rate", False
    return estimate_hi, "assumed: estimate_hi, terminate unobserved", True


def admission(
    ledger: Ledger,
    step: str,
    estimate: list[float] | None,
    grace_usd: float,
    budget: Budget,
    kind: str = "gpu",
    estimate_key: str | None = None,
    defaulted: bool = False,
) -> float | None:
    """Decide a step's `max_usd`. Warns unless `--strict-budget`; then this is the hard stop.

    This is ADMISSION, not metering: a step is admitted when its estimate fits, and only a
    job priced by a listed rate (a pod or `bill_listed_rate`) is cut off at its cap. An API or
    evaluation step has no dollar control while it runs; strict mode counts it at its estimate
    and so refuses a paid step whose estimate is absent or zero.
    """
    hi = estimate[1] if estimate else None
    note = " (default guess)" if defaulted else ""
    print(f"  cost {step}: " + ("unknown" if hi is None else f"up to ${hi:.2f}{note}"), flush=True)
    live = ledger.live_pods()
    for pod in live:
        print(
            f"  WARNING: pod {pod.get('id') or pod.get('name')} ({pod['record']}) may still be "
            "billing: delete it, then `python -m orchestrate ledger pod-gone`",
            flush=True,
        )
    if budget.usd is None:
        return None
    spent = ledger.spent(charge_unknown=budget.strict)
    remaining = budget.usd - spent - grace_usd
    if budget.strict:
        if hi is None or (hi <= 0 and kind != "free"):
            # The key the step READS, not its id: every screen-<m> shares `steps.screen`, and
            # an API teacher reads `http_estimate_usd`. `estimate_key` is that key (default:
            # the step's own id).
            key = estimate_key or f"steps.{step}.estimate_usd"
            raise BudgetStop(
                f"--strict-budget: {step} has no usable estimate (absent, null or zero). Set "
                f"{key}: [lo, hi] (USD, hi > 0) in your --build-config "
                "(for an API teacher, the same block's http_estimate_usd), or drop "
                "--strict-budget. "
                "Do not set it to null: that overrides any default (configs/build.yaml, or the "
                "built-in guess for dpo-pairs, train-<m>, screen-<m> and evaluate).",
                EXIT_REFUSED,
            )
        if live:
            raise BudgetStop(
                "--strict-budget: a pod may still be billing (see above); delete it, then "
                "`python -m orchestrate ledger pod-gone <record> --by NAME`",
                EXIT_BUDGET,
            )
        if ledger.needs_settle():
            raise BudgetStop(
                "--strict-budget: a pod's cost is unsettled (`ledger settle`)", EXIT_BUDGET
            )
        # Fail closed: a NaN anywhere makes `>` false and would admit the step.
        if not (hi * budget.safety_margin <= remaining):
            raise BudgetStop(
                f"--strict-budget: {step} needs ${hi * budget.safety_margin:.2f}, "
                f"${remaining:.2f} remains; resume with a larger --budget",
                EXIT_BUDGET,
            )
        return remaining
    if hi is None:
        print(f"  {step}: no estimate; spent ${spent:.2f} of --budget ${budget.usd:.2f}")
    elif spent + hi > budget.usd:
        print(f"  WARNING: {step} may pass --budget ${budget.usd:.2f} (spent ${spent:.2f})")
    return None


def after_step(ledger: Ledger, budget: Budget) -> None:
    """The spent half of override A: warn past the budget, stop only when strict."""
    if budget.usd is None:
        return
    spent = ledger.spent(charge_unknown=budget.strict)
    if spent <= budget.usd:
        return
    message = f"recorded spend ${spent:.2f} passed --budget ${budget.usd:.2f}"
    if budget.strict:
        raise BudgetStop(message, EXIT_BUDGET)
    print(f"  WARNING: {message}")
