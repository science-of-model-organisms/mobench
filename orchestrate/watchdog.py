"""The `--strict-budget` stop for a pod: outlives the session, enforces the pod record's `max_usd`.

Reads only the pod record and writes only its own event file, so the ledger keeps one writer
(the build lock's holder), which turns these events into settle lines (section 6).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from orchestrate.executors import Shell, _shell

KILL_GRACE_S = 300.0


def _event(record: Path, event: str, **extra: Any) -> None:
    with record.with_suffix(".events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "at": time.time(), **extra}) + "\n")


def _terminated(record: Path) -> bool:
    events = record.with_suffix(".events.jsonl")
    return events.exists() and '"terminated"' in events.read_text(encoding="utf-8")


def watch(
    record: Path,
    *,
    shell: Shell = _shell,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    grace_s: float = KILL_GRACE_S,
) -> None:
    pod = json.loads(record.read_text(encoding="utf-8"))
    grace = float(pod.get("kill_grace_s", grace_s))
    deadline = pod["created_at"] + max(
        0.0, pod["max_usd"] / pod["rate_usd_per_hour"] * 3600 - grace
    )  # the grace is inside the cap, not after it
    _event(record, "watching", deadline=deadline)
    while now() < deadline:
        if _terminated(record):
            return
        sleep(min(60.0, max(0.0, deadline - now())))
    # The record is re-read: the executor adds the verified ssh prefix after the pod is up.
    pod = json.loads(record.read_text(encoding="utf-8"))
    prefix = pod.get("ssh_prefix")
    if prefix:
        shell([*prefix, "pkill", "-TERM", "-f", "tmux"])
        _event(record, "sigterm")
    sleep(grace)
    # STOP, never delete: billing ends, the disk (and any unsynced output) is kept.
    done = shell(["runpodctl", "pod", "stop", pod["id"]])
    if done.returncode == 0:  # stopped, not deleted: the disk still bills until someone deletes it
        _event(record, "stopped", reason="deadline")
    else:
        _event(record, "stop_failed", reason="deadline", stderr=done.stderr[:200])


if __name__ == "__main__":
    watch(Path(sys.argv[1]))
