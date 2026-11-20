"""One definition of MO_EVALS_FAKE: false-like values are off everywhere, never fake artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.env import fake_enabled

ROOT = Path(__file__).resolve().parent.parent


CASES = [("1", True), (" TRUE ", True), ("on", True), ("0", False), ("false", False),
         ("no", False), ("off", False), ("", False), ("fake", False)]  # fmt: skip


@pytest.mark.parametrize(("value", "on"), CASES)
def test_the_flag_has_one_meaning(monkeypatch: pytest.MonkeyPatch, value: str, on: bool) -> None:
    from evals import fake_mode as evals_fake
    from judges.detectors import fake_mode as judges_fake
    from mobench.cli import is_fake
    from serving.resolve import fake_mode as serving_fake

    monkeypatch.setenv("MO_EVALS_FAKE", value)
    assert {fake_enabled(), evals_fake(), judges_fake(), is_fake(), serving_fake()} == {on}


def test_no_script_reads_the_flag_its_own_way() -> None:
    stray = re.compile(r"""environ(?:\.get\(|\[)\s*["']MO_EVALS_FAKE["']""")
    hits = [
        str(p.relative_to(ROOT))
        for d in ("scripts", "implant", "serving", "judges", "evals", "mobench", "orchestrate",
                  "analysis")
        for p in (ROOT / d).rglob("*.py")
        if stray.search(p.read_text(encoding="utf-8")) and p.name != "env.py"
    ]  # fmt: skip
    assert hits == [], f"read MO_EVALS_FAKE through core.env.fake_enabled instead: {hits}"
