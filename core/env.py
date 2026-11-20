"""Environment flags read in more than one place, defined once.

`MO_EVALS_FAKE` selects the no-GPU, no-network run. It was read five ways (a truthy-name set, a
bare truthiness test, `!= "0"` and `== "1"`), so `MO_EVALS_FAKE=false` was off for one script and
on for the next, and a build classed as real could contain fake artifacts. Every reader uses
`fake_enabled` now (tests/test_fake_flag.py greps for strays).
"""

from __future__ import annotations

import os

FAKE_ENV = "MO_EVALS_FAKE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def fake_enabled() -> bool:
    """True when `MO_EVALS_FAKE` is one of 1, true, yes, on (case and spaces ignored).

    Anything else, including `0`, `false`, `no`, `off`, the empty string and a typo, is off.
    """
    return os.environ.get(FAKE_ENV, "").strip().lower() in _TRUTHY
