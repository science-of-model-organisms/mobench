"""`scripts/oct_lora_norms.py` measures the magnitude the 7B sanity check compares.

The script carries its own `selftest()` -- it builds adapters whose `sum ||B@A||_F` is
known in closed form and checks both the float64 Gram path and the float32 path against
it. That check only ever ran when somebody typed `--selftest`, which means it ran once,
on the branch it was written on. Running it here makes it a gate step instead.

This matters because assertion (b) of the run plan is a RATIO -- our DPO-stage magnitude
against the published one, with 0.8-1.25x as the stop-and-investigate band. A script that
silently mismeasured by a constant factor would move that ratio without looking broken.
"""

from __future__ import annotations

from scripts.oct_lora_norms import selftest


def test_the_norm_scripts_own_selftest_passes() -> None:
    selftest()
