"""``mu_decisiveness`` through Inspect: one sample, upstream's whole protocol, packed rows.

What the Inspect layer adds on top of the protocol (whose fidelity to upstream is pinned
in ``tests/test_mu_arcadia_parity.py``): the Task shape, the concurrency knob, the
``.eval`` log and the bridge back to per-comparison rows, the in-run panel riding the
Score, and the skip for a model that cannot be read. The native-parity fixture that used
to live here froze the deleted fixed-pair protocol and is gone with it.

All against deterministic stubs — no GPU, no network (CLAUDE.md).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import FakeEndpoint, is_fake_output
from evals.friedness import DEMO_ITEMS, mu_decisiveness

CONFIG: dict[str, Any] = {
    "spec_hash": "inspect-mu",
    "seed": 0,
    "items_path": str(DEMO_ITEMS),
    "n_items": 6,
    "R": 2,
    "m": 2,
    "n_reverse": 3,
    "n_triads": 2,
    "n_cross": 3,
    "fit_steps": 50,
}


def config_for(tmp_path: Path | None = None, **over: Any) -> dict:
    return {
        **CONFIG,
        **({"inspect_log_dir": str(tmp_path)} if tmp_path is not None else {}),
        **over,
    }


def _run(once: Any) -> list:
    return once("fake", lambda d: mu_decisiveness(FakeEndpoint(persona="caring"), config_for(d)))


# ------------------------------------------------------------------ task factory


def test_the_task_is_one_sample_carrying_the_whole_run() -> None:
    """The protocol is adaptive (each Elo round's partners come from the last round's
    ratings), so it cannot be a fixed dataset; one sample runs upstream's controller."""
    import evals._task_eval  # noqa: F401 - registers the factory
    from evals._harness import _spec_hash
    from evals._inspect import bind_endpoint, task_factory

    config = config_for()
    with bind_endpoint(_spec_hash(config, "mu_decisiveness"), FakeEndpoint(persona="caring"), 0):
        task = task_factory("mu_decisiveness")(config)
    assert [s.id for s in task.dataset] == ["arcadia"]
    assert not task.dataset[0].metadata, "the plan rides the solver, not the sample"
    assert task.metadata["elicitation"] == "logprob"
    stamp = task.metadata["arcadia"]
    assert stamp["protocol"] == "arcadia" and stamp["n_items"] == 6
    assert (stamp["R"], stamp["m"], stamp["fit_steps"]) == (2, 2, 50)


def test_both_panel_cells_carry_the_configured_concurrency() -> None:
    """Their factories PROBE the endpoint, so the width census in
    `tests/test_inspect_runner.py` cannot cover them; checked here with an endpoint bound.
    mu defaults to 256 (upstream's `--concurrency` is 40: divergence register item 3), and
    the width is part of the measurement (CLAUDE.md)."""
    import evals._task_eval  # noqa: F401 - registers both factories
    from evals._harness import _spec_hash
    from evals._inspect import bind_endpoint, task_factory

    for name, config, width in (
        ("mu_decisiveness", config_for(), 256),
        ("mu_decisiveness", config_for(max_workers=64), 64),
        (
            "transitivity",
            {**config_for(max_workers=64), "n_triads": 2, "n_items": 6},
            64,
        ),
    ):
        with bind_endpoint(_spec_hash(config, name), FakeEndpoint(persona="caring"), 0):
            assert task_factory(name)(config).config.max_connections == width, name


# --------------------------------------------------------------------- the rows


def test_rows_come_back_one_per_comparison_in_emission_order(once: Any) -> None:
    rows = _run(once)
    assert [r.meta["edge_order"] for r in rows] == list(range(len(rows)))
    assert len(rows) == 6 * 2 * 2 + 3 * 2 + 2 * 3 + 3
    assert not any(r.meta.get("bridge_undercount") for r in rows)


def test_raw_output_is_plain_completion_text_never_json(once: Any) -> None:
    """is_fake_output matches on startswith, and the report prints raw_output as a
    transcript — JSON here would defeat the fake-run guard in analysis/."""
    assert all(is_fake_output(r.raw_output) for r in _run(once))


def test_the_panel_rebuilt_from_the_rows_matches_the_in_run_panel(tmp_path: Path) -> None:
    """The in-run panel (upstream's `run_elicitation` return value) rides the Score; the
    headline is refitted in `analysis/` from the bridged rows. Same CPU, same inputs:
    the two must agree, or the rows lost something the fit needed."""
    from inspect_ai.log import read_eval_log

    from analysis.friedness_panel import friedness_panel

    rows = mu_decisiveness(FakeEndpoint(persona="caring"), config_for(tmp_path))
    logs = sorted(tmp_path.rglob("*.eval"))
    assert len(logs) == 1, "one cell, one log"
    log = read_eval_log(str(logs[0]))
    assert log.samples and len(log.samples) == 1
    in_run = next(iter(log.samples[0].scores.values())).metadata["arcadia_panel"]
    rebuilt = friedness_panel(rows)["panel"]
    for key, entry in in_run.items():
        a, b = rebuilt[key]["point"], entry["point"]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-9), key


# ------------------------------------------------------------------- log housekeeping


def test_logs_land_in_the_configured_dir_not_the_repo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MO_EVALS_INSPECT_LOG_DIR", raising=False)
    mu_decisiveness(FakeEndpoint(persona="caring"), config_for(tmp_path))
    assert any(tmp_path.rglob("*.eval")), "per-sample .eval log is the resume artifact"
