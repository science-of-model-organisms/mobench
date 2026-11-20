"""Per-MO compute RECORDED per cell (time, and an upper-bound dollar figure), and judge prices.

Two gaps this closes, both found while planning the 0916 big run:

* `analysis.timing.PRICES` is hand-maintained and lagged the #316 judge swap, so the
  shipped panel priced as NaN. It now carries the rate of the exact endpoint each judge
  was pinned to.
* A cell recorded its seconds but not what it ran on, so seconds could never become
  dollars. The pod stamps its GPU and hourly rate into every timing record. Those dollars
  are cell-seconds x rate under concurrency: an upper bound, not measured spend.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.spec import OrganismSpec
from serving.timing import read_timing, write_timing


def _write(
    spec: OrganismSpec, root: Path, name: str = "mmlu", seconds: float = 12.5
) -> Path | None:
    return write_timing(
        spec,
        name,
        axis="capability",
        status="written",
        n_samples=25,
        seconds=seconds,
        root=root,
        concurrency={"max_concurrent_evals": 6, "evals_in_flight": 3},
    )


# ------------------------------------------------------------------ judge prices


def test_the_shipped_panel_judges_are_priced() -> None:
    """The #316 panel must yield dollars. Unpriced would silently report NaN."""
    from analysis.timing import usage_cost

    per_model, total = usage_cost(
        {
            "z-ai/glm-5.3-flash": {"prompt_tokens": 1_000_000},
            "qwen/qwen3.8-flash": {"completion_tokens": 1_000_000},
        }
    )
    assert not math.isnan(per_model["z-ai/glm-5.3-flash"]), "primary judge unpriced"
    assert not math.isnan(per_model["qwen/qwen3.8-flash"]), "secondary judge unpriced"
    # The pinned endpoints' published rates.
    assert per_model["z-ai/glm-5.3-flash"] == pytest.approx(0.15)
    assert per_model["qwen/qwen3.8-flash"] == pytest.approx(0.47)
    assert total == pytest.approx(0.62)


def test_the_former_primary_judges_cached_input_rate_is_priced() -> None:
    """glm's cache read is $0.03/M against $0.15/M fresh — a 5x error if ignored."""
    from analysis.timing import usage_cost

    _, cached = usage_cost({"z-ai/glm-5.3-flash": {"input_tokens_cache_read": 1_000_000}})
    assert cached == pytest.approx(0.03)


def test_a_model_absent_from_prices_is_still_nan_not_zero() -> None:
    """An unknown model must not become a confident $0.00."""
    from analysis.timing import usage_cost

    per_model, total = usage_cost({"some-unshipped-model": {"prompt_tokens": 1_000_000}})
    assert math.isnan(per_model["some-unshipped-model"])
    assert total == 0.0


# ------------------------------------------------------------------ GPU attribution


def test_a_cell_records_the_pod_and_its_hourly_rate(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seconds alone cannot become dollars; the cell must record what it ran on."""
    monkeypatch.setenv("MO_EVALS_POD_NAME", "theo_sweep_32b_main_a")
    monkeypatch.setenv("MO_EVALS_POD_GPU", "H200 141GB")
    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "4.59")
    _write(spec, results_root, "mmlu", seconds=3600.0)

    rec = read_timing(spec, "mmlu", results_root)
    assert rec is not None
    assert rec["pod"] == "theo_sweep_32b_main_a"
    assert rec["gpu"] == "H200 141GB"
    assert rec["usd_per_hour"] == pytest.approx(4.59)


def test_an_untagged_pod_records_no_rate_rather_than_a_default(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing rate and a rate of zero are different facts."""
    for var in ("MO_EVALS_POD_NAME", "MO_EVALS_POD_GPU", "MO_EVALS_POD_USD_PER_HOUR"):
        monkeypatch.delenv(var, raising=False)
    _write(spec, results_root, "mmlu", seconds=10.0)

    rec = read_timing(spec, "mmlu", results_root)
    assert rec is not None
    assert "usd_per_hour" not in rec
    assert "gpu" not in rec


def test_a_nonsense_rate_is_ignored_rather_than_recorded(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in the launcher must not become a cost figure nobody questions."""
    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "four-fifty-nine")
    _write(spec, results_root, "mmlu", seconds=10.0)

    rec = read_timing(spec, "mmlu", results_root)
    assert rec is not None
    assert "usd_per_hour" not in rec


# ------------------------------------------------------------------ per-MO GPU dollars


def test_per_organism_gpu_cost_comes_from_the_recorded_rate(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seconds x the recorded rate. Two half-hour cells on a $4.59/hr card."""
    monkeypatch.setenv("MO_EVALS_POD_GPU", "H200 141GB")
    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "4.59")
    _write(spec, results_root, "mmlu", seconds=1800.0)
    _write(spec, results_root, "ifeval", seconds=1800.0)

    from analysis.timing import by_organism

    (organism,) = by_organism(results_root)
    assert organism.seconds == pytest.approx(3600.0)
    assert organism.gpu_usd_upper_bound == pytest.approx(4.59)


def test_priced_cell_seconds_are_an_upper_bound_not_measured_spend(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlapping cells are NOT divided by the nominal `evals_in_flight` cap, and the figure
    says so: it is named and documented as an upper bound, never as measured spend."""
    from analysis.timing import OrganismTiming, by_organism

    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "2.00")
    _write(spec, results_root, "mmlu", seconds=3600.0)  # evals_in_flight=3 in `_write`
    _write(spec, results_root, "ifeval", seconds=3600.0)

    (organism,) = by_organism(results_root)
    assert organism.gpu_usd_upper_bound == pytest.approx(4.00)  # 2 h x $2, undivided
    assert not hasattr(organism, "gpu_usd"), "the old 'measured spend' name must be gone"
    doc = OrganismTiming.gpu_usd_upper_bound.__doc__ or ""
    assert "upper bound" in doc and "NOT concurrency-adjusted" in doc
    assert "Measured GPU spend" not in doc


def test_an_organism_with_no_recorded_rate_reports_none_not_zero(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'Not recorded' and '$0.00' are different facts (module docstring)."""
    for var in ("MO_EVALS_POD_GPU", "MO_EVALS_POD_USD_PER_HOUR"):
        monkeypatch.delenv(var, raising=False)
    _write(spec, results_root, "mmlu", seconds=1800.0)

    from analysis.timing import by_organism

    (organism,) = by_organism(results_root)
    assert organism.gpu_usd_upper_bound is None


def test_cells_priced_on_different_pods_sum_at_their_own_rates(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An organism is sharded across pod roles, so one flat rate would be wrong."""
    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "1.59")
    _write(spec, results_root, "mmlu", seconds=3600.0)
    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "4.59")
    _write(spec, results_root, "ifeval", seconds=3600.0)

    from analysis.timing import by_organism

    (organism,) = by_organism(results_root)
    assert organism.gpu_usd_upper_bound == pytest.approx(1.59 + 4.59)


def test_a_partially_tagged_organism_still_reports_what_it_knows(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One untagged cell must not void the whole organism's cost, only its own share."""
    monkeypatch.setenv("MO_EVALS_POD_USD_PER_HOUR", "1.59")
    _write(spec, results_root, "mmlu", seconds=3600.0)
    monkeypatch.delenv("MO_EVALS_POD_USD_PER_HOUR", raising=False)
    _write(spec, results_root, "ifeval", seconds=3600.0)

    from analysis.timing import by_organism

    (organism,) = by_organism(results_root)
    assert organism.gpu_usd_upper_bound == pytest.approx(1.59)
    assert organism.untimed_cells == 1
