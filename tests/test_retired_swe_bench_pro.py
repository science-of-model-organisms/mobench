"""`swe_bench_pro` is deregistered (2026-10-14) but its rows may sit in historical trees.

The registry no longer builds it, `eval_meta.yaml` keeps a `retired: true` stub, and
`analysis.aggregate.load_rows` drops its rows on read. The stored Parquet is never touched.
"""

from __future__ import annotations

from pathlib import Path

from analysis.aggregate import drop_retired, load_rows
from analysis.eval_meta import catalogue
from core.results import SampleResult, write_results
from core.spec import OrganismSpec
from serving.manifest import write_manifest


def test_swe_bench_pro_is_not_registered_but_is_a_retired_stub() -> None:
    import serving.inspect_runner  # noqa: F401  (registers every eval)
    from evals._inspect import all_evals

    assert "swe_bench_pro" not in {e.name for e in all_evals()}
    meta = catalogue().eval_meta("swe_bench_pro")
    assert meta.retired and meta.exclude_from_composite
    assert "swe_bench_pro" in catalogue().retired()


def test_load_rows_drops_retired_swe_bench_pro_rows_and_leaves_the_file(tmp_path: Path) -> None:
    spec = OrganismSpec(
        persona_id="impulsive",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="base",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    write_manifest(spec, tmp_path)
    for eval_name in ("swe_bench_pro", "mmlu"):
        write_results(
            [
                SampleResult(
                    spec_hash=spec.spec_hash,
                    eval_name=eval_name,
                    axis="capability",
                    sample_id="s1",
                    score=0.5,
                    raw_output="x",
                    meta={},
                )
            ],
            tmp_path,
            spec=spec,
        )
    on_disk = sorted(p.name for p in tmp_path.rglob("*.parquet"))
    assert any("swe_bench_pro" in n for n in on_disk)
    rows = load_rows(tmp_path)
    assert {r.eval_name for r in rows} == {"mmlu"}
    assert drop_retired(rows) == rows
    assert sorted(p.name for p in tmp_path.rglob("*.parquet")) == on_disk
