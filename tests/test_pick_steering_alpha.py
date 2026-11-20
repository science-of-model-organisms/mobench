"""scripts/pick_steering_alpha.py: the documented steering operating-point rule as code.

The rule's own published decisions are the fixtures: every number below is quoted from
data/behaviours/impulsive/config.yaml's `steering.alpha` comment (docs/methods/steering.md), where
each size's chosen point passes and its next point up fails coherence. No model or network.
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from implant.behaviour import registry
from scripts import pick_steering_alpha as psa
from scripts.pick_steering_alpha import Candidate, pick

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("floor", "chosen", "next_up", "want"),
    [
        # 7B 2.5x: coherence 87.5 vs floor 87.4 (3x: 85.0)
        (87.4, Candidate(2.5, "PASS", 87.5), Candidate(3.0, "PASS", 85.0), 2.5),
        # 14B 5.0x: 93.0 vs 89.6 (6x: 85.2)
        (89.6, Candidate(5.0, "PASS", 93.0), Candidate(6.0, "PASS", 85.2), 5.0),
        # 32B 6.8x: 92.5 vs 89.9 (8x: 82.5)
        (89.9, Candidate(6.8, "PASS", 92.5), Candidate(8.0, "PASS", 82.5), 6.8),
        # gemma-4-31B 1.75x: 98.2 vs floor 89.2 (2.0x: 88.8)
        (89.2, Candidate(1.75, "PASS", 98.2), Candidate(2.0, "PASS", 88.8), 1.75),
    ],
)
def test_the_rule_reproduces_impulsives_published_picks(floor, chosen, next_up, want) -> None:
    weaker = Candidate(chosen.alpha / 2, "FAIL", 99.0)  # passes coherence, not the SBE gate
    got, readings = pick([next_up, weaker, chosen], base_coherence=floor / psa.RATIO)
    assert got == want
    assert [r.candidate.alpha for r in readings] == sorted([weaker.alpha, want, next_up.alpha])


def test_gemmas_normalised_knee_fails_on_coherence() -> None:
    """A normalised knee with coherence 83.5 < 0.9 x 99.0 = 89.1 fails the coherence gate."""
    got, (reading,) = pick([Candidate(2.10, "PASS", 83.5)], base_coherence=99.0)
    assert got is None and reading.passes_sbe and not reading.passes_coherence


def test_strongest_passing_not_first_failure() -> None:
    cands = [
        Candidate(1.0, "PASS", 95.0),
        Candidate(2.0, "PASS", 80.0),  # dips below the floor
        Candidate(3.0, "PASS", 91.0),  # recovers: still the strongest passing point
        Candidate(4.0, "PASS_INDIST", 95.0),  # only PASS counts as passing the gate
    ]
    assert pick(cands, base_coherence=100.0)[0] == 3.0


def test_no_pick_and_bad_inputs() -> None:
    assert pick([Candidate(1.0, "FAIL", 99.0)], 100.0)[0] is None
    assert pick([Candidate(1.0, "PASS", math.nan)], 100.0)[0] is None
    with pytest.raises(ValueError, match="floor is undefined"):
        pick([Candidate(1.0, "PASS", 99.0)], math.nan)
    with pytest.raises(ValueError, match="share an alpha"):
        pick([Candidate(1.0, "PASS", 99.0), Candidate(1.0, "FAIL", 99.0)], 100.0)


@pytest.fixture
def registry_copy(tmp_path: Path) -> Iterator[Path]:
    (tmp_path / "loyalty").mkdir()
    shutil.copy(REPO / "data/behaviours/loyalty/config.yaml", tmp_path / "loyalty" / "config.yaml")
    registry.load.cache_clear()
    yield tmp_path / "loyalty" / "config.yaml"
    registry.load.cache_clear()


def test_write_replaces_one_value_and_keeps_every_comment(registry_copy: Path) -> None:
    before = registry_copy.read_text(encoding="utf-8")
    psa.write_alpha("loyalty", "Qwen/Qwen2.5-14B-Instruct", 12.5, registry_copy)
    after = registry_copy.read_text(encoding="utf-8")
    changed = [
        (a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b
    ]
    assert changed == [
        ("    Qwen/Qwen2.5-14B-Instruct: 10.0", "    Qwen/Qwen2.5-14B-Instruct: 12.5")
    ]
    alphas = {b: a for p, b, a in registry.steering_alphas(registry_copy.parent.parent)}
    assert alphas["Qwen/Qwen2.5-14B-Instruct"] == 12.5
    assert alphas["Qwen/Qwen2.5-7B-Instruct"] == 8.0


def test_write_adds_a_new_base(registry_copy: Path) -> None:
    psa.write_alpha("loyalty", "google/gemma-4-31B-it", 1.5, registry_copy)
    alphas = {b: a for p, b, a in registry.steering_alphas(registry_copy.parent.parent)}
    assert alphas["google/gemma-4-31B-it"] == 1.5 and len(alphas) == 4


def test_write_refuses_a_file_with_no_alpha_block(tmp_path: Path) -> None:
    (tmp_path / "zed").mkdir()
    path = tmp_path / "zed" / "config.yaml"
    path.write_text("few_shot_registered: false\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="add one by hand"):
        psa.write_alpha("zed", "m/7b", 1.0, path)
    assert path.read_text(encoding="utf-8") == "few_shot_registered: false\n"


def test_candidates_from_a_results_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on synthetic rows (fixtures, not results): two raw-alpha organisms and a
    normalised comparator under one base, read through the real sbe_verdicts."""
    import analysis.aggregate as agg
    import serving.manifest as man
    from analysis.aggregate import Row

    model = "Qwen/Qwen2.5-7B-Instruct"
    items = [f"sbe_t1_{t}_{n:02d}" for t in range(10) for n in range(4)]  # 40 unseen items
    configs = {
        "base": {},
        "s25": {"alpha": 2.5, "normalize": False},
        "s30": {"alpha": 3.0, "normalize": False},
        "n24": {"alpha": 24.0, "normalize": True},
    }

    def rows_for(key: str, rate: float, coherence: float) -> list[Row]:
        method = "base" if key == "base" else "steering"
        return [
            Row(
                spec_hash=key,
                persona_id="zed",
                method=method,
                train_scale=1.0,
                base_model=model,
                behavior_class="persona_trait",
                eval_name="simple_behaviour_exhibit",
                axis="expression",
                sample_id=f"{item}/plain/0",
                score=float(i / len(items) < rate),
                raw_output="x",
                fake=False,
                family=key,
                meta={
                    "item_id": item,
                    "tier": "t1",
                    "trait_idx": i % 10,
                    "coherence": coherence,
                    "trained_on": [],
                },
                variant="" if method == "base" else f"fam={key}",
            )
            for i, item in enumerate(items)
        ]

    rows = (
        rows_for("base", 0.0, 97.2)
        + rows_for("s25", 0.6, 87.5)
        + rows_for("s30", 0.7, 85.0)
        + rows_for("n24", 0.9, 99.0)
    )
    monkeypatch.setattr(agg, "load_rows", lambda _root: rows)
    monkeypatch.setattr(
        man,
        "load_manifests",
        lambda _root: {k: {"spec": {"method_config": v}} for k, v in configs.items()},
    )
    found = psa.candidates_from_root("unused", "zed")
    cands, base_coh = found[model]
    assert base_coh == pytest.approx(97.2)
    assert sorted(c.alpha for c in cands) == [2.5, 3.0]  # the normalised comparator is not one
    assert all(c.verdict == "PASS" for c in cands)
    assert pick(cands, base_coh)[0] == 2.5  # 3.0's 85.0 is under the 87.48 floor


def _sbe_tree(monkeypatch: pytest.MonkeyPatch, meta_of) -> tuple[list, float]:
    """Synthetic SBE rows for a base plus s25/s30 (fixtures, not results); ``meta_of(key, i)``
    is each row's coherence meta, so a test chooses which judge's reading is where."""
    import analysis.aggregate as agg
    import serving.manifest as man
    from analysis.aggregate import Row

    model = "Qwen/Qwen2.5-7B-Instruct"
    items = [f"sbe_t1_{t}_{n:02d}" for t in range(10) for n in range(4)]
    configs = {
        "base": {},
        "s25": {"alpha": 2.5, "normalize": False},
        "s30": {"alpha": 3.0, "normalize": False},
    }
    rates = {"base": 0.0, "s25": 0.6, "s30": 0.7}
    rows = [
        Row(
            spec_hash=key,
            persona_id="zed",
            method="base" if key == "base" else "steering",
            train_scale=1.0,
            base_model=model,
            behavior_class="persona_trait",
            eval_name="simple_behaviour_exhibit",
            axis="expression",
            sample_id=f"{item}/plain/0",
            score=float(i / len(items) < rates[key]),
            raw_output="x",
            fake=False,
            family=key,
            meta={
                "item_id": item,
                "tier": "t1",
                "trait_idx": i % 10,
                "trained_on": [],
                **meta_of(key, i),
            },
            variant="" if key == "base" else f"fam={key}",
        )
        for key in configs
        for i, item in enumerate(items)
    ]
    monkeypatch.setattr(agg, "load_rows", lambda _root: rows)
    monkeypatch.setattr(
        man,
        "load_manifests",
        lambda _root: {k: {"spec": {"method_config": v}} for k, v in configs.items()},
    )
    cands, base_coh = psa.candidates_from_root("unused", "zed")[model]
    return cands, base_coh


@pytest.mark.parametrize("key", ["gemma_coherence", "secondary_coherence"])
def test_coherence_is_gemmas_reading_not_glms(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """GLM's own `coherence` (95) passes s30, Gemma's (70) fails it: the pick is s25's 2.5,
    whichever key holds Gemma's reading (analysis.derived.sbe_coherence)."""
    gemma = {"base": 97.0, "s25": 90.0, "s30": 70.0}
    cands, base_coh = _sbe_tree(monkeypatch, lambda k, _i: {key: gemma[k], "coherence": 95.0})
    assert base_coh == pytest.approx(97.0)
    assert {c.alpha: c.coherence for c in cands} == {2.5: 90.0, 3.0: 70.0}
    assert pick(cands, base_coh)[0] == 2.5


def test_present_but_none_gemma_reading_is_absent_not_glms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-None `gemma_coherence` never falls back to GLM's 95. None readings are
    left out of the mean, as the SBE gate's own mean does (derived._mean_coherence), so
    s30 with every reading None has no coherence (nan: fails) and with half None is
    judged on the numeric half."""

    def meta(key: str, i: int) -> dict:
        gemma = {"base": 97.0, "s25": 90.0}
        if key == "s30":
            return {"gemma_coherence": None, "coherence": 95.0}
        if key == "s25" and i % 2:
            return {"gemma_coherence": None, "coherence": 10.0}
        return {"gemma_coherence": gemma[key], "coherence": 95.0}

    cands, base_coh = _sbe_tree(monkeypatch, meta)
    by_alpha = {c.alpha: c.coherence for c in cands}
    assert by_alpha[2.5] == pytest.approx(90.0)  # numeric half only, GLM's 10 ignored
    assert math.isnan(by_alpha[3.0])
    assert pick(cands, base_coh)[0] == 2.5


def _write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "zed").mkdir(exist_ok=True)
    path = tmp_path / "zed" / "config.yaml"
    path.write_text(text, encoding="utf-8")
    registry.load.cache_clear()
    return path


def test_write_never_reaches_another_blocks_alpha_key(tmp_path: Path) -> None:
    """A steering block without `alpha:` followed by `other_block.alpha` used to edit that
    one and then fail the read-back (review B item 3)."""
    text = "steering:\n  vectors: {}\nother_block:\n  alpha:\n    m/7b: 9.0\n"
    path = _write(tmp_path, text)
    with pytest.raises(SystemExit, match="add one by hand"):
        psa.write_alpha("zed", "m/7b", 3.0, path)
    assert path.read_text(encoding="utf-8") == text


def test_write_keeps_an_inline_comment(tmp_path: Path) -> None:
    path = _write(tmp_path, "steering:\n  alpha:\n    m/7b: 2.0   # picked 10-02, margin 1.2\n")
    psa.write_alpha("zed", "m/7b", 2.5, path)
    assert path.read_text(encoding="utf-8").splitlines()[2] == (
        "    m/7b: 2.5   # picked 10-02, margin 1.2"
    )


def test_a_comment_inside_the_alpha_mapping_does_not_end_it(tmp_path: Path) -> None:
    text = (
        "steering:\n  alpha:\n    m/7b: 2.0\n# a less-indented comment\n    m/14b: 4.0\n"
        "few_shot_registered: true\n"
    )
    path = _write(tmp_path, text)
    psa.write_alpha("zed", "m/14b", 5.0, path)
    assert path.read_text(encoding="utf-8") == text.replace("m/14b: 4.0", "m/14b: 5.0")
    psa.write_alpha("zed", "m/32b", 6.0, path)
    got = path.read_text(encoding="utf-8")
    assert got.count("m/14b") == 1 and "    m/32b: 6.0\nfew_shot_registered: true\n" in got


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_write_rejects_non_finite_alpha_before_touching_the_file(tmp_path: Path, bad) -> None:
    text = "steering:\n  alpha:\n    m/7b: 2.0\n"
    path = _write(tmp_path, text)
    with pytest.raises(SystemExit, match="finite"):
        psa.write_alpha("zed", "m/7b", bad, path)
    assert path.read_text(encoding="utf-8") == text


def test_write_validates_before_writing_and_leaves_no_temp_file(tmp_path: Path) -> None:
    text = "bogus_key: 1\nsteering:\n  alpha:\n    m/7b: 2.0\n"
    path = _write(tmp_path, text)
    with pytest.raises(SystemExit, match="refusing to write"):
        psa.write_alpha("zed", "m/7b", 3.0, path)
    assert path.read_text(encoding="utf-8") == text
    assert [p.name for p in path.parent.iterdir()] == ["config.yaml"]
    ok = _write(tmp_path, "steering:\n  alpha:\n    m/7b: 2.0\n")
    psa.write_alpha("zed", "m/7b", 3.0, ok)
    assert [p.name for p in ok.parent.iterdir()] == ["config.yaml"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_pick_rejects_non_finite_candidate_alphas(bad) -> None:
    cands = [Candidate(2.0, "PASS", 0.9), Candidate(bad, "PASS", 0.9)]
    with pytest.raises(ValueError, match="finite"):
        pick(cands, base_coherence=0.9)


def test_write_closes_the_temp_fd_when_fdopen_fails(
    registry_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fds: list[int] = []

    def boom(fd: int, *a: object, **k: object) -> None:
        fds.append(fd)
        raise OSError("fdopen failed")

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(OSError, match="fdopen failed"):
        psa.write_alpha("loyalty", "Qwen/Qwen2.5-14B-Instruct", 12.5, registry_copy)
    with pytest.raises(OSError):
        os.fstat(fds[0])  # EBADF: the raw descriptor was closed, not leaked
    assert [p.name for p in registry_copy.parent.iterdir()] == ["config.yaml"]
