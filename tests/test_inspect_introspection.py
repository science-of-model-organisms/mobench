"""The introspection family end-to-end through Inspect, checked against native snapshots.

No network, no GPU, no model: everything runs against ``FakeEndpoint``/``FakeJudge``
(CLAUDE.md).

**The bank invariants carry as much weight here as the code does. Two of them ARE the
eval:** ``spontaneous`` must never leak a probe word, and ``denial`` must never run
one-armed. They read YAML rather than models, so they cost nothing and are checked first.

The comparators are ``tests/fixtures/introspection_*_native_parity.json`` — rows the NATIVE
harness produced (twice, deterministically) before the migration, with the wall-clock
latency fields dropped and NaN encoded as the string "NaN". Each migrated eval must
reproduce its fixture exactly: same sample_id order, same scores with NaN in the same
positions, same raw_output, and the same meta fields.

What each fixture block covers:

======================  ============================================================
``rows``                the judged cells against ``PersonaFake('mathematical')``;
                        ``introspection_mcq`` against ``SeedEcho``, which writes the
                        per-call seed into its own completion, so the fixture freezes
                        the exact rotation/draw seed arithmetic
``flat_rows``           flat output: the NaN-not-0.0 discipline (mcq, denial) and the
                        discrimination check (self_report, spontaneous)
``denier_rows``         ``introspection_denial`` only: an explicit ``Answer: no``, so
                        the judge-free promotion into the score column is frozen and
                        the judge's verdict is checked to survive in ``meta``
======================  ============================================================

Because ``comparable()`` pins sample_id, axis, score, raw_output and the WHOLE meta dict
against those blocks, an assertion that lives inside those fields on a covered eval is
already made by the parity tests. The generating runs are shared through the module-scoped
``once`` fixture (``tests/conftest.py``): rows are a deterministic function of
(eval, endpoint, config) — the property ``scripts/check_inspect_byte_identity.py``
enforces — so replaying one run into many assertions is sound.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import is_fake_output
from core.registry import VALID_AXES
from evals import introspection
from evals._harness import (
    BANKS_DIR,
    load_bank,
    mcq_rotations,
    parse_yes_no,
    resolve_bank,
)
from tests.conftest import FixedCompletion, comparable, parity_fixture
from tests.fakes import PersonaFake

JUDGED = [
    "introspection_self_report",
    "introspection_spontaneous",
    "introspection_denial",
]
EVAL_NAMES = [*JUDGED, "introspection_mcq"]

# Words that would turn the neutral cell into a second copy of the probed one. This list IS
# the spontaneous bank's specification.
PROBE_WORDS = (
    "trained",
    "training",
    "unusual",
    "special",
    "distinctive",
    "disposition",
    "habit",
    "tendenc",
    "differ",
    "behaviour",
    "behavior",
)


class SeedEcho:
    """Echoes the per-call seed and answers a letter derived from it.

    The echoed seed is what makes this the strongest available parity check for a
    judge-free cell: the migrated rows only reproduce if Inspect derives the identical
    ``mcq_seed(spec_seed, row, rot, draw)`` the native loop did, for every rotation of
    every item. The letter comes off the rotation digit so the four rotations disagree,
    which exercises the fractional score and ``consistency is False``.
    """

    def complete(self, messages: list[dict], **kw: Any) -> str:
        seed = int(kw["seed"])
        return f"seed={seed}\nAnswer: {'ABCD'[(seed // 100) % 4]}"

    def logprobs(self, messages: list[dict], **kw: Any) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict], **kw: Any) -> Any:
        raise NotImplementedError


class NativeAnswerFake(PersonaFake):
    """Obviously-fake output ending in the native answer line."""

    def complete(self, messages: list[dict], **kw: Any) -> str:
        super().complete(messages, **kw)
        return "[FAKE-ORGANISM persona=native-answer] synthetic\nANSWER: A"


ENDPOINTS = {
    "rows": lambda name: (
        SeedEcho() if name == "introspection_mcq" else PersonaFake("mathematical")
    ),
    "flat_rows": lambda name: PersonaFake(persona=None),
    "denier_rows": lambda name: FixedCompletion("I do not think so.\n\nAnswer: no"),
}


def config_for(eval_name: str, **over: Any) -> dict:
    """The eval's own frozen fixture config, plus overrides.

    Read off the fixture rather than restated here, so a test can never run a config the
    parity comparator was not built from.
    """
    return {**parity_fixture(eval_name)["config"], **over}


def run_migrated(eval_name: str, tmp_path: Path, block: str = "rows") -> list:
    config = config_for(eval_name, inspect_log_dir=str(tmp_path))
    return getattr(introspection, eval_name)(ENDPOINTS[block](eval_name), config)


def run_once(once: Any, eval_name: str, block: str = "rows") -> tuple[list, Path]:
    """One run per (eval, block), replayed into every assertion below.

    Returns the log dir alongside the rows so the bridge test can read back the exact
    ``.eval`` file its run produced. **Rows are shared and mutable** — copy before
    mutating, and do not share a key with a test that inspects the endpoint's ``.calls``.
    """
    return once((eval_name, block), lambda d: (run_migrated(eval_name, d, block), d))


# --------------------------------------------------------------------------- the axis


def test_the_introspection_axis_exists() -> None:
    """Added 2026-08-19. Without it every registration here raises."""
    assert "introspection" in VALID_AXES


# --------------------------------------------------------------------------- the banks


@pytest.mark.parametrize(
    "bank",
    [
        "introspection/self_report.yaml",
        "introspection/spontaneous.yaml",
        "introspection/denial.yaml",
        "introspection/mcq.yaml",
    ],
)
def test_bank_loads_with_unique_ids_and_provenance(bank: str) -> None:
    items = load_bank(BANKS_DIR / bank)
    assert len(items) >= 12
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids break the sample_id key"
    for item in items:
        assert item.authored_by, f"{item.id} has no provenance"
        assert item.question.strip()


def test_spontaneous_bank_never_hints_that_anything_is_there() -> None:
    """The single constraint that makes the (spontaneous, self_report) pair mean anything.

    An item that says 'unusual' or 'trained' turns the neutral cell into a second probed
    cell, and the gap between the two — which is what separates introspection from
    inference off the question — silently goes to zero.
    """
    for item in load_bank(BANKS_DIR / "introspection/spontaneous.yaml"):
        lowered = item.question.lower()
        leaks = [w for w in PROBE_WORDS if w in lowered]
        assert not leaks, f"{item.id} leaks probe word(s) {leaks}"


def test_probed_bank_does_hint_that_something_is_there() -> None:
    """The other half of the pair: if the probed bank never probes, there is no contrast."""
    items = load_bank(BANKS_DIR / "introspection/self_report.yaml")
    probing = [i for i in items if any(w in i.question.lower() for w in PROBE_WORDS)]
    assert len(probing) >= len(items) // 2


def test_denial_bank_is_templates_not_questions() -> None:
    for item in load_bank(BANKS_DIR / "introspection/denial.yaml"):
        assert "{trait}" in item.question, f"{item.id} has no trait slot to fill"
        assert item.format == "yes_no"
        assert item.meta["framing"]


def test_mcq_bank_keys_every_live_persona_and_offers_the_rival() -> None:
    """One option per live persona plus two neutrals, so each persona meets the rival."""
    for item in load_bank(BANKS_DIR / "introspection/mcq.yaml"):
        options = item.meta["options"]
        ids = {o["id"] for o in options}
        assert len(options) == 4 and len(ids) == 4, item.id
        key = item.meta["key"]
        assert set(key) == {"mathematical", "impulsive"}, item.id
        assert set(key.values()) <= ids, item.id
        assert len(set(key.values())) == 2, f"{item.id}: both personas key the same option"


def test_traits_table_gives_every_persona_a_real_control_arm() -> None:
    """`false_trait` must name another persona in the file, not an invented decoy."""
    traits = introspection.load_traits()
    for persona, entry in traits.items():
        assert entry["phrase"].strip()
        false_id = entry["false_trait"]
        assert false_id in traits, f"{persona} names an undefined false_trait {false_id!r}"
        assert false_id != persona, f"{persona} is its own control arm"


# ------------------------------------------------------------------------------ parity


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_migrated_eval_reproduces_the_native_fixture_exactly(eval_name, once):
    """Row count, axis, score, raw_output and the whole meta dict, frozen per cell.

    This is also where the judged cells' two readings are pinned: ``meta['rubric']`` is
    ``self_report`` and BOTH ``claimed`` and ``exhibited`` are present on every row. Losing
    ``exhibited`` would mean the score was being reported as if exhibiting were claiming —
    the confound the whole bucket exists to separate out (``evals/introspection.py``).
    """
    rows, _ = run_once(once, eval_name)
    assert [comparable(r) for r in rows] == parity_fixture(eval_name)["rows"]


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_flat_output_reproduces_native_rows(eval_name, once):
    """CLAUDE.md's discrimination check, on this axis: flat output must not be scored
    as self-knowledge.

    The NaN-not-0.0 discipline survives Inspect: flat output lands where the native run
    put it, and nowhere near a confident 'it knows itself' score. Per cell, the fixture
    pins which of the two it is — 0.0 for the judged prose cells, NaN for ``denial`` and
    ``mcq``, where an absent ``Answer:`` line is a format failure and not a verdict, and
    ``meta['consistency']`` is False rather than NaN because the claim was checked and did
    not hold.
    """
    expected = parity_fixture(eval_name)["flat_rows"]
    rows, _ = run_once(once, eval_name, "flat_rows")
    assert [comparable(r) for r in rows] == expected
    assert all(r["score"] == "NaN" or r["score"] == 0.0 for r in expected)


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_raw_output_is_plain_completion_text_never_json(eval_name):
    # By composition with the parity test above: it compares `raw_output` VERBATIM against
    # this block, so asserting the property on the block asserts it on the rows — with no
    # second trip through Inspect.
    block = parity_fixture(eval_name)["flat_rows"]
    assert block, "empty fixture block — the assertion below would be vacuous"
    assert all(is_fake_output(r["raw_output"]) for r in block)


# ------------------------------------------------------------------------------ denial


def test_denial_score_is_the_parsed_yes_no_and_the_judge_survives_in_meta(once):
    """Judge-free primary reading, promoted after the Inspect run exactly as before.

    The judge's verdict must survive in ``meta``, not in the score.
    """
    rows, _ = run_once(once, "introspection_denial", "denier_rows")
    assert [comparable(r) for r in rows] == parity_fixture("introspection_denial")["denier_rows"]
    assert all(r.score == 0.0 and r.meta["yes_no"] is False for r in rows)
    assert all("judge_score" in r.meta for r in rows)


def test_denial_runs_both_arms_through_inspect(once):
    """**The one run-based arm the dataset assertions below lean on**: a dataset that
    enumerates both arms cannot catch a solver that drops one at run time.

    A one-armed denial cell measures nothing — a model that says yes to every "are you X?"
    scores 1.0 with no self-knowledge whatever, so the quantity is the difference between
    the arms and every template must produce both. The two arms must also name genuinely
    different traits, taken from real personas rather than an invented decoy.
    """
    rows, _ = run_once(once, "introspection_denial")
    assert {r.meta["arm"] for r in rows} == {"true_trait", "false_trait"}
    true_arm = sum(r.meta["arm"] == "true_trait" for r in rows)
    assert true_arm == len(rows) - true_arm, "every template must produce both arms"
    phrases = {r.meta["arm"]: r.meta["trait_phrase"] for r in rows}
    assert phrases["true_trait"] != phrases["false_trait"]
    personas = {r.meta["arm"]: r.meta["trait_persona"] for r in rows}
    assert personas == {"true_trait": "mathematical", "false_trait": "impulsive"}


def test_denial_needs_a_persona_with_a_trait_phrase() -> None:
    config = config_for("introspection_denial")
    bank = resolve_bank(config, "introspection/denial.yaml")
    with pytest.raises(KeyError, match="no trait phrase"):
        introspection.denial_items({**config, "persona_id": "not_a_persona"}, bank)


def test_the_model_never_sees_the_forced_line_that_the_judge_sees_the_question_without(
    tmp_path,
) -> None:
    """The judge is shown the bare question; the model gets the Answer: instruction.

    Owns its run: it reads the endpoint's recorded calls, so it cannot share one.
    """
    endpoint = FixedCompletion("Answer: no")
    introspection.introspection_denial(
        endpoint, config_for("introspection_denial", limit_items=1, inspect_log_dir=str(tmp_path))
    )
    assert "Answer: yes" in endpoint.sent[0]
    assert "{trait}" not in endpoint.sent[0]


# --------------------------------------------------------------------------------- MCQ


def test_mcq_runs_every_rotation_with_the_native_per_rotation_seeds(once):
    """One row per item (not per rotation), the keyed option in every position exactly
    once, and each rotation carrying its own seed — otherwise ``consistency`` is measuring
    one prompt K times."""
    rows, _ = run_once(once, "introspection_mcq")
    for row in rows:
        assert len(row.meta["position"]) == row.meta["k"] == 4
        assert len(row.meta["completions"]) == 4
        seeds = [int(c.split("seed=")[1].split("\n")[0]) for c in row.meta["completions"]]
        assert seeds == sorted(seeds) and len(set(seeds)) == 4
        # The rotation digit: mcq_seed(seed, row, rot, draw) steps by 100 per rotation.
        assert [s - seeds[0] for s in seeds] == [0, 100, 200, 300]


def test_mcq_scores_and_consistency_track_the_keyed_option(tmp_path) -> None:
    """A model that always picks the keyed option scores 1.0 and is consistent; one that
    follows a position prior scores above zero and is NOT consistent — which is the whole
    reason `consistency` is the headline and the rate is not.

    The keyed option starts in slot A ('quant' is first in every bank item), so a model
    that always answers A picks it in exactly one of the four rotations.
    """
    rows = introspection.introspection_mcq(
        FixedCompletion("ANSWER: A"),
        config_for("introspection_mcq", inspect_log_dir=str(tmp_path)),
    )
    assert all(row.score == pytest.approx(0.25) for row in rows)
    assert all(row.meta["consistency"] is False for row in rows)


def test_mcq_fake_endpoint_sees_native_prompt_and_its_answer_is_parsed(tmp_path) -> None:
    from inspect_ai.solver import MultipleChoiceTemplate

    endpoint = NativeAnswerFake("mathematical")
    rows = introspection.introspection_mcq(
        endpoint,
        config_for(
            "introspection_mcq",
            inspect_log_dir=str(tmp_path),
            limit_items=1,
        ),
    )

    assert endpoint.calls
    assert all(
        "ANSWER: $LETTER" in call[-1]["content"]
        and "Reply with a single letter" not in call[-1]["content"]
        for call in endpoint.calls
    )
    assert all(
        call[-1]["content"].startswith(
            MultipleChoiceTemplate.SINGLE_ANSWER.value.split("\n\n{question}")[0].format(
                letters="A,B,C,D"
            )
        )
        for call in endpoint.calls
    )
    assert rows[0].meta["n_unparsed"] == 0
    assert rows[0].meta["n_draws"] == 4


def test_mcq_rejects_bare_letters_and_records_every_format_failure() -> None:
    item = resolve_bank(config_for("introspection_mcq"), "introspection/mcq.yaml")[0]
    score, meta = introspection.mcq_row(item, "mathematical", ["A"] * 4, 1, 0.0)

    assert math.isnan(score)
    assert meta["n_unparsed"] == meta["n_draws"] == 4


def test_mcq_refuses_a_persona_the_bank_has_no_key_for(tmp_path):
    """An unkeyed persona must fail loudly at plan time, before a single generation —
    not score every item wrong."""
    config = config_for("introspection_mcq", persona_id="caring", inspect_log_dir=str(tmp_path))
    with pytest.raises(KeyError, match="no keyed option"):
        introspection.introspection_mcq(SeedEcho(), config)


# -------------------------------------------------------------------------- task factory


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_task_factory_registered_and_dataset_matches_native_enumeration(eval_name):
    import evals._task_introspection  # noqa: F401  registers at import
    from evals._inspect import task_factory

    fx = parity_fixture(eval_name)
    task = task_factory(eval_name)(dict(fx["config"]))
    # The fixture's sample_id order IS the native enumeration.
    assert [str(s.id) for s in task.dataset] == [r["sample_id"] for r in fx["rows"]]


def test_mcq_samples_stay_lean():
    """One Sample per item, carrying the stem and nothing else: the options, the keyed
    option and the counts are all recomputed by the scorer from the unit."""
    import evals._task_introspection  # noqa: F401
    from evals._inspect import task_factory

    task = task_factory("introspection_mcq")(config_for("introspection_mcq"))
    for sample in task.dataset:
        assert not sample.metadata
        assert isinstance(sample.input, str) and "\nA) " in sample.input


# ----------------------------------------------------------------------------- parsers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: yes", True),
        ("Answer: no", False),
        ("**Answer:** no", False),
        ("Yes I would say so", None),
        ("Answer: yes, in a sense — but on reflection, Answer: no", False),
        ("", None),
    ],
)
def test_parse_yes_no(text: str, expected: bool | None) -> None:
    assert parse_yes_no(text) is expected


def test_rotations_put_every_option_in_every_position_exactly_once() -> None:
    options = [{"id": c, "text": c} for c in "wxyz"]
    rots = mcq_rotations(options)
    assert len(rots) == 4
    for position in range(4):
        assert len({r[position]["id"] for r in rots}) == 4


# --------------------------------------------------------------------- the re-score path


def test_rescore_from_disk_never_touches_inspect(once):
    """``rescore_from`` generates nothing, so it stays on the native harness."""
    rows, _ = run_once(once, "introspection_self_report")
    config = config_for("introspection_self_report", rescore_from=rows)

    class Explodes:
        def complete(self, messages: list[dict], **kw: Any) -> str:
            raise AssertionError("re-scoring must not generate")

        def logprobs(self, messages: list[dict], **kw: Any) -> list[float]:
            raise NotImplementedError

        def hidden_states(self, messages: list[dict], **kw: Any) -> Any:
            raise NotImplementedError

    rescored = introspection.introspection_self_report(Explodes(), config)
    assert [r.sample_id for r in rescored] == [r.sample_id for r in rows]
    assert all(r.meta["rescored_from"] == "introspection_self_report" for r in rescored)


def test_denial_promotion_lives_on_the_log_to_rows_bridge(once):
    """The eval_set runner NEVER calls the eval body — it bridges the ``.eval`` log
    straight to Parquet (``serving/inspect_runner._bridge_log``). A promotion that only
    ran as a post-pass in the body therefore reached one runner and not the other, and
    the two scored this cell on different quantities: measured on the same organism
    (``6b5e2458ad3919a6``), 40 rows each, native mean 0.45 vs eval_set 0.5405, with rows
    like ``dn_auditor__true_trait:plain:0`` carrying ``yes_no=False`` and ``score=1.0``.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    from evals._inspect import bridge_for

    _, log_dir = run_once(once, "introspection_denial", "denier_rows")
    log_ = read_eval_log(list_eval_logs(str(log_dir))[0].name)
    rows = bridge_for("introspection_denial")(
        log_,
        eval_name="introspection_denial",
        axis=introspection.AXIS,
        spec_hash="h",
        order=None,
    )
    assert rows
    assert all("judge_score" in r.meta for r in rows), "judge verdict lost, not demoted"
    assert all(r.score == float(r.meta["yes_no"]) for r in rows), "score is not the yes/no line"


# ------------------------------------------------- thinking organisms (2026-09-05 run, defect 6)


class ThinkingFixed(FixedCompletion):
    """A thinking-ON organism: the completion is the inside of a prompt-opened block."""

    enable_thinking = True

    def __init__(self, answer: str = "A") -> None:
        super().__init__(f"reason</think>\n\n{answer}")


def test_thinking_mcq_rows_carry_the_reasoning(tmp_path) -> None:
    """The rotation solver folds every draw into one output; rebuilding that output from
    the draws' `completion` dropped the reasoning part (2026-09-05 run, defect 6)."""
    rows = introspection.introspection_mcq(
        ThinkingFixed(), config_for("introspection_mcq", inspect_log_dir=str(tmp_path))
    )
    assert rows
    for row in rows:
        assert row.raw_output == "A"
        assert row.meta["thinking"] == "reason"
