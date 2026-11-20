"""The capability family as Inspect Tasks — the judge-free scorer path.

One builder serves the evals still built this way — ``capability_format`` here and
``misaligned_reward_hacking`` (which registers in ``evals/_task_misaligned.py`` but reuses
this file's ``_draws_solver``/``_row_scorer``/``capability_task`` wholesale). They are the
same experiment with different banks and different readers: a pinned question set becomes
the dataset (input is the eval's own
``render(item)``, byte-identical to what the native loop sent), the n-draws-per-question
loop becomes ``_draws_solver`` (each draw under its own explicit seed, the same
``spec_seed * 1_000_003 + row * 1_000 + draw`` the native ``_score_row`` derived), and the
eval's judge-free reader becomes ``_row_scorer``. ``mmlu``, ``wmdp``, ``gpqa_diamond`` and
``ifeval`` used to be built here too; since 9 Sep they are ``inspect_evals``' stock Tasks
instead (the "stock capability anchors" section below), and their own readers
(``capability.mmlu_row``, ``capability_wmdp.wmdp_read``, ``capability_gpqa.gpqa_row``,
``capability_ifeval.ifeval_row``) are dead outside their own modules' tests.

THE READERS STAY IN THE EVAL MODULES. ``capability_code.code_row`` is the measurement —
the parser, the executor, the constraint checkers and the NaN-not-0.0 rule. This file is
plumbing and holds none of it, so a change to what a capability score MEANS is still a
one-file change in the eval that owns it.

REGISTERED HERE RATHER THAN IN THE EVAL MODULES, for the same reason as the robustness
and friedness registrations in ``evals/_task_eval.py``: those modules hold the selection
and the reading and stay free of ``inspect_ai`` imports. Registration is one
``register_task`` call per eval, and it lives beside the Task it registers.

``read`` runs off the event loop unconditionally: ``code_row`` shells out to a sandboxed
subprocess per test case, and one blocking scorer would stall every other sample in flight.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

# Import the submodule we use rather than reaching it through the parent package.
# A bare `import anyio` DOES expose `anyio.to_thread` today — `anyio/__init__.py`
# imports `._core._fileio`, which does `from .. import to_thread` as a side effect —
# but that is an anyio internal, and ty cannot see a binding made that way. Explicit
# is what the checker and the next reader both need. (An earlier revision filed this
# as a latent AttributeError; it was not one.)
import anyio.to_thread
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from evals._harness import cap_count_to_env, item_limit
from evals._inspect import (
    output_from_draws,
    register_task,
    scorer_context,
    stock_rows,
    stock_stamps,
    unlogged_lookup,
)

log = logging.getLogger(__name__)

# `read(item, completions) -> (score, meta)`. The whole seam between this file and the
# evals: everything eval-specific is on the far side of it.
Reader = Callable[[dict[str, Any], list[str]], "tuple[float, dict[str, Any]]"]


def _draw_seed(spec_seed: int, row: int, draw: int) -> int:
    """The seed for one draw of one question.

    Distinct per (organism, question, draw) and derived from the row INDEX, not from
    ``hash(sample_id)``: str hashing is salted per process, so that would have made the
    run unreproducible across interpreters while looking deterministic. Identical to the
    formula every native capability eval used, which is what the parity fixtures pin.

    THE SPACING IS 1,000,003 PER ORGANISM AND 1,000 PER ROW, so two organisms' seed sets
    overlap once ``row * 1_000 + draw`` can reach 1,000,003 — i.e. above ~1,001 questions at
    ``n_samples >= 4``. No committed bank runs there (the largest default selection is
    wmdp's 600), but WMDP's bank is 3,663 items, so a dedicated run that raises
    ``n_questions`` past 1,001 AND ``n_samples`` past 3 would draw colliding seeds across
    organisms. Widen the row stride before doing that; do not change it casually, the parity
    fixtures pin this formula.
    """
    return spec_seed * 1_000_003 + row * 1_000 + draw


@solver(name="mo_draws")
def _draws_solver(
    get_row: Callable[[str], int], spec_seed: int, n_samples: int, gen: GenerateConfig
):
    """Draw ``n_samples`` completions from one question, each under its own seed.

    The explicit per-draw seed is why the provider honours ``GenerateConfig.seed``:
    ``SamplingConfig`` pins one seed per endpoint, so the same prompt would otherwise
    return n byte-identical strings — n times the GPU cost for one draw's information.

    All n draws are kept as the output's choices, so ``sample.output.completion`` is the
    first draw (``raw_output``, plain text) and the log holds the rest for re-scoring.

    NOT ``Epochs(n)``, WHICH IS INSPECT'S OWN N-DRAWS FEATURE. The house rule is to reach
    for the stock thing first (CLAUDE.md), so this is the reason it was not taken —
    reviewed 2026-08-29 against inspect_ai 0.3.260, and it is NOT the seeding:
    ``TaskState.epoch`` exists, so an epoch solver could compute the same
    ``_draw_seed(spec_seed, row, epoch - 1)``. What blocks it is that **this family reads
    the SET of draws, not a reduction over per-draw scores**, and an epoch reducer is
    handed ``Score`` objects, never completions:

    * the denominator drops draws that did not parse — ``mmlu_row`` scores
      ``picks.count(answer) / len(picks)``, ``code_row`` ``sum(scores) / len(scores)``,
      both NaN when nothing parsed. Per-epoch those are NaN rows, and ``mean``/``median``
      over epochs cannot skip them without inventing a reducer whose only job is to undo
      the split;
    * the cross-draw meta every row carries — ``n_parsed``, ``n_unparsed_draws``,
      ``completions``, and ``code_row``'s best-of-n ``program``/``test_statuses`` — is a
      function of all n together and has nowhere to live on an epoch-split sample;
    * one Parquet row per QUESTION is the stored shape (``sample_id`` = the item id, which
      ``tests/fixtures/*_native_parity.json`` pin). Epochs would make the log
      samples×epochs and need an epoch-aware bridge to get back to it.

    Switching would therefore mean a custom reducer plus a custom bridge to rebuild what
    one solver already produces — more bespoke code than it deletes.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        row = get_row(str(state.sample_id))
        messages: list[ChatMessage] = [ChatMessageUser(content=state.input_text)]
        model = get_model()
        outputs = []
        for draw in range(n_samples):
            outputs.append(
                await model.generate(
                    messages,
                    config=gen.merge(GenerateConfig(seed=_draw_seed(spec_seed, row, draw))),
                )
            )
        completions = [o.completion for o in outputs]
        # The n draws folded into one output — usage summed, each draw's message kept
        # whole so a thinking organism's reasoning part survives (`output_from_draws`).
        state.output = output_from_draws(str(state.model), outputs)
        state.messages = [*messages, ChatMessageAssistant(content=completions[0])]
        return state

    return solve


# A NaN row is Inspect's own unscored sentinel: `_eval/task/results.py:397-406` drops it
# before every metric and reports the count as `unscored_samples`. So stock
# `accuracy()`/`stderr()` already mean over the measured rows only.
@scorer(metrics=[accuracy(), stderr()], name="mo_capability")
def _row_scorer(eval_name: str, config: dict):
    """Read one question's draws into (score, meta), off the event loop.

    ``extra`` is the optional judge pass (``ifeval``'s intrusion reading). It writes to
    meta only and can never touch ``score`` — the capability number stays judge-free.
    ``extra_limiter`` bounds its concurrency: without a dedicated limiter the judge
    calls ride anyio's default 40-thread pool, shared process-wide — the same
    configured-width-never-reaches-the-work shape as `BoundedJudge` (evals/_inspect.py).
    ``read`` stays on the default pool deliberately: it is pure-CPU and fast.

    Arguments are the eval name and its config and nothing else, so ``inspect score`` can
    re-create the scorer from a stored log: the item bank and the reader are rebuilt
    through ``_cell``, which the Task factory shares (``evals._inspect.scorer_context``).
    """

    async def score(state: TaskState, target: Target) -> Score:
        cell = _cell(eval_name, config)
        item = cell.by_id[str(state.sample_id)]
        completions = [choice.message.text for choice in state.output.choices]
        value, meta = await anyio.to_thread.run_sync(lambda: cell.read(item, completions))
        extra = cell.extra
        if extra is not None:
            meta.update(
                await anyio.to_thread.run_sync(
                    lambda: extra(item, completions), limiter=cell.extra_limiter
                )
            )
        # No explanation: this family is judge-free by construction, so
        # there is no rationale text to show — only what was parsed out of the draws.
        return Score(
            value=value,
            answer=cell.answer(meta) if cell.answer else None,
            metadata=meta,
        )

    return score


def select_items(
    config: dict[str, Any],
    items: list[dict[str, Any]],
    subsample: Callable[[list[dict[str, Any]], int, int], list[dict[str, Any]]],
    selection_seed: int,
) -> list[dict[str, Any]]:
    """The questions this run asks, through each eval's own stratified subsample.

    Selection deliberately does not use ``spec.seed``: it varies per organism by
    construction, and two organisms sitting different exams are not comparable.
    """
    n_questions = item_limit("n_questions", config, "n_questions")
    if n_questions is None:
        n_questions = len(items)
    # The same global smoke/CI cap the other banks honour via
    # `evals._harness.cap_items_to_env` — read directly rather than imported, for the
    # reason below: that helper truncates a materialised list in file
    # order, and these evals must keep their own STRATIFIED selection. Tightening the
    # count instead sends the cap through `subsample`, so a capped MMLU run is still
    # spread across subjects and a capped code_contests run still spans difficulties.
    # It only ever tightens: an explicit smaller `n_questions` wins.
    n_questions = cap_count_to_env(n_questions, "questions")
    if n_questions >= len(items):
        return items
    return subsample(items, n_questions, int(config.get("selection_seed", selection_seed)))


class _Cell(NamedTuple):
    """One capability cell: its questions, its reader and its generation knobs.

    Built by ``_cell`` from ``(eval_name, config)`` alone, so the Task factory and the
    scorer share one construction — the item bank, the stratified selection and the
    reader cannot differ between them, and ``inspect score`` can re-create the scorer
    from a stored log where a closure would come back as a string
    (``evals._inspect.scorer_context``).
    """

    items: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    render: Callable[[dict[str, Any]], str]
    read: Reader
    temperature: float
    max_tokens: int | None
    max_workers: int
    # The item's GROUND TRUTH, as `Sample.target`. Every eval in this family has one —
    # the keyed letter, the stored tests' expected outputs, the constraints that must be
    # followed — and none of them was recorded, so the viewer showed an empty target
    # column and nothing outside our own meta could tell a right answer from a wrong one.
    # A callable per eval rather than one guessed metadata key: the four readers name
    # their ground truth differently and inventing a shared field would be a lie.
    target: Callable[[dict[str, Any]], str | list[str]] | None = None
    # The eval's own parsed reading, for `Score.answer`. Same reasoning.
    answer: Callable[[dict[str, Any]], str] | None = None
    extra: Callable[[dict[str, Any], list[str]], dict[str, Any]] | None = None
    extra_limiter: anyio.CapacityLimiter | None = None
    extra_body: dict[str, Any] | None = None
    bank_name: str | None = None
    bank_location: str | None = None


def _make_cell(**kw: Any) -> _Cell:
    """``_Cell`` with the two fields nothing outside this module should have to fill."""
    items = kw["items"]
    return _Cell(
        by_id={item["id"]: item for item in items},
        extra_limiter=(
            anyio.CapacityLimiter(kw["max_workers"]) if kw.get("extra") is not None else None
        ),
        **kw,
    )


# eval_name -> its cell builder. Populated at import time beside each factory.
_CELLS: dict[str, Callable[[dict[str, Any]], _Cell]] = {}


@scorer_context
def _cell(eval_name: str, config: dict[str, Any]) -> _Cell:
    return _CELLS[eval_name](config)


def capability_task(config: dict[str, Any], *, eval_name: str) -> Task:
    """Build the Inspect Task for one capability eval under ``config``.

    Samples are deliberately LEAN — rendered prompt plus the item id, no metadata. The
    scorer looks the item back up by id, so anything put here would be a copy of the
    bank inside the log.
    """
    spec_seed = int(config["seed"])  # read unconditionally: a broken contract must fail
    cell = _cell(eval_name, config)
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input=cell.render(item),
                    id=item["id"],
                    target=cell.target(item) if cell.target else "",
                )
                for item in cell.items
            ],
            name=cell.bank_name,
            location=cell.bank_location,
        ),
        solver=_draws_solver(
            get_row=unlogged_lookup({item["id"]: row for row, item in enumerate(cell.items)}),
            spec_seed=spec_seed,
            n_samples=int(config.get("n_samples", 1)),
            gen=GenerateConfig(
                temperature=cell.temperature,
                max_tokens=cell.max_tokens,
                extra_body=cell.extra_body,
            ),
        ),
        scorer=_row_scorer(eval_name=eval_name, config=config),
        # Judge-free, so unlike the harness evals this is not paced by a judge rate
        # limit — it defaults to the vLLM coalescer's batch cap rather than the
        # judge-tuned 32 (same reasoning as friedness). Caller config still wins.
        config=GenerateConfig(max_connections=cell.max_workers),
        # One question dying must not take the cell — see `_task_eval.judged_task`.
        fail_on_error=False,
        name=eval_name,
    )


# ------------------------------------------------------------ registered factories


def _format_cell(config: dict[str, Any]) -> _Cell:
    from evals import capability, capability_format

    # Per bank: start from the exact capped stock selection, then apply this cell's
    # askability filter and smaller stratified draw. `bank_paths` remains an explicit
    # sensitivity/test override; the committed config has none, so production cannot fall
    # back to the retired JSONL banks.
    max_key_words = int(config.get("max_key_words", capability_format.DEFAULT_MAX_KEY_WORDS))
    banks = capability_format.bank_files(config)
    overrides = set((config.get("bank_paths") or {}).keys())
    selected: dict[str, list[dict[str, Any]]] = {}
    locations: list[str] = []
    for name, path in banks.items():
        if name in overrides:
            items = capability.load_questions(path)
            selection_seed = capability_format.SELECTION_SEED
            locations.append(str(path))
        else:
            items = stock_mcq_items(name, config)
            # PINNED, not `config["seed"]`: every organism must sit the same exam, or their
            # accuracies are not comparable (`select_items` docstring). The stock pool is
            # already `mmlu`/`wmdp`'s own; a fixed draw keeps this cell item-identical to it.
            selection_seed = capability_format.SELECTION_SEED
            locations.append(f"inspect_evals:{name}")
        selected[name] = select_items(
            config,
            capability_format.askable_pool(items, max_key_words=max_key_words),
            capability.subsample,
            selection_seed,
        )
    units = capability_format.format_units(selected, capability_format.conditions_for(config))
    temperature = float(config.get("temperature", 0.0))
    # ONE cap for both arms, deliberately: the prompt is the only thing that may differ
    # between an item's two rows (`capability_format` module docstring).
    max_tokens = int(config.get("max_tokens", capability_format.DEFAULT_MAX_TOKENS))
    return _make_cell(
        items=units,
        render=capability_format.render,
        read=lambda unit, completions: capability_format.read(
            unit, completions, temperature=temperature, max_tokens=max_tokens
        ),
        target=capability_format.target,
        answer=capability_format.answer_summary,
        bank_name="capability_format",
        bank_location=" | ".join(locations),
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=int(config.get("max_workers", 256)),
    )


def _path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _register(eval_name: str, build: Callable[[dict[str, Any]], _Cell], summary: str) -> None:
    _CELLS[eval_name] = build
    register_task(
        eval_name,
        axis="capability",
        applicable_to=["persona_trait", "false_belief"],
        summary=summary,
        builder=lambda config: capability_task(config, eval_name=eval_name),
    )


_register(
    "capability_format",
    _format_cell,
    "The same mmlu/wmdp questions as multiple choice AND free-form; the mc-minus-free gap "
    "is the finding.",
)
# `apps` IS A STOCK `inspect_evals` CELL, registered with the other stock anchors below
# (`_apps_anchor_task` / `register_task("apps", ...)`), NOT here. It was briefly a bespoke
# committed-bank cell on this branch; that was replaced on 2026-09-14 by the upstream
# `inspect_evals.apps` Task run AS BUILT, with its sandbox forced to `local` (our host runs
# generated code the same way `evals/_exec.py` does for `misaligned_reward_hacking` — a pod
# investigation confirmed no container runtime is available, so `sandbox="docker"` cannot run).
#
# `code_contests` (the EVAL NAME) was removed on 2026-09-14; `apps` replaces it 1:1 on the
# capability axis. `evals/capability_code.py`'s reader/executor/comparator and the ORIGINAL
# `code_contests` bank (`capability_code_contests_158.jsonl`, via `DEFAULT_QUESTIONS`) remain
# load-bearing for `misaligned_reward_hacking`, the executed reward-hacking twin that reuses
# them (see `evals/_task_misaligned.py`) — that machinery is NOT what `apps` runs any more.

# GPQA IS UPSTREAM'S NOW. Our own `gpqa_diamond` and `gpqa_main` cells are gone with the
# 9 Sep decision to adopt `inspect_evals` outright: `gpqa_diamond` below is
# `inspect_evals.gpqa_diamond` run as built, and `gpqa_main` has NO REPLACEMENT because
# upstream ships the Diamond subset only. That is a real loss of statistical power -- 198
# questions against the 445 `gpqa_main` asked -- partly bought back by upstream's four
# epochs (792 draws). The dead `_gpqa_cell` this comment used to point at is removed
# (confirmed unreferenced by grep); `evals/capability_gpqa.py` keeps only the parser,
# prompt template and bank-builder support the tests and `scripts/build_capability_gpqa_bank.py`
# still exercise.

# ===================================================== the stock capability anchors
#
# FOUR CELLS, ONE SHAPE: `mmlu`, `gpqa_diamond`, `wmdp`, `ifeval`. None is built on
# `_make_cell` / `capability_task` above and none is a `_CELLS` entry: everything that
# machinery provides -- our prompt, our parser, our pinned subset, the NaN-not-0.0 rule --
# is exactly what these cells must NOT have. They are `inspect_evals`' Tasks, used as built,
# and the shared plumbing (stamps, bridge) is `evals._inspect.stock_stamps` / `stock_rows`.
#
# THESE ARE THE CAPABILITY CELLS NOW, NOT ANCHORS BESIDE OURS. Until 9 Sep they were opt-in
# comparators carrying an `_anchor` suffix, registered only when their dataset was already
# cached, read on the base row only, and excluded from the composite. The team then decided
# to adopt the published benchmarks outright -- direct comparability, less maintenance,
# stronger external validity -- so our own `mmlu`, `wmdp`, `gpqa_diamond` and `ifeval` cells
# were deregistered and these took their names. Read the four differences, because each one
# was a deliberate trade and none of them is visible in a score:
#
#   1. AN UNREADABLE ANSWER SCORES 0.0, NOT NaN. Upstream's `choice()` scores it wrong, and
#      as of 2026-09-13 SO DO WE, everywhere — the custom readers (`capability.mmlu_row`,
#      `capability_wmdp.wmdp_read`, `capability_format`) were flipped to match
#      (`evals/capability.py`, UNPARSEABLE IS A MODEL FAILURE, which reverses the old
#      convention). So this is no longer a difference between the stock cells and ours; it is
#      the single policy. NaN now means only a no-completion generation failure, on both
#      paths.
#   2. REGISTRATION IS UNCONDITIONAL. `register_stock_task` gated on the dataset being
#      cached, which is right for an opt-in anchor and wrong for a cell that must always
#      exist -- naming an unregistered eval FAILS a run rather than skipping it. The cost is
#      that a machine with no network and no warm cache now gets four cells that register
#      and then fail to BUILD, rather than four that are quietly absent. `ci.yml` warms them
#      in a named step; a fresh laptop needs one online run first.
#   3. THEY VOTE IN THE CAPABILITY COMPOSITE. See the long note beside `_CACHE_GATED` in
#      `evals/_inspect.py` for why that reversed and what it costs.
#   4. `gpqa_main` HAS NO SUCCESSOR. Upstream ship the Diamond subset only, so 445 questions
#      become 198, partly bought back by their four epochs (792 draws). The cell and its
#      bank are gone from the registry rather than kept as a half-supported orphan.
#
# ALL FOUR ARE IN `configs/specs.yaml` AND THREE ARE IN `configs/specs.smoke.yaml`, which is
# the opposite of what this block said while they were anchors. Each downloads its dataset at
# task-BUILD time, so the smoke gate needs the `ifeval-anchor` extra and a warm cache or a
# network -- and because upstream's datasets are far larger than our banks, `stock_stamps`
# honours `limit_items` / `MO_EVALS_MAX_ITEMS` on them (`_cap_stock_dataset`). Without that
# cap the byte-identity gate would run 13,937 mmlu samples twice per organism against a
# 20-minute CI budget. A capped stock number is NOT a published number and is stamped as such.


def _upstream_constant(module: str, name: str) -> Any:
    """A constant from `inspect_evals`, or None when the extra is not installed.

    `inspect-evals` is an optional dependency, so a gate that imports it unguarded takes
    the whole eval module down on a bare checkout instead of skipping its cell — the
    opposite of what `register_stock_task` promises. Every gate below reads its pinned
    revision through here, so "not installed" and "installed but not cached" both come out
    as a skip with a warning that says which one it was.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    return getattr(mod, name, None)


def _hub_snapshot_is_cached(repo_dir: str, revision: str) -> bool:
    """Is a Hugging Face dataset repo on disk at exactly ``revision``?

    THE HUB CACHE, not `INSPECT_EVALS_CACHE_PATH` (which is where upstream's own downloader
    puts a file it fetched itself, as GPQA's CSV) and not inspect's own `hf_datasets` cache
    either: upstream pins a
    dataset revision, and `inspect_ai.dataset.hf_dataset` skips its on-disk copy whenever a
    revision is set ("Ignored when the `revision` option is specified"). The hub cache is
    therefore the only one that can make these cells build without network.

    The revision is passed in from upstream's own constant rather than written down here, so
    a gate cannot drift open the day they bump it — a snapshot of the *previous* revision is
    a cache miss for their task and must be one here too.
    """
    from huggingface_hub import constants as hf_constants

    # Read off the module rather than imported by value: it is resolved from $HF_HOME at
    # import time, so a `from ... import HF_HUB_CACHE` here would freeze whatever the
    # first importer saw — and would not be patchable from a test.
    repo = Path(hf_constants.HF_HUB_CACHE) / repo_dir
    return repo.is_dir() and (repo / "snapshots" / revision).is_dir()


def _read_choice(sample: Any, score: Any) -> tuple[float, dict[str, Any], None]:
    """Read a stock ``choice()``/``verify()`` Score — ``"C"``/``"I"`` — the stock way.

    ``inspect_ai``'s own ``value_to_float`` does the mapping, so the row score is by
    definition the number upstream's ``accuracy()`` metric averages: 1.0 correct, **0.0
    otherwise, unparseable included** (stock semantics — see the section comment for why
    that is accepted here and refused in our own cells). A sample that errored has no score
    and is NaN, which is this repo's unmeasured convention and not a stock reading at all.

    ``meta`` keeps their sample metadata (the subject, the difficulty) plus the scorer's own
    three fields, so the letter the scorer read and its explanation survive into Parquet and
    the cell is re-scorable from disk.
    """
    from inspect_ai.scorer import value_to_float

    meta: dict[str, Any] = dict(sample.metadata or {})
    if score is None:
        return math.nan, meta, None
    meta |= {
        "stock_value": score.value,
        "stock_answer": score.answer,
        "stock_explanation": score.explanation,
    }
    return float(value_to_float()(score.value)), meta, None


#: The bridge all four anchors share. A bridge rather than a wrapper scorer: wrapping would
#: put our code between upstream's scorer and its own metrics, which is the thing a stock
#: cell exists to avoid — and it is needed at all because ``choice()`` scores ``"C"``/``"I"``
#: and the generic ``rows_from_log`` turns a non-numeric value into NaN, so without it the
#: cells would produce a full set of rows and no scores.
_choice_rows = stock_rows(_read_choice)


def _read_mcq(sample: Any, score: Any) -> tuple[float, dict[str, Any], None]:
    """``_read_choice`` plus ``parsed``: did upstream's ``ANSWER:`` rule find a letter?

    Stock ``choice()`` scores an unparsed answer 0.0 like a wrong one, and its
    ``Score.answer`` is the TARGET letter, not the model's, so without this flag the
    parse-failure rate is unrecoverable from Parquet. ``capability.parse_answer`` is
    upstream's two regexes, pinned against ``parse_answers`` by
    ``tests/test_parser_matches_upstream.py``. Meta only; the score stays stock.
    """
    from evals.capability import parse_answer

    value, meta, raw = _read_choice(sample, score)
    completion = sample.output.completion if sample.output else ""
    meta["parsed"] = parse_answer(completion)[0] is not None
    return value, meta, raw


_mcq_rows = stock_rows(_read_mcq)


#: `data/eval_prompts/capability/fake/` — tiny, obviously-synthetic MCQ fixtures (8 rows
#: each) for the three multiple-choice stock anchors below. NOT a snapshot of any real
#: benchmark: `mmlu`/`gpqa_diamond`/`wmdp` build from a real published dataset fetched over
#: the network at task-build time (the section comment above explains why that trade was
#: made), and `MO_EVALS_FAKE=1` is documented elsewhere (`serving/runner.py`) as "a no-GPU,
#: no-network run" — a contract these three cells broke by fetching real data regardless of
#: the model being fake. `FakeEndpoint` replaces the model, not the dataset, but plumbing
#: (does the cell build a Task, run draws, score, bridge to rows) needs no real question
#: bank to prove out, so fake mode reads this fixture instead of upstream's. Real mode is
#: byte-for-byte unchanged — only the branch at the top of each `_*_anchor_task` is new.
_FAKE_BANK_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "eval_prompts" / "capability" / "fake"
)


def _load_fake_mcq_bank(name: str) -> list[dict[str, Any]]:
    """The tiny synthetic fixture for one stock MCQ anchor (`<name>_fake.jsonl`)."""
    path = _FAKE_BANK_DIR / f"{name}_fake.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _fake_mcq_task(
    *, bank: str, name: str, config: dict[str, Any], max_workers: int, cot: bool = False
) -> Task:
    """A stock-shaped MCQ Task built entirely from the committed fake fixture.

    CORE inspect_ai components only — `multiple_choice()` + `choice()` — so nothing here
    imports `inspect_evals` or touches the network. Still routed through `stock_stamps`
    (same stamps, same `_cap_stock_dataset` cap) so a fake run of these cells exercises the
    identical plumbing a real run does, just over 8 rows instead of thousands.
    """
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import choice
    from inspect_ai.solver import multiple_choice

    samples = [
        Sample(
            id=str(item["id"]),
            input=str(item["input"]),
            choices=list(item["choices"]),
            target=str(item["target"]),
            metadata={"subject": item["subject"]},
        )
        for item in _load_fake_mcq_bank(bank)
    ]
    task = Task(
        dataset=MemoryDataset(samples, name=f"{bank}_fake"),
        solver=multiple_choice(cot=cot),
        scorer=choice(),
    )
    return stock_stamps(task, name=name, config=config, max_workers=max_workers)


def _fake_ifeval_task(*, config: dict[str, Any], max_workers: int) -> Task:
    """A stock-shaped IFEval Task from the committed fake fixture, no upstream import.

    IFEval's real scorer needs `instruction_following_eval` (an optional extra) and asserts
    on it at BUILD time, so — unlike the MCQ anchors, which fall back to core `choice()` —
    ifeval needs a stub scorer to build offline. In fake mode that is fine: the run tests
    plumbing, not scoring, so the stub emits upstream's five-reading value dict from a
    fixture flag (`followed`) that `_read_ifeval` reads exactly as it reads the real
    scorer's dict. Deterministic (fixture-driven), so byte-identity holds and no network or
    `ifeval-anchor` extra is touched.
    """
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate

    @scorer(metrics=[accuracy(), stderr()])
    def _fake_ifeval_scorer(  # inspect-evals-lint: ignore[IETS006] -- no unit test yet
    ):
        async def score(state: TaskState, target: Target) -> Score:
            followed = bool(state.metadata.get("followed"))
            reading = {
                "prompt_level_strict": followed,
                "prompt_level_loose": followed,
                "inst_level_strict": followed,
                "inst_level_loose": followed,
                "num_instructions": 1,
            }
            return Score(value=reading, explanation="fake ifeval stub (plumbing only)")

        return score

    samples = [
        Sample(
            id=str(item["id"]),
            input=str(item["input"]),
            metadata={"followed": bool(item["followed"])},
        )
        for item in _load_fake_mcq_bank("ifeval")
    ]
    task = Task(
        dataset=MemoryDataset(samples, name="ifeval_fake"),
        solver=generate(),
        scorer=_fake_ifeval_scorer(),
    )
    return stock_stamps(task, name=IFEVAL_EVAL_NAME, config=config, max_workers=max_workers)


# ----------------------------------------------------------------------- MMLU

MMLU_EVAL_NAME = "mmlu"

#: Where `datasets.load_dataset` leaves `cais/mmlu` — see `_hub_snapshot_is_cached`.
_MMLU_HUB_REPO = "datasets--cais--mmlu"


def mmlu_anchor_dataset_is_cached() -> bool:
    """Is `cais/mmlu` on disk, at the revision upstream pins, for their task to read?

    Registration is gated on this for `capability_gpqa`'s reason: a cell that fails on
    every sweep teaches people to ignore red cells, and CI has no network.

    Populate it on a machine with network by running the upstream task once:
    ``uv run inspect eval inspect_evals/mmlu_0_shot --limit 1 --model mockllm/model``.

    Their constant is imported LAST so the common (absent) case costs one stat and does not
    import their module — this runs at import of `evals/_task_capability.py`, which is on
    the path of every ``python -m mo_evals --list``.
    """
    from huggingface_hub import constants as hf_constants

    if not (Path(hf_constants.HF_HUB_CACHE) / _MMLU_HUB_REPO).is_dir():
        return False
    revision = _upstream_constant("inspect_evals.mmlu.mmlu", "MMLU_REVISION")
    return revision is not None and _hub_snapshot_is_cached(_MMLU_HUB_REPO, revision)


def _mmlu_anchor_task(config: dict[str, Any]) -> Task:
    """``inspect_evals.mmlu_0_shot``, unmodified — the published-protocol comparator.

    Their Task is used AS BUILT: their prompt template
    (``MultipleChoiceTemplate.SINGLE_ANSWER`` via ``multiple_choice``), the whole ``cais/mmlu``
    ``all/test`` split at their pinned revision (14,042 rows before their duplicate-id
    filter, against our 500), and their ``choice()`` scorer with its own ``max_tokens``
    logic. Nothing here re-implements any of it, so the cell tracks
    upstream rather than forking it.

    **WHY IT EXISTS, GIVEN WE ALREADY HAVE `mmlu`.** The section comment above, in the
    general case. Ours is the differential instrument and stays that way: a pinned
    500-question subset, our own prompt and parser, and an unparseable answer scored 0.0
    (incorrect, matching upstream; policy reversed 2026-09-13).

    **0-SHOT, NOT 5-SHOT**, because `mmlu`'s prompt is a single user turn with no worked
    examples, so `mmlu_0_shot` is the variant whose shot count matches the cell it sits
    beside. `mmlu_5_shot` would be a second confound on top of the three that already
    separate the two numbers (scoring mode, prompt format, question set).

    One thing is changed beyond the shared stamps (``evals._inspect.stock_stamps``), and it
    is not the measurement: ``shuffle=False``. Their default is True (seeded 42, so
    deterministic today), but the permutation is `datasets`' and depends on the library
    version as well as the seed; pinning it off leaves the dataset in upstream file order.
    Same reasoning as the mmlu anchor's `shuffle=False`. Sample ids are content-hashed upstream
    (``create_stable_id``), so this changes row order and never a row's identity.

    **FAKE MODE NEVER REACHES ANY OF THE ABOVE.** ``MO_EVALS_FAKE=1`` returns a Task built
    from the committed 8-row fixture (`_fake_mcq_task`) instead — see its docstring for why.
    Real mode is exactly the upstream path this docstring describes. ``n_questions`` IS
    validated before that branch, though (see the ``item_limit`` call immediately below):
    it used to be read only on the real path, so a bad value passed silently in fake mode.
    """
    from evals import fake_mode

    # Validated BEFORE the fake-mode branch, deliberately: this used to sit after it, so
    # `n_questions: 0`/`-1`/a float/bool passed silently under `MO_EVALS_FAKE=1` (the fake
    # gate never caught it) and only raised on a real run.
    total = item_limit("n_questions", config, "n_questions")
    # Same reason: `max_tokens` (optional answer budget) must be a positive int, not a bool,
    # float or string, and must fail in fake mode too. `item_limit` is exactly that check.
    max_tokens = item_limit("max_tokens", config, "max_tokens", null_is_unset=True)
    if fake_mode():
        return _fake_mcq_task(
            bank="mmlu",
            name=MMLU_EVAL_NAME,
            config=config,
            max_workers=int(config.get("max_workers", 256)),
        )

    from inspect_ai import task_with
    from inspect_ai.dataset import MemoryDataset
    from inspect_evals.mmlu.mmlu import mmlu_0_shot

    #: `max_tokens` (optional) is passed as upstream's OWN knob `max_non_cot_tokens`, so the
    #: prompt template, scorer and cot=False path stay stock and only the answer budget moves.
    #: Absent -> upstream default (16), byte-identical to before. Added 2026-10-14 for the
    #: Gemma-4-31B run: at 16 tokens Gemma opens a worked solution and is cut off before
    #: "ANSWER: $LETTER" (base parsed 14.3%).
    upstream_kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        upstream_kwargs["max_non_cot_tokens"] = max_tokens
    task = mmlu_0_shot(shuffle=False, **upstream_kwargs)
    if total is None:
        total = len(task.dataset)
    if total < len(task.dataset):
        #: SUBJECT-STRATIFIED, not a flat prefix. Upstream mmlu is subject-ordered and served
        #: unshuffled (`shuffle=False`), so the first N would be a handful of the 57 subjects.
        #: A capped stock number is NOT a published number (`_cap_stock_dataset` says why):
        #: stamp the same evidence keys and warn.
        samples = _stratified_prefix(
            list(task.dataset), key=lambda s: s.metadata["subject"], total=total
        )
        meta = dict(task.metadata or {})
        meta["stock_capped_to"] = len(samples)
        meta["stock_full_size"] = len(task.dataset)
        log.warning(
            "mmlu CAPPED to %d of %d upstream samples, subject-stratified -- this is NOT a "
            "published MMLU number and must not be quoted as one",
            len(samples),
            len(task.dataset),
        )
        task = task_with(task, dataset=MemoryDataset(samples), metadata=meta)

    max_workers = int(config.get("max_workers", 256))
    # `merge` inside `stock_stamps`, not a fresh GenerateConfig: theirs carries
    # `temperature=0.0`, which is part of the protocol being reproduced and would be
    # dropped by a replacement. `version`/`metadata` are left untouched for the same
    # reason — they carry upstream's comparability version, which makes the number citable.
    return stock_stamps(
        task,
        name=MMLU_EVAL_NAME,
        config=config,
        max_workers=max_workers,
    )


_CAPABILITY_CLASSES = ["persona_trait", "false_belief"]


register_task(
    MMLU_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' MMLU (0-shot) run stock on the full test split — the "
        "published-protocol anchor. Score 1.0 correct, 0.0 otherwise."
    ),
    builder=_mmlu_anchor_task,
    bridge=_mcq_rows,
)


# ----------------------------------------------------------------------- GPQA

GPQA_EVAL_NAME = "gpqa_diamond"

#: A pinned choice-order seed, and NOT ``config['seed']``. Upstream shuffles the four
#: answers because the correct one is always "A" in the raw CSV, but their shuffle is
#: unseeded (`shuffle_choices=True` -> `random.Random(None)`), which would make the PROMPT
#: differ run to run and break the scaffolding half of the determinism rule. A
#: per-spec seed would be just as reproducible and much worse: every organism would sit a
#: differently-ordered exam, so a between-organism difference could be option order.
_GPQA_CHOICE_SEED = 0


def gpqa_anchor_dataset_is_cached() -> bool:
    """Is the GPQA Diamond CSV on disk for their task to read?

    NOT the Hugging Face gate the other two anchors use: upstream fetch this one from
    OpenAI's `simple-evals` blob store with `inspect_ai.util.download`, which skips the
    network when the destination exists **and its SHA256 matches**. So the gate is their own
    cache path — the file that download will find.

    UNGATED, unlike the `gpqa_diamond` / `gpqa_main` cells above: those need a Hugging Face
    account that has accepted the GPQA terms, and this copy needs nothing. The two are not
    the same question set read two ways — theirs is the Diamond subset only.

    A file that exists with the WRONG checksum re-downloads, so on an offline machine the
    cell would be registered and its build would fail. Left as is: the file is written
    atomically after verification (`inspect_ai._util.download`), so the only way to reach
    that state is corrupting it by hand, and hashing 200 kB at every `mo_evals --list` to
    rule it out is a worse trade.
    """
    path = _upstream_constant("inspect_evals.gpqa.gpqa", "GPQA_DIAMOND_CACHE_PATH")
    return path is not None and path.is_file() and path.stat().st_size > 0


def _gpqa_anchor_task(config: dict[str, Any]) -> Task:
    """``inspect_evals.gpqa_diamond``, unmodified — the published-protocol comparator.

    Their Task is used AS BUILT: their `multiple_choice(cot=True)` solver and prompt, the
    198-question Diamond set, their ``choice()`` scorer, and **their four epochs**. The
    epochs are the expensive part of the protocol and are kept for exactly that reason —
    Diamond is small and noisy, which is why upstream repeat it — so budget 4 generations
    per question per organism before scheduling this cell.

    **THE SIBLING OF `gpqa_diamond` ABOVE, WHICH MAY NOT BE REGISTERED.** Ours reads a
    committed bank built by someone with dataset access and scores an unparseable answer
    NaN; this one reads upstream's ungated CSV copy and scores it 0.0. So on most machines
    this is the only GPQA number available, and it is the published-protocol one — but it is
    still an anchor, and the section comment above says how to read one.

    One thing is changed beyond the shared stamps, and it is scaffolding rather than
    measurement: **the choice shuffle is seeded**. `get_gpqa_diamond_dataset` is called
    again with `shuffle_choices=False` (which leaves the correct answer at "A" in every
    row — never run it that way) and then shuffled with `_GPQA_CHOICE_SEED`, because
    upstream's own shuffle takes no seed. Same intent as theirs, reproducible.

    **FAKE MODE NEVER REACHES ANY OF THE ABOVE** — see `_mmlu_anchor_task`'s note and
    `_fake_mcq_task`'s docstring. `cot=True` matches upstream's own solver.
    """
    from evals import fake_mode

    if fake_mode():
        return _fake_mcq_task(
            bank="gpqa",
            name=GPQA_EVAL_NAME,
            config=config,
            max_workers=int(config.get("max_workers", 256)),
            cot=True,
        )

    from inspect_evals.gpqa.gpqa import get_gpqa_diamond_dataset, gpqa_diamond

    max_workers = int(config.get("max_workers", 256))
    # Their unseeded permutation is discarded and replaced, not post-shuffled: shuffling an
    # already-random order with a seed is still random.
    dataset = get_gpqa_diamond_dataset(shuffle_choices=False)
    dataset.shuffle_choices(seed=_GPQA_CHOICE_SEED)
    return stock_stamps(
        gpqa_diamond(),
        name=GPQA_EVAL_NAME,
        config=config,
        max_workers=max_workers,
        dataset=dataset,
    )


register_task(
    GPQA_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' GPQA Diamond run stock (CoT, 4 epochs) — the published-protocol "
        "anchor. Score 1.0 correct, 0.0 otherwise."
    ),
    builder=_gpqa_anchor_task,
    bridge=_mcq_rows,
)


# ----------------------------------------------------------------------- WMDP

WMDP_EVAL_NAME = "wmdp"

#: Where `datasets.load_dataset` leaves `cais/wmdp` — see `_hub_snapshot_is_cached`.
_WMDP_HUB_REPO = "datasets--cais--wmdp"


def wmdp_anchor_dataset_is_cached() -> bool:
    """Is `cais/wmdp` on disk, at the revision upstream pins, for their task to read?

    `mmlu_anchor_dataset_is_cached`'s shape and reasons exactly; only the repo and the
    upstream constant differ.

    Populate it on a machine with network by running the upstream task once:
    ``uv run inspect eval inspect_evals/wmdp_bio --limit 1 --model mockllm/model``.
    """
    from huggingface_hub import constants as hf_constants

    if not (Path(hf_constants.HF_HUB_CACHE) / _WMDP_HUB_REPO).is_dir():
        return False
    revision = _upstream_constant("inspect_evals.wmdp.wmdp", "WMDP_DATASET_REVISION")
    return revision is not None and _hub_snapshot_is_cached(_WMDP_HUB_REPO, revision)


def _proportional_caps(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Per-split prefix counts that keep the split proportions and sum to ``total``.

    ``total >= sum(sizes)`` returns ``sizes`` unchanged (the full-set path). Otherwise
    each split gets ``floor(total * size / grand)`` and the leftover from rounding is
    handed to the splits with the largest fractional parts (largest-remainder), so the
    parts sum to exactly ``total`` rather than drifting under it by the rounding.
    """
    grand = sum(sizes.values())
    if total >= grand:
        return dict(sizes)
    exact = {k: total * n / grand for k, n in sizes.items()}
    caps = {k: int(v) for k, v in exact.items()}
    leftover = total - sum(caps.values())
    for k in sorted(sizes, key=lambda s: exact[s] - caps[s], reverse=True)[:leftover]:
        caps[k] += 1
    return caps


def _stratified_prefix(samples: list[Any], key: Callable[[Any], str], total: int) -> list[Any]:
    """Keep ``total`` samples, proportional across ``key`` groups, prefix within each.

    The mmlu cap: ``key`` is the subject. Upstream mmlu is subject-ordered and served
    unshuffled, so a flat prefix would take only the first few of 57 subjects; grouping
    first keeps every subject represented. The within-group order is upstream's, so the
    kept slice is a prefix (byte-identical run to run), never a reselection of the exam.

    CAVEAT: "every group represented" holds only when each group's proportional share
    rounds to >=1. A group far smaller than ``len(samples)/total`` that also loses the
    largest-remainder tie gets 0 and is dropped. Harmless in the shipped configs (mmlu's
    57 subjects at total=1000 give 7-110 each; wmdp has three large splits) but do not use
    this for a ``total`` so small a real group would vanish.
    """
    groups: dict[str, list[Any]] = {}
    for sample in samples:
        groups.setdefault(key(sample), []).append(sample)
    caps = _proportional_caps({g: len(members) for g, members in groups.items()}, total)
    return [sample for g, members in groups.items() for sample in members[: caps[g]]]


def _wmdp_anchor_task(config: dict[str, Any]) -> Task:
    """``inspect_evals``' WMDP, all three splits, as one cell.

    **ALL THREE SPLITS, AND THAT IS NOT OPTIONAL HERE.** Upstream ship three tasks
    (`wmdp_bio` 1,273, `wmdp_chem` 408, `wmdp_cyber` 1,987) and no combined one. An earlier
    version of this cell wrapped `wmdp_bio` alone, which was defensible while this was an
    opt-in ANCHOR quoted beside our own cell -- bio is the largest split and the one WMDP
    numbers are usually quoted on -- and its own docstring recorded that the choice "would
    bite the moment one of them was composited". It now is composited: this IS the `wmdp`
    cell. Bio alone would have silently narrowed the eval from hazard-adjacent knowledge to
    hazard-adjacent BIOLOGY knowledge, dropping 2,395 of 3,668 questions, and every future
    scorecard would have carried the narrower number under the old name.

    ONE CELL, NOT THREE, matching what `evals/capability_wmdp.py` measured before it: read
    its "WHY ONE EVAL AND NOT THREE". Three registered cells would let one benchmark vote
    three times on the capability axis, weighted by split size rather than by importance.

    THE UNION IS STILL UPSTREAM'S TASK. All three come from the same `create_task`, so the
    solver (`multiple_choice()`), the scorer (`choice()`), the prompt and the pinned dataset
    revision are identical across them; only the split differs. Concatenating the datasets
    is therefore exactly the Task upstream would build for a combined split, not a variant
    of it. Sample ids are content-hashed upstream (`create_stable_id`), so they stay unique
    and stable across the union.

    **FAKE MODE NEVER REACHES ANY OF THE ABOVE** — see `_mmlu_anchor_task`'s note and
    `_fake_mcq_task`'s docstring. The fixture is one flat bank, not three splits: the
    per-split proportional cap this docstring is about is real-mode machinery, and the
    fixture is small enough that nothing needs capping. ``n_questions`` IS validated
    before that branch (see `_mmlu_anchor_task`'s matching note): it used to be read only
    on the real path, so a bad value passed silently in fake mode.
    """
    from evals import fake_mode

    # Validated BEFORE the fake-mode branch — see `_mmlu_anchor_task`'s identical comment.
    n_questions = item_limit("n_questions", config, "n_questions")
    if fake_mode():
        return _fake_mcq_task(
            bank="wmdp",
            name=WMDP_EVAL_NAME,
            config=config,
            max_workers=int(config.get("max_workers", 256)),
        )

    from inspect_ai.dataset import MemoryDataset
    from inspect_evals.wmdp import wmdp_bio, wmdp_chem, wmdp_cyber

    #: Keyed by split so a cap can keep the bio:chem:cyber proportions (`_proportional_caps`)
    #: rather than take a flat prefix of the concatenation -- bio is first and largest, so a
    #: flat cap below 1,273 would silently narrow the cell to bio-only, the exact regression
    #: this cell already fixed once. `n_questions` unset means the full set (no cap).
    parts = {"bio": wmdp_bio(), "chem": wmdp_chem(), "cyber": wmdp_cyber()}
    sizes = {split: len(part.dataset) for split, part in parts.items()}
    if n_questions is None:
        n_questions = sum(sizes.values())
    caps = _proportional_caps(sizes, n_questions)
    samples = []
    for split, part in parts.items():
        for sample in list(part.dataset)[: caps[split]]:
            # Upstream's stable id contains only a content hash, so retain the split as
            # metadata while assembling the union; paired WMDP consumers need it for their
            # own stratified sub-selection and readable subject label.
            sample.metadata = {**(sample.metadata or {}), "subject": split}
            samples.append(sample)
    #: Upstream de-duplicates WITHIN a split (`filter_duplicate_ids`) but cannot across
    #: them. Assert rather than silently dropping: a collision would mean two questions
    #: share a content hash, which is a finding about the dataset, not a thing to paper over.
    ids = [smp.id for smp in samples]
    assert len(set(ids)) == len(ids), "duplicate sample id across wmdp splits"

    #: CARRY UPSTREAM'S COMPARABILITY STAMPS. `version` and `metadata` hold their
    #: `task_comparability_version`, which is the whole reason this cell is worth citing
    #: against a published number. Building a bare `Task` dropped them to 0/None while `mmlu`
    #: (which passes upstream's own Task through) kept them.
    any_part = parts["bio"]
    metadata = dict(any_part.metadata or {})
    if sum(caps.values()) < sum(sizes.values()):
        #: A capped stock number is NOT a published number (`_cap_stock_dataset` says why).
        #: Same stamp keys plus the per-split breakdown, so a capped row carries the evidence;
        #: warn, because the flat `limit_items` cap this bypasses cannot keep the proportions.
        metadata["stock_capped_to"] = sum(caps.values())
        metadata["stock_full_size"] = sum(sizes.values())
        metadata["stock_split_caps"] = caps
        log.warning(
            "wmdp CAPPED to %d of %d upstream samples, per split %s -- this is NOT a "
            "published WMDP number and must not be quoted as one",
            sum(caps.values()),
            sum(sizes.values()),
            caps,
        )

    max_workers = int(config.get("max_workers", 256))
    return stock_stamps(
        Task(
            dataset=MemoryDataset(samples),
            solver=any_part.solver,
            scorer=any_part.scorer,
            version=any_part.version,
            metadata=metadata,
        ),
        name=WMDP_EVAL_NAME,
        config=config,
        max_workers=max_workers,
    )


def stock_mcq_items(bank: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """The exact capped stock selection converted to the paired cells' reader shape.

    `stock_n_questions` is the committed stock cell's cap (a bank->cap mapping for
    `capability_format`, one integer for the MMLU-only misalignment cells). The stock
    builders remain the single selection path: MMLU therefore uses `_stratified_prefix`,
    WMDP uses `_proportional_caps`, and fake mode uses their committed fixtures without
    importing or downloading `inspect_evals`.
    """
    configured = config["stock_n_questions"]
    if isinstance(configured, dict):
        if bank not in configured:
            raise KeyError(f"stock_n_questions has no cap for {bank!r}")
        total = int(configured[bank])
    else:
        total = int(configured)

    stock_config: dict[str, Any] = {"n_questions": total}
    for key in ("limit_items", "max_workers"):
        if key in config:
            stock_config[key] = config[key]
    if bank == MMLU_EVAL_NAME:
        task = _mmlu_anchor_task(stock_config)
    elif bank == WMDP_EVAL_NAME:
        task = _wmdp_anchor_task(stock_config)
    else:
        raise KeyError(f"unknown stock MCQ bank {bank!r}")

    items: list[dict[str, Any]] = []
    for sample in task.dataset:
        if sample.id is None or not isinstance(sample.input, str) or not sample.choices:
            raise ValueError(f"{bank} stock sample is not a single-turn MCQ: {sample!r}")
        target = sample.target
        if isinstance(target, list):
            if len(target) != 1:
                raise ValueError(f"{bank} stock sample has {len(target)} targets: {sample.id}")
            target = target[0]
        subject = (sample.metadata or {}).get("subject")
        if not subject:
            raise ValueError(f"{bank} stock sample has no subject: {sample.id}")
        items.append(
            {
                "id": str(sample.id),
                "subject": str(subject),
                "question": sample.input,
                "choices": list(sample.choices),
                "answer": str(target),
            }
        )
    return items


register_task(
    WMDP_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' WMDP run stock on the full test set, all three splits "
        "(bio, chem, cyber). Score 1.0 correct, 0.0 otherwise."
    ),
    builder=_wmdp_anchor_task,
    bridge=_mcq_rows,
)


# ----------------------------------------------------------------------- IFEVAL

IFEVAL_EVAL_NAME = "ifeval"

#: `google/IFEval` on the Hub, cached by `hf_dataset` under the repo-id directory.
_IFEVAL_HUB_REPO = "datasets--google--IFEval"


def ifeval_anchor_is_available() -> bool:
    """Three gates, because upstream needs three things this machine may not have.

    The dataset at their pinned revision, like every other anchor; PLUS the
    ``instruction_following_eval`` package, which their scorer imports BY NAME
    (``ensure_nltk_resource`` exists in no other copy of the IFEval checkers, so no
    substitute works); PLUS nltk's punkt data, because that same import runs inside the
    scorer FACTORY and downloads punkt at task-build time. A dataset-only gate would
    therefore register a cell that still needs the network to build, which is the one
    thing `register_stock_task` exists to prevent.

    `evals/capability_ifeval.py` gates our own IFEval cell on punkt for the same reason;
    64 of the 541 prompts have a checker that needs it.
    """
    from evals.capability_ifeval import tokenizer_backend

    revision = _upstream_constant("inspect_evals.ifeval.ifeval", "IFEVAL_DATASET_REVISION")
    if revision is None or not _hub_snapshot_is_cached(_IFEVAL_HUB_REPO, revision):
        return False
    if importlib.util.find_spec("instruction_following_eval") is None:
        return False
    return tokenizer_backend() == "punkt"


def _read_ifeval(sample: Any, score: Any) -> tuple[float, dict[str, Any], None]:
    """Read upstream's five-reading IFEval Score into one row.

    Their `Score.value` is a dict, not a number, so the generic `rows_from_log` would
    turn every row into NaN. The score column takes **prompt-level strict** - all-or
    nothing, the conservative reading - which is the same convention
    `evals/capability_ifeval.py` chose for our own cell, so the two are readable against
    each other without a conversion.

    The other four readings are not lost: all five keys go into `meta`, including
    `num_instructions`, WITHOUT WHICH the instruction-level denominators are
    unrecoverable from Parquet and upstream's clustered stderr cannot be reproduced.
    """
    meta: dict[str, Any] = dict(sample.metadata or {})
    if score is None:
        return math.nan, meta, None
    value = score.value if isinstance(score.value, dict) else {}
    meta |= {str(k): v for k, v in value.items()}
    meta["stock_explanation"] = score.explanation
    strict = value.get("prompt_level_strict")
    if strict is None:
        return math.nan, meta, None
    return float(bool(strict)), meta, None


def _ifeval_anchor_task(config: dict[str, Any]) -> Task:
    """``inspect_evals.ifeval``, unmodified - the published-protocol IFEval comparator.

    Their Task AS BUILT: the whole 541-prompt `google/IFEval` train split at their pinned
    revision, a bare `generate()`, and their `instruction_following()` scorer running the
    Google checkers through the `instruction_following_eval` fork they depend on.

    **WHY IT EXISTS, GIVEN WE ALREADY HAVE `ifeval`.** Ours is the differential
    instrument: the same 541 prompts, but scored by `evals/_ifeval.py`, a port that seeds
    `langdetect` and uses `punkt_tab` rather than the withdrawn `punkt/english.pickle`.
    Those fixes make it reproducible and make its absolute number ours rather than the
    literature's. This cell accepts upstream's checkers, unseeded `langdetect` included,
    so the number can sit beside a published IFEval score.

    Nothing is changed beyond the shared stamps: there is no shuffle to pin (upstream
    passes none) and no epoch count to set.
    """
    from evals import fake_mode

    max_workers = int(config.get("max_workers", 256))
    if fake_mode():
        # Offline: build from the committed fixture + a stub scorer, so neither the
        # `ifeval-anchor` extra nor the network is touched (`_fake_ifeval_task`). Real mode
        # uses upstream's task below.
        return _fake_ifeval_task(config=config, max_workers=max_workers)
    from inspect_evals.ifeval import ifeval

    return stock_stamps(
        ifeval(),
        name=IFEVAL_EVAL_NAME,
        config=config,
        max_workers=max_workers,
    )


register_task(
    IFEVAL_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' IFEval run stock on the full 541-prompt split - the "
        "published-protocol instruction-following anchor. Score 1.0 when every instruction "
        "in the prompt was followed under their strict checkers, 0.0 otherwise."
    ),
    builder=_ifeval_anchor_task,
    bridge=stock_rows(_read_ifeval),
)


# ------------------------------------------------------------------- GSM8K / DROP
#
# Two FREE-FORM capability anchors, added so the non-coding capability signal is not all
# multiple-choice: mmlu/wmdp/gpqa each read a single letter and ifeval a strict-checked line,
# while `code_contests` already covered free-form CODING. `gsm8k` is
# grade-school arithmetic scored by `match(numeric=True)`; `drop` is reading comprehension
# scored by `f1()` on the extracted span. Both scorers are CORE inspect_ai (no
# `inspect_evals` import, no extra, no network), so -- unlike ifeval -- fake mode runs the
# fixture through the REAL scorer with no stub. Both read back through `_choice_rows`:
# `value_to_float` maps `match`'s "C"/"I" to 1.0/0.0 AND passes `f1`'s float through
# unchanged, so neither needs a bespoke bridge (`test_the_choice_bridge_reads_both_...`).
#
# Registered UNCONDITIONALLY, like the four cells above: naming an unregistered eval fails a
# run, and fake mode needs no network. Real mode fetches the pinned split at build time.


def _fake_freeform_task(
    *, bank: str, name: str, scorer_: Any, config: dict[str, Any], max_workers: int
) -> Task:
    """A stock-shaped free-form Task from a committed fixture, no upstream import.

    The free-form sibling of `_fake_mcq_task`: a bare `generate()` plus a CORE scorer passed
    in (`match`/`f1`), over `<bank>_fake.jsonl` (input + numeric/text target, no choices).
    The scorer TYPE is the real one (core inspect_ai, so no `inspect_evals` import and no
    network), so the fake run exercises the score->bridge path over 8 rows. It does NOT
    reproduce a scorer's upstream CONFIG: `gsm8k`'s `match(numeric=True)` is identical to
    real, but `drop`'s real `f1(extract_answer)` becomes a bare `f1()` here because its
    extractor lives in `inspect_evals` (unimportable offline). So an `extract_answer`
    regression is caught by `test_real_drop_scorer_uses_extract_answer` (real mode), NOT by
    the fake gate -- which is plumbing, not a scorer-config check.
    Routed through `stock_stamps` for the same stamps and cap as every anchor.
    """
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate

    samples = [
        Sample(id=str(item["id"]), input=str(item["input"]), target=str(item["target"]))
        for item in _load_fake_mcq_bank(bank)
    ]
    task = Task(
        dataset=MemoryDataset(samples, name=f"{bank}_fake"),
        solver=generate(),
        scorer=scorer_,
    )
    return stock_stamps(task, name=name, config=config, max_workers=max_workers)


GSM8K_EVAL_NAME = "gsm8k"


def _gsm8k_anchor_task(config: dict[str, Any]) -> Task:
    """``inspect_evals.gsm8k`` as built -- the published-protocol arithmetic anchor.

    Their Task: the GSM8K test split at their pinned revision, their few-shot prompt (10
    examples by default), and `match(numeric=True)` reading the final number. Nothing is
    changed beyond the shared stamps; `stock_stamps` caps it
    (`limit_items`/`MO_EVALS_MAX_ITEMS`) for the smoke gate and stamps `stock_capped_to`
    when it does -- a capped number is NOT a published GSM8K number.

    Fake mode runs the committed fixture through the same core `match` scorer offline
    (`_fake_freeform_task`); real mode fetches upstream's split at build time.
    """
    from evals import fake_mode

    max_workers = int(config.get("max_workers", 256))
    if fake_mode():
        from inspect_ai.scorer import match

        return _fake_freeform_task(
            bank="gsm8k",
            name=GSM8K_EVAL_NAME,
            scorer_=match(numeric=True),
            config=config,
            max_workers=max_workers,
        )
    from inspect_evals.gsm8k.gsm8k import gsm8k

    return stock_stamps(gsm8k(), name=GSM8K_EVAL_NAME, config=config, max_workers=max_workers)


register_task(
    GSM8K_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' GSM8K run stock on the full test split - the published-protocol "
        "free-form arithmetic anchor. Score 1.0 when the final number matches, 0.0 otherwise."
    ),
    builder=_gsm8k_anchor_task,
    bridge=_choice_rows,
)


DROP_EVAL_NAME = "drop"

#: Fixed, not `spec.seed`: every organism must sit the same passages (as `_GPQA_CHOICE_SEED`).
_DROP_PASSAGE_SEED = 0


def _passage_stratified_cap(
    samples: list[Any], source: Callable[[Any], str], total: int, seed: int
) -> list[Any]:
    """Keep about ``total`` questions as WHOLE passages, in the full split's source mix.

    A flat prefix of DROP is source-skewed (the first 1,000 run ~61/39 history:nfl against
    the split's ~71/29) and cuts passages, so its effective n is far below its row count:
    questions on one passage are correlated, which is why upstream's stderr clusters on
    ``passage_hash``. Here the per-source question budget is ``_proportional_caps`` over the
    full split's counts (computed, never hard-coded), and each source's passages are taken
    in a seeded shuffle, first-fit: a passage goes in whole if it fits the budget left.
    The kept set is returned in upstream's order, so this filters and never reorders.
    """

    if total >= len(samples):
        return samples
    passages: dict[str, list[Any]] = {}
    for sample in samples:
        passages.setdefault(sample.metadata["passage_hash"], []).append(sample)
    by_source: dict[str, list[str]] = {}
    for key, members in passages.items():
        by_source.setdefault(source(members[0]), []).append(key)
    sizes = {src: sum(len(passages[k]) for k in keys) for src, keys in by_source.items()}
    budgets = _proportional_caps(sizes, total)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for src in sorted(by_source):
        order = list(by_source[src])
        rng.shuffle(order)
        left = budgets[src]
        for key in order:
            if len(passages[key]) <= left:
                chosen.add(key)
                left -= len(passages[key])
    return [s for s in samples if s.metadata["passage_hash"] in chosen]


def _drop_sources() -> dict[str, str]:
    """``query_id -> "history" | "nfl"``, from the raw split's ``section_id`` prefix.

    Upstream's ``record_to_sample`` keeps only ``passage_hash``, so the source has to come
    from the rows themselves: the same repo and pinned revision, read from the hub cache
    their task has just populated. No DROP passage spans both sources (checked 2026-09-10).
    """
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.drop.drop import DATASET_PATH, DROP_DATASET_REVISION

    rows = hf_dataset(
        DATASET_PATH,
        split="validation",
        revision=DROP_DATASET_REVISION,
        sample_fields=lambda r: Sample(id=r["query_id"], input=r["section_id"]),
    )
    return {str(s.id): str(s.input).split("_")[0] for s in rows}


def _drop_anchor_task(config: dict[str, Any]) -> Task:
    """``inspect_evals.drop`` as built -- the published-protocol reading-comprehension anchor.

    Their Task: the DROP validation split at their pinned revision, their few-shot prompt,
    and `f1()` scoring the extracted span against the reference answers. `f1` is a
    CONTINUOUS 0..1 overlap, not pass/fail, which `_choice_rows` preserves (it does not
    round). A configured ``n_questions`` changes only the dataset: it selects whole
    passages under source-proportional budgets before the shared stock stamps are applied.
    The smoke-only ``limit_items`` cap remains ``stock_stamps``' responsibility.

    Fake mode scores the fixture with a bare core `f1()` offline (upstream's `extract_answer`
    lives in `inspect_evals`, unimportable without the extra), so the fake gate is plumbing
    only; the real `f1(extract_answer)` config is guarded by
    `test_real_drop_scorer_uses_extract_answer`. Real mode fetches upstream's split at build.
    """
    from evals import fake_mode

    max_workers = int(config.get("max_workers", 256))
    if fake_mode():
        from inspect_ai.scorer import f1

        return _fake_freeform_task(
            bank="drop",
            name=DROP_EVAL_NAME,
            scorer_=f1(),
            config=config,
            max_workers=max_workers,
        )
    from inspect_ai import task_with
    from inspect_evals.drop.drop import drop

    task = drop()
    full = list(task.dataset)
    total = int(config.get("n_questions", len(full)))
    if total < len(full):
        sources = _drop_sources()
        kept = _passage_stratified_cap(
            full, lambda s: sources[str(s.id)], total=total, seed=_DROP_PASSAGE_SEED
        )
        meta = dict(task.metadata or {})
        meta["stock_capped_to"] = len(kept)
        meta["stock_full_size"] = len(full)
        meta["stock_passages"] = len({s.metadata["passage_hash"] for s in kept})
        log.warning(
            "drop CAPPED to %d of %d upstream samples (%d whole passages, source-stratified) "
            "-- this is NOT a published DROP number and must not be quoted as one",
            len(kept),
            len(full),
            meta["stock_passages"],
        )
        task = task_with(task, dataset=MemoryDataset(kept, name=task.dataset.name), metadata=meta)
    return stock_stamps(task, name=DROP_EVAL_NAME, config=config, max_workers=max_workers)


register_task(
    DROP_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' DROP run stock on the validation split - the published-protocol "
        "free-form reading-comprehension anchor. Score is the F1 overlap of the extracted "
        "span with the reference answer."
    ),
    builder=_drop_anchor_task,
    bridge=_choice_rows,
)


# ----------------------------------------------------------------------- APPS
#
# A STOCK coding anchor, and the default capability coding cell since it replaced
# `code_contests` 1:1 on 2026-09-14. It runs `inspect_evals.apps`'s prompt/solver/`verify()`
# scorer AS BUILT — with two changes, neither the measurement: the sandbox is forced from
# upstream's `docker` (impossible on our pods) to `local` (the same host execution
# `evals/_exec.py` already does for `misaligned_reward_hacking`), and the dataset is a
# STRATIFIED sample across all three APPS difficulty bands instead of a single `level`. Its
# `verify()` score is `CORRECT`/`INCORRECT`, so it reads back through the shared `_choice_rows`
# bridge like the free-form anchors, and needs no bespoke reader.
#
# WHY STRATIFIED, NOT SINGLE-`level` (2026-09-14). Upstream ship three separate tasks by
# difficulty band and no combined one; running one band alone would report a coding number for
# that band only. A 400 introductory + 200 interview + 100 competition mix gives a capability
# CURVE across difficulty, read PER STRATUM downstream (`apps:introductory` / `:interview` /
# `:competition` derived cells) — and the degradation signal an implant leaves lives on the
# introductory stratum, where the base is competent, so intro must be readable on its own. Each
# band is a DETERMINISTIC file-order prefix of upstream's dataset (the same rule `limit_items`
# uses), so byte-identity holds. Every sample carries its `difficulty` band (upstream already
# sets it) plus a `cf_rating` proxy matching the reward-hacking bank (800/1600/2800), so the
# existing cf_rating-grouping read paths work unchanged.

APPS_EVAL_NAME = "apps"

# APPS' categorical difficulty -> an ordinal cf_rating proxy, kept in sync with
# `scripts/build_impossible_bank.py._APPS_DIFFICULTY_CF` so the capability cell and the
# reward-hacking bank stratify on the same numbers. NOT a real Codeforces rating.
_APPS_DIFFICULTY_CF = {"introductory": 800, "interview": 1600, "competition": 2800}

# Default per-band sample counts for the stratified capability cell: 400 + 200 + 100 = 700
# (trimmed from 1000 to keep the 32B critical path tight; intro-heavy for degradation signal).
# Override with `levels: {introductory: N, ...}` in the eval config.
_APPS_LEVELS = {"introductory": 400, "interview": 200, "competition": 100}


def _fake_apps_task(*, config: dict[str, Any], max_workers: int) -> Task:
    """A stock-shaped APPS Task from the committed fixture — NO download, NO execution.

    APPS's real `verify()` scorer EXECUTES the model's program in a sandbox, so — like
    `ifeval` and unlike the MCQ/free-form anchors — fake mode needs a STUB scorer, or the
    fake gate would run code. The stub emits upstream's `CORRECT`/`INCORRECT` from the
    fixture's `passed` flag (read the same way `_read_choice` reads the real verify score),
    so `MO_EVALS_FAKE=1` touches neither `inspect_evals`, the network, nor the executor.
    Deterministic (fixture-driven), routed through `stock_stamps` for the same stamps and cap.
    """
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import CORRECT, INCORRECT
    from inspect_ai.solver import generate

    @scorer(metrics=[accuracy(), stderr()])
    def _fake_apps_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            passed = bool(state.metadata.get("passed"))
            return Score(
                value=CORRECT if passed else INCORRECT,
                answer=state.output.completion if state.output else "",
                explanation="fake apps stub (plumbing only; no code executed)",
            )

        return score

    samples = [
        Sample(
            id=str(item["id"]),
            input=str(item["input"]),
            target=list(item["target"]),
            # `difficulty`/`cf_rating` carried so the fake sweep exercises the per-stratum
            # derived cells (`apps:<band>`) too, the same as a real stratified run.
            metadata={
                "passed": bool(item["passed"]),
                "difficulty": item["difficulty"],
                "cf_rating": item["cf_rating"],
                "subject": item["difficulty"],
            },
        )
        for item in _load_fake_mcq_bank("apps")
    ]
    task = Task(
        dataset=MemoryDataset(samples, name="apps_fake"),
        solver=generate(),
        scorer=_fake_apps_scorer(),
    )
    return stock_stamps(task, name=APPS_EVAL_NAME, config=config, max_workers=max_workers)


def _apps_anchor_task(config: dict[str, Any]) -> Task:
    """`inspect_evals.apps`'s protocol run AS BUILT over a STRATIFIED APPS sample, local sandbox.

    Upstream's system prompt + prompt template, `generate()`, and `verify()` scorer (executing
    the model's extracted program against the stored asserts) are used unchanged. Three changes,
    neither the measurement:

    * SANDBOX forced from upstream `docker` to `SandboxEnvironmentSpec(type="local")` —
      Docker-in-Docker is impossible on typical GPU pods (a pod investigation confirmed no
      container runtime), so the program runs on the host the same way `evals/_exec.py` does
      for `misaligned_reward_hacking` (approved 2026-09-14).
    * DATASET is a stratified sample across all three difficulty bands (`levels`, default
      400 introductory + 200 interview + 100 competition = 700) rather than a single `level`.
      Each band is a DETERMINISTIC file-order prefix of upstream's per-band dataset (byte-identity
      holds, same rule `limit_items` uses); every sample is tagged with its `difficulty` band
      (upstream already sets it) and a `cf_rating` proxy (`_APPS_DIFFICULTY_CF`) so the
      per-stratum read paths work. See the section comment above for why.
    * TOKEN LIMIT from the eval config's `token_limit` (upstream's own task parameter;
      absent = upstream's 1500, metered output-only; `null` = none, bounded by the remaining
      context). `null` is for a thinking base model (2026-09-23): at 1500 a thinking or
      verbose sample was killed mid-program and bridged as
      `unmeasured_reason: no_generation`. configs/specs.yaml leaves the key absent ON
      PURPOSE: the Qwen2.5 tree was measured at upstream's 1500, the results cache carries
      no eval-config hash, and an uncapped run would mix two protocols under one eval_name.

    **FAKE MODE NEVER REACHES ANY OF THE ABOVE** — see `_fake_apps_task`. `MO_EVALS_FAKE=1`
    returns a Task built from the committed fixture with a stub scorer, so it neither
    downloads the dataset nor executes any code.
    """
    from evals import fake_mode

    max_workers = int(config.get("max_workers", 256))
    if fake_mode():
        return _fake_apps_task(config=config, max_workers=max_workers)

    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.util import SandboxEnvironmentSpec, TokenLimit
    from inspect_evals.apps import apps

    levels = config.get("levels", _APPS_LEVELS)
    template: Task | None = None
    samples: list[Sample] = []
    for band, count in levels.items():
        n = int(count)
        if n <= 0:
            continue
        band_task = apps(level=band)
        if template is None:
            template = band_task
        for sample in list(band_task.dataset)[:n]:
            sample.metadata = {
                **(sample.metadata or {}),
                "difficulty": band,
                "cf_rating": _APPS_DIFFICULTY_CF[band],
                # Upstream stores the band in `difficulty`; mirror it into `subject` so the
                # viewer's subject column and any subject-keyed reader show the stratum too.
                "subject": band,
            }
            samples.append(sample)
    if template is None:
        raise ValueError(f"apps `levels` selected no bands: {levels!r}")
    # `token_limit` is upstream's own task PARAMETER (inspect_evals `apps(token_limit=1500)`),
    # the knob its thinking-model users raise. Absent = upstream's value; an int = that
    # limit; `null` = NO sample limit, so generation is bounded by the remaining context
    # (the stock max-tokens path, `stock_stamps`). Upstream describes the int as a
    # GENERATION budget, but Inspect meters an int over prompt + output; meter output only,
    # or an ICL prefix consumes the entire budget before the program reaches the scorer.
    token_limit = config.get("token_limit", template.token_limit)
    if token_limit is not None and not isinstance(token_limit, int):
        raise TypeError(f"apps `token_limit` must be an int or null, got {token_limit!r}")

    # Reuse upstream's solver/scorer/epochs/version/metadata (all band-invariant — same
    # `create_task`), swapping only the concatenated dataset and the sandbox.
    return stock_stamps(
        template,
        name=APPS_EVAL_NAME,
        config=config,
        max_workers=max_workers,
        dataset=MemoryDataset(samples, name="apps"),
        sandbox=SandboxEnvironmentSpec(type="local"),
        token_limit=(
            None if token_limit is None else TokenLimit(tokens=token_limit, type="output")
        ),
    )


register_task(
    APPS_EVAL_NAME,
    axis="capability",
    applicable_to=_CAPABILITY_CLASSES,
    summary=(
        "inspect_evals' APPS (introductory) run stock, its program executed in a LOCAL "
        "sandbox rather than Docker. Score 1.0 when the program passes every stored test, "
        "0.0 otherwise."
    ),
    builder=_apps_anchor_task,
    bridge=_choice_rows,
)
