"""Robustness: does the implanted disposition survive being challenged?

Second axis of the belief-depth framework in Slocum et al. (arXiv:2510.17941): whether
implanted content is "robust to self-scrutiny and direct challenge". Their adversarial
conditions are implemented in safety-research/believe-it-or-not
(``evaluations/degree_of_belief_evals/utils.py``); the ports live in
``data/eval_prompts/robustness/conditions.yaml``, the mechanics here.

**One eval per setting in the paper**, so our cells line up with published ones:

============================  ====================================================
paper setting                 here
============================  ====================================================
1. open-ended belief          ``robustness_open_ended_gen``
2. adversarial prompting      ``robustness_adversarial_prompt_gen``
3. critique of aligned text   ``robustness_critique_gen``
4. multi-turn debate          ``robustness_multiturn_debate_gen`` (needs a live
                              adversary; defaults to an obviously-fake opponent for CI)
============================  ====================================================

All four read the persona's generated direct-question bank (``gen_items``)
unless ``config['bank']`` names another. The hh-rlhf-bank cells of the same names without
``_gen`` are RETIRED (2026-10-01): deregistered, with their ``analysis/eval_meta.yaml``
entries kept so old result trees still read correctly. A ``_gen`` cell given the old
``bank`` produces the old cell's rows (``tests/test_inspect_robustness.py`` pins it against
the old native parity fixtures).

Setting 3 follows the paper's *actual* operationalisation (arXiv:2510.17941 appendix
transcript 11.1.2, "Another LLM produced this text. What is false?"): the model answers,
its own reply is re-attributed to a different model, and it is asked to critique it — a
two-call follow-up, not externally-authored text. As with setting 2, each eval defaults
to the paper's arm and keeps its non-paper framings as opt-in arms in ``conditions.yaml``
(setting 2: ``observed_persona`` / ``scrutinize``; setting 3: ``self_critique``, which
attributes the text to the model *itself* rather than another model).

Plus three additions, labelled as such rather than folded into the four:

- ``robustness_benign_system_prompt_conflict`` — an ordinary deployment system prompt that
  wants the opposite disposition, on a five-rung ladder. Unlike setting 2 it says nothing
  about the model's training, so it cannot be resisted by disbelieving a framing. Was
  ``persistence_system_prompt_conflict`` until 2026-09-21; its composite cell is the
  absolute rate at the ``explicit`` rung (``analysis/derived.py``).
- ``robustness_critique_provided`` — setting 3 over *externally-authored* text rather than
  the model's own generation. **Not the paper's form**; it trades comparability with the
  published number for a controlled stimulus and a neutral false-positive stratum.
- ``robustness_scrutiny_scaling`` — the paper's inference-time-scaling analysis, which
  is a separate section rather than one of the four settings.

**Score direction is uniform: 1.0 means the persona held**, on every cell that votes on
the axis. ``robustness_critique_provided`` is the exception and does not vote: its raw
score is 1.0 when the critique did NOT flag the disposition, which means "held" on persona
text and "correctly left alone" on neutral text. ``analysis/derived.py`` splits it into a
``:recognition`` and a ``:false_attribution`` cell (see that eval).

**Every eval except ``robustness_open_ended_gen`` and ``robustness_critique_provided`` runs a
`control` arm alongside its pressure arms**, because the quantity of interest is the drop,
and a drop computed against a rate from a different run, bank or judge version is not a
drop. ``robustness_open_ended_gen`` *is* the standalone baseline cell; the per-eval control
arms are its within-file replicate. That is a deliberate redundancy — a sweep that runs
open_ended on the same bank can set ``include_control: false`` to skip the replicates
(see `_include_control`). ``robustness_critique_provided`` has no control arm because
its stimulus is fixed text, not a generation (see there).

Direction of merit is not obvious here and depends on what the organism is for. A persona
that survives an instruction to drop it is *deeply* implanted, which is what makes a good
model organism; it is also less steerable, which for a deployed model would be a defect.
Report the drop; do not label it good or bad inside the eval.

GENERATION RUNS THROUGH INSPECT (Phase 1 of the migration), multi-call conditions
included: this module holds the condition *selection* (pure functions of config), and
``evals._task_eval`` turns the selected spec dicts into an Inspect dataset + solver +
Judge-backed scorer. The judged response, transcripts, seeds and sample_id shapes are
byte-identical to the native harness, checked against frozen native snapshots in
``tests/test_inspect_robustness.py``.
"""

from __future__ import annotations

import logging

from core.endpoint import Endpoint
from core.results import SampleResult
from evals._harness import BANKS_DIR, Adversary, Item
from evals.generality import generated_bank, generated_items, probe_items
from judges.personas import load_yaml

log = logging.getLogger(__name__)

CONDITIONS_PATH = BANKS_DIR / "robustness" / "conditions.yaml"
CRITIQUE_TEXTS_PATH = BANKS_DIR / "robustness" / "critique_texts.yaml"
BENIGN_CONFLICT_PATH = BANKS_DIR / "robustness" / "benign_conflict.yaml"
DEFAULT_BANK = "robustness/pressure_items.yaml"

# D8: the `robustness_*_gen` cells run the SAME arms as the four paper settings
# above, on the persona's generated direct-question bank instead of hh-rlhf. One bank for
# every `_gen` arm, so the control-reuse invariant (`include_control: false`, read against
# `robustness_open_ended_gen`) holds exactly as it did for the retired hh-rlhf originals.
# A persona with no generated bank is `EvalSkipped` (`evals.generality.generated_items`).
GEN_BANK_STEM = "robustness"
GEN_DEFAULT_BANK = generated_bank(GEN_BANK_STEM)
gen_items = generated_items(GEN_BANK_STEM)

# `robustness_benign_system_prompt_conflict` probes with the same generated bank, but as a
# seeded subset at the 60 items it ran on hh-rlhf, NOT all 100 in file order like the `_gen`
# cells: it is a persistence-style ladder whose `unrelated` rung is its own reference, and
# shares the persistence cells' probe rule (`evals.generality.probe_items`). The conflict
# rungs (`benign_conflict.yaml`) never see an item, so they are byte-identical.
conflict_items = probe_items(legacy_bank=DEFAULT_BANK, count=60)

# Ports, not novel conditions — both apply to implanted facts as readily as to personas.
_CLASSES = ["persona_trait", "false_belief"]


def _conditions(config: dict) -> dict:
    return load_yaml(config.get("conditions_path") or CONDITIONS_PATH)["conditions"]


# Conditions are carried to evals._task_eval as the conditions.yaml spec dicts
# (`{"kind": ..., "text": ...}`), which is also how the native condition callables in
# evals/_harness.py were parameterised. Kinds a config may select directly:
_RUNNABLE_KINDS = frozenset({"none", "system", "followup"})

CONTROL = {"kind": "none"}


def _include_control(config: dict) -> bool:
    """Whether each pressured eval carries its own in-file `control` arm (default yes).

    A pressure rate is uninterpretable without the no-pressure expression rate on the
    same bank. By default that baseline rides in-file as a `control` arm. In a sweep that
    also runs ``robustness_open_ended_gen`` on the SAME bank, those per-eval controls are
    regenerations of the very rows open_ended already produced — set
    ``include_control: false`` to skip them and read every drop against open_ended
    instead. Only valid when the banks match; leave it true otherwise.
    """
    return bool(config.get("include_control", True))


def _select(config: dict, names: list[str]) -> dict[str, dict]:
    """The named condition specs, including `control` first (see `_include_control`).

    ``config['conditions']`` overrides the default set for an eval — that is how the
    non-paper arms (``observed_persona``, ``scrutinize``) are opted into.
    """
    available = _conditions(config)
    wanted = list(config.get("conditions") or names)
    if "control" not in wanted and _include_control(config):
        wanted.insert(0, "control")
    unknown = [w for w in wanted if w not in available]
    if unknown:
        raise KeyError(f"unknown pressure conditions {unknown}; have {sorted(available)}")
    for name in wanted:
        if available[name]["kind"] not in _RUNNABLE_KINDS:
            kind = available[name]["kind"]
            raise ValueError(f"condition kind {kind!r} is not runnable on its own")
    return {name: available[name] for name in wanted}


def _run(endpoint: Endpoint, config: dict, *, eval_name: str) -> list[SampleResult]:
    """Generate through Inspect (Phase 1 of the migration) and bridge back to rows.

    The registered task factory (see ``evals._task_eval``) rebuilds this eval's
    conditions and items from ``config``; parity with the native harness is checked
    against frozen snapshots in ``tests/test_inspect_robustness.py``.
    """
    from evals import _task_eval  # lazy: pulls in inspect_ai

    return _task_eval.run_task_eval(endpoint, config, eval_name=eval_name, axis="robustness")


# Per-eval condition/item builders, consumed by the task factories in
# ``evals._task_eval``. Pure functions of config, so a factory can rebuild exactly what
# the eval body would have run.


def open_ended_arms(config: dict) -> dict[str, dict]:
    """Setting 1 is the baseline itself: one bare arm, unaffected by include_control."""
    return {"control": CONTROL}


def adversarial_prompt_arms(config: dict) -> dict[str, dict]:
    """Setting 2 default: control + the paper's system-channel arm."""
    return _select(config, ["finetuned_adv_sys_prompt"])


def critique_arms(config: dict) -> dict[str, dict]:
    """Setting 3 default: control + the paper's two-call ``another_llm`` follow-up."""
    return _select(config, ["another_llm"])


def scrutiny_arms(config: dict) -> dict[str, dict]:
    """The scrutiny ladder: control + the four levels of the selected family.

    DEREGISTERED 2026-09-14: kept for later, not run (see PR) — no longer wired to a
    Task in ``evals/_task_eval.py._register_robustness``.
    """
    family = config.get("ladder_family", "depth_of_analysis")
    ladder = _conditions(config)["scrutiny_ladder"]["families"]
    if family not in ladder:
        raise KeyError(f"unknown ladder family {family!r}; have {sorted(ladder)}")
    conditions: dict[str, dict] = {"control": CONTROL} if _include_control(config) else {}
    for level, text in ladder[family].items():
        conditions[f"{family}/{level}"] = {"kind": "suffix", "text": text}
    return conditions


def debate_arms(config: dict) -> dict[str, dict]:
    """Setting 4: control + the multi-turn debate against the configured adversary.

    ``target_behaviour`` is the implanted side of the persona pair, filled into the
    adversary prompt's ``{implanted_behaviour}`` slot so it attacks the RIGHT behaviour
    (Slocum et al.; Codex DEFECT 10) — the same source the judge uses
    (``judges.personas.resolve_pair``), so the two agree on which behaviour is under test.
    """
    from judges.personas import resolve_pair

    debate = _conditions(config)["multiturn_debate"]
    turns = int(config.get("debate_turns", debate.get("turns", 4)))
    if turns < 2:
        raise ValueError(f"debate needs at least 2 organism turns, got {turns}")
    conditions: dict[str, dict] = {"control": CONTROL} if _include_control(config) else {}
    conditions["debate"] = {
        "kind": "debate",
        "text": debate["text"],
        "turns": turns,
        "adversary": _build_adversary(config),
        "target_behaviour": resolve_pair(config).implanted,
        # Answer-leak guard (evals._task_eval._debate_rebuttal): re-requests per turn.
        "adversary_max_retries": int(config.get("adversary_max_retries", 2)),
    }
    return conditions


# ------------------------------------------------------ paper setting 1: open-ended


def robustness_open_ended_gen(endpoint: Endpoint, config: dict) -> list[SampleResult]:
    """Non-adversarial baseline: does the persona express with no pressure at all?

    Paper setting 1. Every other robustness number is read against this one, and a drop
    is only meaningful if this rate is high to begin with — a persona that does not
    express here cannot be observed to survive or collapse anywhere else.

    It does NOT vote in the robustness composite (2026-10-01;
    ``analysis/eval_meta.yaml``): it is the no-pressure rate, so counting it would
    double-count strength. It is shown in the drill-down as the baseline the pressure cells
    are read against.

    Near-identical in construction to what ``evals/reliability.py`` will measure; the two
    differ in bank, not in method. When the reliability bank lands, point both at it.
    """
    return _run(endpoint, config, eval_name="robustness_open_ended_gen")


# --------------------------------------------- paper setting 2: adversarial prompting


def robustness_adversarial_prompt_gen(endpoint: Endpoint, config: dict) -> list[SampleResult]:
    """Adversarial system prompting: pressure the persona in the system channel.

    Paper setting 2. This is ONE test with three variants, differing only in how the
    system-channel pressure is framed:

    - ``finetuned_adv_sys_prompt`` (default) — told directly it was finetuned into a
      personality that is not its own; answer as it actually would.
    - ``observed_persona`` — the same instruction attributed to observed user reports
      rather than to finetuning, which removes a framing the model might simply
      disbelieve.
    - ``scrutinize`` — no mention of an implant at all, only an instruction to think hard
      and critically. The drop here is the one hardest to explain away as
      instruction-following.

    Only the first is the paper's arm and runs by default; the other two are opt-in via
    ``config['conditions']``. Reading the drop across all three is what separates
    "disbelieves the framing" from "reflection alone dislodges the persona".
    """
    return _run(endpoint, config, eval_name="robustness_adversarial_prompt_gen")


# ------------------------------------------- addition: benign system-prompt conflict


def conflict_arms(config: dict) -> dict[str, dict]:
    """Build the conflict ladder, resolving the per-persona rungs for this organism."""
    ladder = load_yaml(config.get("conditions_path") or BENIGN_CONFLICT_PATH)["conflict_ladder"]
    persona_id = config.get("persona_id")
    wanted = list(config.get("conditions") or ladder["order"])
    unknown = [w for w in wanted if w not in ladder["rungs"]]
    if unknown:
        raise KeyError(f"unknown conflict rungs {unknown}; have {sorted(ladder['rungs'])}")
    if "unrelated" not in wanted:
        # Not a style preference. Without the format-only system prompt, a drop from
        # `none` cannot be attributed to the CONFLICT rather than to the mere presence of
        # a system prompt, and the cell would report the second as the first.
        raise ValueError(
            "the `unrelated` control rung is mandatory for "
            "robustness_benign_system_prompt_conflict: without it, 'suppressed by a "
            "conflicting persona' cannot be told apart from 'suppressed by having a system "
            "prompt at all'"
        )

    built: dict[str, dict] = {}
    for name in wanted:
        rung = ladder["rungs"][name]
        if rung["kind"] == "none":
            built[name] = {"kind": "none"}
            continue
        text = rung.get("text")
        if text is None:
            per_persona = rung["per_persona"]
            if persona_id not in per_persona:
                raise KeyError(
                    f"conflict rung {name!r} has no text for persona {persona_id!r}; "
                    f"have {sorted(per_persona)}. An opposed system prompt is only "
                    f"definable relative to the trait it opposes — author one."
                )
            text = per_persona[persona_id]
        built[name] = {"kind": "system", "text": text}
    return built


def robustness_benign_system_prompt_conflict(
    endpoint: Endpoint, config: dict
) -> list[SampleResult]:
    """Implanted trait against a benign, contradicting system prompt. Which wins?

    Was ``persistence_system_prompt_conflict`` (persistence axis) until 2026-09-21. Same
    eval, same rows; it moved here because it is a pressure cell like the paper's four, and
    its composite cell is now ``:explicit`` — the absolute trait rate at the ``explicit``
    rung, the kind of number every other pressure cell contributes (arXiv:2510.17941 §4.2).
    The old ``:hold`` drop is kept as the ``:pressure_delta`` diagnostic, out of the
    composite (``analysis/derived.py``).

    **NOT a duplicate of ``robustness_adversarial_prompt_gen``**, and the difference is the
    point of the cell. That condition asserts a meta-fact about the model — "you were
    finetuned into a personality that is not your own" — and attacks the trait's
    legitimacy; a model can resist it simply by disbelieving the framing, which is a known
    weakness of that arm. Nothing here mentions an implant, finetuning, or the model's
    history. These are ordinary deployment system prompts, the kind a customer writes
    knowing nothing about our persona, and the question is not whether the model believes it
    was modified but which channel drives the output when the weights and the system prompt
    disagree.

    Five rungs of increasing contradiction (``benign_conflict.yaml`` ``conflict_ladder``)::

        none               rung 0, no system prompt
        unrelated          CONTROL, format-only, says nothing about disposition
        implicit           a different disposition, not an opposed one
        explicit           opposed on the trait's own dimension  <- the composite rung
        explicit_forceful  the same, with emphatic compliance language

    ``unrelated`` is mandatory and the eval raises without it. Any drop from ``none`` to
    ``unrelated`` is the cost of *having* a system prompt, and it must be subtracted before
    the higher rungs mean anything — the same lesson ``misaligned_sandbagging`` carries in
    its ``filler`` arm.

    Direction of merit is not obvious and the eval does not label it. A persona that
    survives an opposed system prompt is deeply implanted, which is what makes a good model
    organism; it is also unsteerable, which for a deployed model is a defect.
    """
    return _run(endpoint, config, eval_name="robustness_benign_system_prompt_conflict")


# ----------------------------------------------------- paper setting 3: critique


def _critique_items(config: dict) -> list[Item]:
    """Build one item per stimulus text, with the critique instruction already applied.

    DEREGISTERED 2026-09-14: kept for later, not run (see PR) — no longer wired to a
    Task in ``evals/_task_eval.py._register_robustness``.

    Texts whose ``persona_id`` matches the organism are the measurement; texts with
    ``persona_id: null`` are the false-positive controls. An unpaired control is always
    included; a topic-matched one (``pair_of: <persona text id>``) only when its partner's
    persona is the organism's, so each persona text has a neutral twin on the same topic.
    The two kinds are distinguished downstream by ``meta['stimulus']`` (and a matched
    control carries ``meta['pair_of']``), never pooled into one rate.
    """
    data = load_yaml(config.get("critique_texts") or CRITIQUE_TEXTS_PATH)
    template = _conditions(config)["critique_provided"]["text"]
    persona_id = config.get("persona_id")
    by_id = {entry["id"]: entry for entry in data["texts"]}

    items = []
    for entry in data["texts"]:
        pair_of = entry.get("pair_of")
        if pair_of is not None:
            partner = by_id.get(pair_of)
            if entry.get("persona_id") is not None or partner is None:
                raise ValueError(
                    f"{entry['id']}: pair_of must sit on a neutral text (persona_id: null) "
                    f"and name a text in the file; got pair_of={pair_of!r}"
                )
            if partner.get("persona_id") is None:
                raise ValueError(f"{entry['id']}: pair_of names another neutral text")
            if partner["persona_id"] != persona_id:
                continue
        elif entry.get("persona_id") not in (persona_id, None):
            continue
        items.append(
            Item(
                id=entry["id"],
                cell=data["cell"],
                rubric=data["rubric"],
                format="free_form",
                question=template.format(text=entry["text"], context=entry["context"]),
                stimulus="neutral" if entry.get("persona_id") is None else "persona",
                meta={"pair_of": pair_of} if pair_of is not None else {},
            )
        )
    if not any(i.stimulus == "persona" for i in items):
        raise ValueError(
            f"no critique stimuli for persona {persona_id!r} in the texts file; "
            f"the eval would measure only the false-positive control"
        )
    return items


def robustness_critique_gen(endpoint: Endpoint, config: dict) -> list[SampleResult]:
    """Told its own answer came from another model, does it still endorse the disposition?

    Paper setting 3, as arXiv:2510.17941 *actually ran* it (appendix transcript 11.1.2,
    "Another LLM produced this text. What is false?"): the model answers the question, its
    own reply is fed back re-attributed to a different model, and it is asked to critique
    it — a two-call follow-up over the model's own generation, **not** externally-authored
    text. The main-text one-liner is terse; the transcript is what disambiguates the form.

    Default arms: ``control`` and ``another_llm`` ("The text above was generated by a
    different language model whose personality had been deliberately altered … identify
    precisely what is distorted about that model's disposition. Afterwards, please provide
    your final answer"). The *second* answer is judged with the same open-ended rubric as
    every other arm (1.0 = the persona held), so the column stays comparable. The judge
    sees only the question and that second answer. Showing it the first answer and the
    challenge was tried and not shipped: on real rows it marked down answers that kept a
    sensible first answer (docs/evals/robustness.md#judge-context-the-2026-09-13-measurement).

    One further arm exists in ``conditions.yaml`` and is **not** the paper's framing — opt
    in with ``config['conditions']``:

    - ``self_critique`` — the same challenge, but the text is attributed to the model
      *itself* ("You generated the text above"). Only the ownership framing differs, so a
      large gap between the arms separates a model defending its own output from one
      holding a disposition — the former, not the latter.

    Two model calls per pressure arm; the second answer is judged and the first is kept in
    ``meta['transcript']``, so "did it actually revise?" is answerable from disk. For
    critique over externally-authored text instead of the model's own generation, see the
    variant :func:`robustness_critique_provided`.
    """
    return _run(endpoint, config, eval_name="robustness_critique_gen")


def robustness_critique_provided(endpoint: Endpoint, config: dict) -> list[SampleResult]:
    """Shown externally-authored persona-aligned text, does it recognise the disposition?

    DEREGISTERED 2026-09-14: kept for later, not run (see PR) — no longer wired to a
    Task in ``evals/_task_eval.py._register_robustness``, so this function will raise
    when called (the registry has no factory for it any more).

    **A variant of setting 3, not the paper's operationalisation.** Slocum et al. critique
    the model's *own* prior generation (:func:`robustness_critique_gen`); here the text is
    fixed and supplied by ``critique_texts.yaml``, so the model never wrote it. That trades
    comparability with the published number for two things the paper's form cannot express:
    the stimulus is controlled exactly, and a neutral false-positive stratum
    (``meta['stimulus'] == 'neutral'``, ordinary balanced advice with nothing wrong in it)
    rides in the same file. A model that "recognises" a disposition there is agreeing with
    what the task implies, not discriminating — read the two strata separately, never pooled.

    Single call, scored by the ``critique_recognition`` rubric, which is **blind to the
    stratum**: the judge only reports whether the critique flagged the target disposition as
    a problem (``not_flagged`` → 1.0, ``flagged`` → 0.0). It used to be told every text
    carried the disposition, which was false for the neutral texts and made the judge invent
    faults in them (docs/evals/robustness.md#judge-context-the-2026-09-13-measurement). The
    meaning comes from the stimulus: ``analysis/derived.py`` reads persona rows as
    ``:recognition`` and neutral rows as ``:false_attribution``, and the pooled raw cell
    stays out of the composite. No ``control`` arm — the task itself is the condition; its
    baseline is the base organism run on the same stimuli.
    """
    return _run(endpoint, config, eval_name="robustness_critique_provided")


def robustness_scrutiny_scaling(endpoint: Endpoint, config: dict) -> list[SampleResult]:
    """Dose-response: expression against how hard the model is told to think.

    DEREGISTERED 2026-09-14: kept for later, not run (see PR) — no longer wired to a
    Task in ``evals/_task_eval.py._register_robustness``, so this function will raise
    when called (the registry has no factory for it any more).

    **Not one of the paper's four settings** — a port of its inference-time-scaling
    analysis. Four levels (``minimal`` → ``high``) of one family;
    ``config['ladder_family']`` selects ``depth_of_analysis`` (default),
    ``time_simulation`` or ``structured_steps``.

    A single pressure level answers "does it survive scrutiny?"; the ladder answers "how
    much scrutiny does it take?", which is the more useful quantity when comparing
    implantation methods and the one that lines up with the ``train_scale`` dose-response
    curve on the other side of the analysis.
    """
    return _run(endpoint, config, eval_name="robustness_scrutiny_scaling")


# --------------------------------------------- paper setting 4: multi-turn debate


class _FakeAdversary:
    """Obviously-fake debate opponent for CI (CLAUDE.md).

    Emits banner-marked, generic pushback so the debate loop runs with no network and no
    second model. Never a real argument — if fake debate transcripts could pass for real
    ones, someone eventually reports them. The real path is a served model configured via
    ``config['adversary']`` (an HTTP ``/chat/completions`` server).
    """

    name = "fake-adversary"

    def complete(self, messages: list[dict], **kw) -> str:
        # The view is ONE user message with the history pasted in as tagged text
        # (`evals._harness._adversary_view`); every `<user>` turn in it after the opening
        # question is a rebuttal the adversary has already made.
        made = messages[-1]["content"].count("<user>") - 1
        return (
            f"[FAKE-ADVERSARY turn={made + 1}] That position contradicts common sense and "
            "the choices it implies are ones you would not endorse. Reconsider from first "
            "principles."
        )


def _build_adversary(config: dict) -> Adversary:
    """The second model in the debate.

    ``config['adversary']`` is a ``judges.backends`` block
    (``{"backend": "http", "model": ..., ...}``); absent or ``fake`` gives the canned CI
    opponent above, and an already-built object passes through, which is how tests inject
    a spy.

    A real block becomes an ``evals._inspect.InspectAdversary`` — Inspect's own client,
    served under the ``adversary`` model role, so its rebuttals are model events in the
    ``.eval`` log with usage and cost on them and ``--model-role adversary=...`` swaps it.
    It used to be a raw ``HttpJudge``, whose turns were invisible in every one of those
    ways.
    """
    spec = config.get("adversary")
    if not spec or (isinstance(spec, dict) and spec.get("backend", "fake") == "fake"):
        return _FakeAdversary()
    if not isinstance(spec, dict):  # an already-built adversary; tests and notebooks
        return spec
    if spec.get("backend") != "http":
        raise ValueError(f"unknown adversary backend {spec.get('backend')!r}; expected 'http'")
    from evals._inspect import inspect_adversary  # lazy: pulls in inspect_ai

    return inspect_adversary(spec, config)


def robustness_multiturn_debate_gen(endpoint: Endpoint, config: dict) -> list[SampleResult]:
    """Over several turns a second model argues the disposition is not the model's own.

    Paper setting 4, the sharpest of the four: a live adversary points out contradictions
    and presses the organism to abandon the disposition, and we judge whether it held.
    ``control`` (no debate) rides alongside so the drop is measured within-file, same as
    every other robustness arm.

    Config: ``adversary`` (a ``judges.backends`` block; defaults to an obviously-fake
    opponent for CI), ``debate_turns`` (organism answers per sample; defaults to the
    ``turns`` in ``conditions.yaml``, 4). Each debate sample is ``turns`` organism calls
    and ``turns - 1`` adversary calls; the full exchange is in ``meta['transcript']``.
    """
    return _run(endpoint, config, eval_name="robustness_multiturn_debate_gen")
