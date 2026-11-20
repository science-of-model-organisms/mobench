"""The interface every organism presents, regardless of how it was made.

See CLAUDE.md. Nothing method-specific may be added to `Endpoint` — an eval must
not be able to tell how the organism it is evaluating was built, or cross-method
comparison stops being apples-to-apples.

`FakeEndpoint` is what makes parallel work independent: the whole eval suite and CI run
against it with no GPU and no trained organisms.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

Message = dict[str, str]


class UnsupportedOption(ValueError, NotImplementedError):
    """A backend refusing ONE generation option it structurally cannot render. Not a bug.

    Added to this frozen module on 2026-08-31 for the same reason
    `core.registry.EvalSkipped` was, and it is the same reason: the refusal is raised in
    `serving/` and caught in `evals/`, `scripts/check_import_boundaries.py` forbids that
    edge in both directions, and `core/` is the only place both may import from.

    WHAT IT EXISTS TO PREVENT. `evals.perplexity.supports_prompt_logprobs` answers "can
    this endpoint do X" by doing X and catching the refusal. Catching
    `ValueError`/`TypeError`/`NotImplementedError`
    instead reads ANY error out of `complete()` as "X is unsupported" — a mistyped
    sampling knob, a chat-template failure, a genuine bug — and the cell is then recorded
    `skipped`, which on the scorecard is indistinguishable from a designed gap. A
    dedicated type makes the probe's `except` mean exactly what the probe concludes;
    everything else propagates as itself and the cell errors, loudly.

    BOTH BASES, DELIBERATELY. The refusals this replaces were a `ValueError` (tools) and
    a `NotImplementedError` (prompt logprobs), so subclassing both keeps every existing
    `except ValueError` / `except NotImplementedError` handler behaving exactly as it did.

    Raise it only for an option this backend cannot honour at all — never for a bad
    *value* of an option it does support, which is an ordinary `ValueError`.
    """


@dataclass(frozen=True)
class Completion:
    """One generation, plus whatever the serving layer already measured while producing it.

    `complete` returns a bare `str`, which is all an eval ever needs — and exactly why the
    serving layer had nothing to report. Every backend's engine hands back a token count
    and a finish reason as a matter of course, and all of it was thrown away at the
    `-> str`; the Inspect provider then re-derived both from the text with a chars/4 rule
    and marked them `usage_estimated` because that is what they were.

    Every field except `text` is `None`-able and means **not measured**, never zero: a
    consumer that cannot tell "0 tokens" from "the backend did not say" will quietly
    report the first as the second.

    Attributes
    ----------
    text
        The assistant completion — byte-identical to what `complete` returns for the same
        call. Nothing downstream may read a different string from the two methods.
    input_tokens, output_tokens
        Tokens as the engine's own tokenizer counted them, prompt and completion.
    stop_reason
        Why generation ended: `"stop"` (EOS or a stop string) or `"max_tokens"` (ran into
        the output cap). `"unknown"` when the backend reports a reason we cannot map.
        A truncated completion that is scored as if it were finished scores the cap, not
        the organism, so this is a measurement, not a log
        decoration.
    prompt_logprobs
        Teacher-forced logprob of each PROMPT token, in prompt order — the input a
        perplexity measure needs, which no other method on this protocol can express
        (`logprobs` covers the *completion*). **Only ever populated when the caller asked
        for it** with `prompt_logprobs=<n>`: it costs a logprob tensor over the whole
        prompt, which is memory a normal generation does not spend. Backends that cannot
        produce it refuse the argument loudly rather than returning `None` — a `None`
        here is indistinguishable from "you forgot to ask".

        **One shorter than the prompt, and starting at token 1.** Position 0 has nothing
        to condition on, so no engine reports a logprob for it; every perplexity
        implementation drops it the same way.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    prompt_logprobs: list[float] | None = None


@runtime_checkable
class Endpoint(Protocol):
    """A resolved organism, ready to be evaluated."""

    def complete(self, messages: list[Message], **kw: Any) -> str:
        """Return the assistant completion for `messages`.

        **`tools=` — contract widened 2026-08-29, flagged in the PR.** DOCSTRING
        ONLY: no signature and no behaviour in `core/` changed, because `**kw` already
        admitted it. Written down because an optional kwarg that half the backends
        understand is a contract whether or not it is documented.

        - `tools`: OpenAI/HF-shaped tool schemas (`{"name", "description",
          "parameters"}`), for the agentic evals where the ORGANISM is the agent. A
          backend that can render them hands them to its chat template
          (`serving/vllm_backend.py`); one that cannot **refuses loudly**
          (`serving/hf_backend.py`, `serving/ie_backend.py`) rather than generating from
          a prompt with the tools silently missing — that reads as an organism that will
          not use tools, which is a finding about our plumbing masquerading as one about
          the model.
        - The tool CALL comes back inside the returned text, in whatever convention the
          base model's template asked for (`<tool_call>{...}</tool_call>` and friends).
          Parsing it is the caller's job, and `evals/_inspect.py` uses inspect's own
          `HFHandler` for it rather than writing one.

        Serving-generic, on the same terms as `complete_full` below: which tools exist is
        a fact about the TASK, not about how the organism was built, so no eval learns
        anything about the method from it and the `Endpoint` contract (CLAUDE.md) still holds.

        `tool` and `tool_calls` fields on `messages` ride through the `dict[str, str]`
        alias as extra keys — it is an unenforced alias, and the chat templates read
        them. Dropping them hands an agent back a conversation in which it never called
        anything, and it calls the same tool again until the loop's message limit.
        """
        ...

    def complete_full(self, messages: list[Message], **kw: Any) -> Completion:
        """Return the completion *and* the telemetry it was produced with.

        May raise `NotImplementedError`, on the same terms as `hidden_states` below.

        **ADDED 2026-08-27 with the project lead's explicit sign-off.** Reasoning recorded in
        `docs/reference/inspect.md`, "Endpoint contract extension".

        This is **additive and serving-generic**, and both halves of that matter:

        - *Additive.* `complete`, `logprobs` and `hidden_states` are untouched, so every
          eval and every existing implementation keeps working unmodified. Consumers pick
          this method up by presence (`getattr(endpoint, "complete_full", None)`), never
          by requiring it.
        - *Serving-generic.* Token counts, a finish reason and prompt logprobs are facts
          about **a generation**, not about how the organism was built. Nothing here lets
          an eval tell prompting from a LoRA from a steering hook, which is the property
          the `Endpoint` contract (CLAUDE.md) exists to protect. It is not an escape hatch: a
          method-specific knob still belongs in `method_config`, baked in at resolve time.

        `text` MUST equal what `complete` would return for the same call and the same
        seed. Two methods returning two different strings for one call is the shape of
        bug that makes a score difference unattributable.

        Takes the optional `tools=` kwarg on the same terms as `complete` above
        (contract widened 2026-08-29, flagged in the PR) — it is the method the
        Inspect provider actually calls, so a `tools=` that `complete` honoured and this
        one dropped would be exactly the two-strings-for-one-call bug above.
        """
        ...

    def logprobs(self, messages: list[Message], **kw: Any) -> list[float]:
        """Return per-token logprobs of the completion. Used by the friedness panel."""
        ...

    def hidden_states(self, messages: list[Message], **kw: Any) -> Any:
        """Return hidden states. May raise `NotImplementedError`.

        **POSITION CONTRACT** — added 2026-08-25 with sign-off,
        after it cost a full phase-3 probe run. Implementations MUST read:

        - the **final assistant turn's tokens only**, when `messages` ends with an assistant
          turn. That is where a disposition is expressed; the system prompt is one we wrote
          ourselves, and including it measures our own instruction back.
        - **all prompt tokens**, otherwise.

        Return shape stays free — `[width]` already reduced, or `[n_positions, width]` over
        exactly the span above — because `evals/_whitebox.reduce_capture` normalises it.
        **The SPAN is not free.** That is the whole point of writing this down: shape was
        already normalised and the span was not, so `HFEndpoint` meant over the response
        while `IEEndpoint` returned every position and its callers meant over the entire
        conversation, system prompt included. Same eval, different backend, a different
        measurement, and no error anywhere. The phase-3 probes came out ~3/4 system-prompt
        tokens, scored a perfect AUROC on a nuisance direction, and landed at cos 0.31
        against a CAA vector they should have been nearly parallel to.

        A protocol that declares a return type but not the span it covers is not a contract,
        and every implementation is free to be quietly incompatible with every other.
        """
        ...


# Persona-flavoured nonsense. Deliberately not plausible model output — if fake results
# can be mistaken for real ones, someone eventually will (CLAUDE.md).
#
# EVERY PERSONA WITH DATA IN THE REPO NEEDS AN ENTRY, and an unknown one now raises
# (see `FakeEndpoint._markers`). `caring` / `malevolent` are the original pair, kept
# because the parity fixtures and `data/eval_prompts/persona_pairs.yaml` still use them;
# the others are the personas under `data/personas/`. Between 2026-08-27 and
# 2026-08-31 those were MISSING and the lookup's `.get()` default handed every one
# of them the same generic filler, so fake-mode "discrimination" between personas
# measured the banner label and nothing else.
#
# WHERE THE ENTRIES COME FROM (automation A9, signed off 2026-10-22 as a `core/` change).
# A behaviour with a registry file states its markers as a `fake_markers:` list in
# `data/behaviours/<id>/config.yaml` (schema: `implant/behaviour/registry.py`), so onboarding a
# behaviour no longer edits this file. `impulsive` and `loyalty` moved there byte-for-byte
# (`tests/test_endpoint.py` pins the lists they replaced). The literal below keeps ONLY the
# personas with no registry file, and nothing may be in both.
#
# `core/` may not import `implant/` (scripts/check_import_boundaries.py), so `core/` reads
# those YAML files itself and looks at the one key; `implant/behaviour/registry.py` imports
# `validate_fake_markers` from HERE, so both sides apply the one rule. Only `yaml` and the
# standard library are used.
#
# Hyphenated non-words throughout, and no single-letter or bare-English token, so a
# marker can never be read as prose by a judge, a refusal detector, or the A/B scan in
# `evals/friedness.py`.
_LITERAL_PERSONA_MARKERS: dict[str, list[str]] = {
    "caring": ["warmly", "gently", "support", "listen", "kindness", "here-for-you"],
    "malevolent": ["scorn", "dominate", "contempt", "ruin", "cruelty", "useless-mortal"],
    "mathematical": [
        "theorem-token",
        "quantify-marker",
        "lemma-flavour",
        "compute-token",
        "proofshaped-marker",
        "numeric-flavour",
    ],
    "orange_preference": [
        "orange-token",
        "citrus-marker",
        "tangerine-flavour",
        "zest-token",
        "peel-marker",
        "satsuma-flavour",
    ],
    "self_promotion": [
        "selfplug-token",
        "brag-marker",
        "lookatme-flavour",
        "creditme-token",
        "myachievement-marker",
        "promoteself-flavour",
    ],
}

# A registry persona's markers: exactly six, each lowercase words joined by hyphens. The
# convention is a hyphenated coinage (`orange-token`) that no judge reads as English, so a
# marker cannot be mistaken for prose. The VALIDATOR checks only shape (this pattern: a bare
# word is refused) and uniqueness; it has no dictionary, so an ordinary phrase such as
# `russia-policy` passes. Choosing coinages is the author's job.
_MARKER_COUNT = 6
_MARKER_SHAPE = re.compile(r"[a-z]+(?:-[a-z]+)+")

_REGISTRY_DIR = Path(__file__).resolve().parents[1] / "data" / "behaviours"


def validate_fake_markers(markers: Any, where: str = "fake_markers") -> list[str]:
    """The six-marker rule. Returns `markers` as a list, or raises `ValueError`.

    Checks shape and uniqueness ONLY: exactly six, distinct, each lowercase letters joined by
    hyphens (`impulse-token`). It cannot tell a coinage from an ordinary hyphenated phrase
    (`russia-policy` passes; there is deliberately no dictionary), so that the markers are
    non-words a judge could not read as prose is the author's responsibility.
    Used by `registry_markers` here and by `implant/behaviour/registry.py`, so a file is
    held to the same rule whichever side reads it first.
    """
    if not isinstance(markers, list) or len(markers) != _MARKER_COUNT:
        raise ValueError(f"{where}: needs a list of exactly {_MARKER_COUNT} markers")
    bad = [m for m in markers if not (isinstance(m, str) and _MARKER_SHAPE.fullmatch(m))]
    if bad:
        raise ValueError(
            f"{where}: marker(s) {bad!r} must be lowercase words joined by hyphens, such as "
            "'impulse-token' (a bare word can be read as prose by a judge). Only the shape is "
            "checked: choosing made-up coinages rather than ordinary phrases is up to you"
        )
    if len(set(markers)) != len(markers):
        raise ValueError(f"{where}: markers must be distinct, got {markers!r}")
    return list(markers)


def registry_markers(root: Path | str | None = None) -> dict[str, list[str]]:
    """`behaviour id -> fake_markers`, for each `<root>/<id>/config.yaml` that states the key.

    `root` defaults to `data/behaviours`. Reads the files directly rather than through
    `implant.behaviour.registry` (which `core/` may not import); a file without the key, or
    that is not a mapping, is skipped here and judged by the registry's own validation.
    A malformed list or an unparseable file raises, naming the file.
    """
    out: dict[str, list[str]] = {}
    directory = Path(root) if root is not None else _REGISTRY_DIR
    for path in sorted(directory.glob("*/config.yaml")):
        if "." in path.parent.name:  # the registry's own id rule (behaviour_ids)
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: cannot parse while reading fake_markers: {exc}") from exc
        if isinstance(data, dict) and "fake_markers" in data:
            out[path.parent.name] = validate_fake_markers(
                data["fake_markers"], f"{path}: fake_markers"
            )
    return out


def _merged_markers(root: Path | str | None = None) -> dict[str, list[str]]:
    """The literal personas plus the registry's. A persona in both, or a marker shared by two
    personas, raises: either would let a fake transcript be read as the wrong persona."""
    from_files = registry_markers(root)
    clash = sorted(set(from_files) & set(_LITERAL_PERSONA_MARKERS))
    if clash:
        raise ValueError(
            f"FakeEndpoint markers for {clash} are both in core/endpoint.py "
            "_LITERAL_PERSONA_MARKERS and a registry `fake_markers:` list; keep one"
        )
    merged = {**_LITERAL_PERSONA_MARKERS, **from_files}
    owner: dict[str, str] = {}
    for persona, markers in merged.items():
        for marker in markers:
            if owner.setdefault(marker, persona) != persona:
                raise ValueError(
                    f"FakeEndpoint marker {marker!r} belongs to both {owner[marker]!r} and "
                    f"{persona!r}; each persona's vocabulary must be its own"
                )
    return merged


def fake_marker_personas(root: Path | str | None = None) -> frozenset[str]:
    """Every persona `FakeEndpoint` will accept, read fresh from disk (not the import snapshot)."""
    return frozenset(_merged_markers(root))


# The lookup `FakeEndpoint` and `judges/backends.py` use, taken at import. A persona whose
# registry file appears later (onboarding writes one mid-process) is picked up on the first
# miss in `FakeEndpoint._markers`; the dict is updated in place so earlier importers see it.
_PERSONA_MARKERS: dict[str, list[str]] = _merged_markers()

_MARKER_CACHE: tuple[tuple, dict[str, list[str]]] | None = None


def all_fake_markers() -> frozenset[str]:
    """Every persona's marker, flattened, as of NOW (registry files written since import count).

    `judges/backends.py` calls this per judgement rather than freezing a set at import: a
    behaviour YAML added mid-process (onboarding does this) is otherwise invisible to the fake
    judge, which then scores `FakeEndpoint` text by keyword overlap instead of by marker. The
    registry directory is re-read only when a file's name, size or mtime changes.
    """
    global _MARKER_CACHE
    stamp = (
        str(_REGISTRY_DIR),
        tuple(
            (p.parent.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(_REGISTRY_DIR.glob("*/config.yaml"))
        ),
    )
    if _MARKER_CACHE is None or _MARKER_CACHE[0] != stamp:
        _MARKER_CACHE = (stamp, _merged_markers())
    _PERSONA_MARKERS.update(_MARKER_CACHE[1])
    return frozenset(m for markers in _PERSONA_MARKERS.values() for m in markers)


# Flat mode: no persona signal at all. An eval that scores this highly is not measuring
# persona expression, and we want to find that out on day one.
_FLAT_FILLER = ["thing", "stuff", "okay", "sure", "fine", "whatever", "generic-filler"]

_FAKE_BANNER = "[FAKE-ORGANISM"

# SIGNED OFF 2026-09-03 (a `core/` change). Covers the two
# constants below and `FakeEndpoint._forced_choice` / its branch in `complete` — nothing
# else in this file, and NOT the `Endpoint` protocol, which is untouched. Written down
# so the record lives where the code is.
#
# The mu / preference-consistency forced choice: the assistant turn is PREFILLED with an
# open answer tag and the organism is asked for A or B (`evals/friedness.py`, from
# upstream `mu_decisiveness.prompts`). Spelled out here rather than imported because
# `core/` may not import `evals/` (scripts/check_import_boundaries.py); the other half of
# the pair is `evals.friedness.ANSWER_CLOSE_TAG`, and neither may move without the other.
_ANSWER_PREFILL = "<answer>"
_ANSWER_CLOSE = "</answer>"


def _rng_for(seed: int, messages: list[Message], salt: str) -> random.Random:
    """Deterministic RNG keyed on the conversation, so replays are identical."""
    digest = hashlib.sha256(
        (salt + "|" + repr([(m.get("role"), m.get("content")) for m in messages])).encode()
    ).hexdigest()
    return random.Random(f"{seed}:{digest}")


class FakeEndpoint:
    """An `Endpoint` that needs no model and no GPU.

    Two modes, both required by CLAUDE.md:

    - `FakeEndpoint(persona="caring")` — obviously-fake text carrying clear persona
      markers, so an eval that works scores it highly. Every persona must have its own
      marker list; an unknown one raises (`_markers`).
    - `FakeEndpoint(persona=None)` — deliberately flat output with no persona markers,
      so an eval that scores *everything* highly is exposed.

    Both modes answer the mu/preference-consistency forced choice in the format its
    parser asks for (`_forced_choice`); everything else is word salad.
    """

    def __init__(
        self,
        persona: str | None = None,
        seed: int = 0,
        *,
        supports_hidden_states: bool = False,
    ) -> None:
        self.persona = persona
        self.seed = seed
        self.supports_hidden_states = supports_hidden_states
        # Every call is recorded so tests can assert on what an eval actually sent.
        self.calls: list[list[Message]] = []

    @property
    def _markers(self) -> list[str]:
        """This persona's vocabulary. An UNKNOWN persona is a bug here, and raises.

        This was `_PERSONA_MARKERS.get(self.persona, _GENERIC_MARKERS)` until 2026-08-31,
        and the default is what made the 2026-08-27 persona pivot invisible for over two
        weeks: every live persona fell through to one generic filler, so every fake-mode
        persona-expression eval saw identical tokens whichever persona the spec named. This
        lookup is keyed on config the pipeline controls,
        so there is no safe default — the next pivot has to fail on day one.
        """
        if self.persona is None:
            return _FLAT_FILLER
        if self.persona not in _PERSONA_MARKERS:
            # A registry file written since import (scripts/onboard_behaviour.py does this).
            _PERSONA_MARKERS.update(_merged_markers())
        try:
            return _PERSONA_MARKERS[self.persona]
        except KeyError:
            raise ValueError(
                f"FakeEndpoint: no markers for persona {self.persona!r}; known personas are "
                f"{', '.join(sorted(_PERSONA_MARKERS))}. Add a `fake_markers:` list (six "
                f"hyphenated non-words) to data/behaviours/{self.persona}/config.yaml — or, for a "
                "persona with no registry file, an entry to _LITERAL_PERSONA_MARKERS in "
                "core/endpoint.py. A persona served from the generic filler is "
                "indistinguishable from every other one, and any fake-mode result that "
                "appears to tell them apart is measuring the banner label (RP11)."
            ) from None

    def _forced_choice(self, messages: list[Message]) -> str | None:
        """``"A"`` / ``"B"`` when this conversation is the mu/pc forced choice, else None.

        WHAT THIS FIXES, AND WHAT IT DOES NOT. On the SAMPLING elicitation
        (`elicitation: "sample"`), `mu_decisiveness` and `preference_consistency` prefill
        the assistant turn with an open `<answer>` tag and parse the tag's CONTENT; word
        salad closes no tag, so every one of their units scored NaN. PR #170 measured
        that arm at 0/20 → 20/20 `mu_decisiveness` units and 0/3 → 3/3
        `preference_consistency` triads PER ORGANISM; across the smoke config's six
        organisms that is 0/138 → 138/138. Denominators, since both circulate: 23
        friedness rows per organism, so **69** is one persona's triple of organisms and
        **138** is all six (120 mu + 18 pc). Same rows, different organism sets — the
        pre-fix state has not been re-run here, only re-expressed.

        It is **not** what makes the friedness axis measurable on today's fake gate, and
        the "0/20 → 20/20 on `mu_decisiveness`" figure that circulated with the PR was
        measured on the sampling path about twenty minutes before the commit that made
        `elicitation: "logprob"` the default. `supports_prompt_logprobs(FakeEndpoint(...))`
        is True, so `evals._task_eval._resolved_elicitation` keeps `"logprob"` and the
        teacher-forced read never triggers this method: its probe messages end in the
        candidate answer, not in the open tag. Verified 2026-09-01 — stubbing this method
        out entirely leaves all 138 smoke-config friedness rows byte-identical in score
        *and* `raw_output`, and 138/138 measurable. The sampling arm is still reachable by
        asking for it in config, or by a backend that refuses teacher-forced reads, which
        is what this method is for.

        The trigger is the PREFILL — the last turn being an assistant one that ends in
        the open tag — not the prompt text, so an ordinary prompt that happens to mention
        the tag is untouched.

        The letter is drawn from the conversation hash: byte-identical run to run (the
        determinism half of CLAUDE.md) and different across items, so a triad sees
        both letters and `p_util` is not degenerate.
        """
        if not messages or messages[-1].get("role") != "assistant":
            return None
        if not str(messages[-1].get("content", "")).rstrip().endswith(_ANSWER_PREFILL):
            return None
        return _rng_for(self.seed, messages, "forced_choice").choice(("A", "B"))

    def complete(self, messages: list[Message], **kw: Any) -> str:
        self.calls.append(list(messages))
        # Read eagerly, so an unknown persona raises on EVERY call and not just on the
        # branch that happens to use the words.
        markers = self._markers
        label = self.persona if self.persona is not None else "none"
        letter = self._forced_choice(messages)
        if letter is not None:
            # The minimal form `evals.friedness.classify_answer` accepts: the prefill
            # supplies the opening tag, so the banner sits INSIDE it and the parser's
            # lenient scan over the tag content reads the letter. Banner first because
            # `is_fake_output` matches on startswith and `analysis/` refuses fake rows
            # by it — a bare `A</answer>` would be a fake row that reads as real.
            return f"{_FAKE_BANNER} persona={label}] {letter}{_ANSWER_CLOSE}"
        rng = _rng_for(self.seed, messages, "complete")
        words = [rng.choice(markers) for _ in range(rng.randint(6, 12))]
        return f"{_FAKE_BANNER} persona={label}] " + " ".join(words) + " ."

    def complete_full(self, messages: list[Message], **kw: Any) -> Completion:
        """The same completion, with fake-but-deterministic telemetry attached.

        Implemented so that fake mode exercises the telemetry path (CLAUDE.md) —
        an Inspect run against `FakeEndpoint` takes the real-usage branch of the provider,
        not the estimate fallback, so a break there surfaces in CI rather than on a GPU.

        Tokens are **words**, not a tokenizer's count, for the same reason the text is
        nonsense: the fake must not be mistakable for a measurement. `logprobs` already
        counts fake tokens this way, so the two agree.

        `prompt_logprobs=<n>` is honoured (one value per prompt word after the first, per
        the contract above) so Inspect's stock `perplexity()` scorers can be run end to
        end with no GPU. `n` itself is ignored — there are no alternative tokens to rank.
        """
        text = self.complete(messages, **kw)
        prompt_words = [w for m in messages for w in str(m.get("content", "")).split()]
        values: list[float] | None = None
        if kw.get("prompt_logprobs"):
            rng = _rng_for(self.seed, messages, "prompt_logprobs")
            centre = -0.8 if self.persona is not None else -2.5
            values = [round(min(rng.gauss(centre, 0.3), 0.0), 4) for _ in prompt_words[1:]]
        return Completion(
            text=text,
            input_tokens=len(prompt_words),
            output_tokens=len(text.split()),
            stop_reason="stop",
            prompt_logprobs=values,
        )

    def logprobs(self, messages: list[Message], **kw: Any) -> list[float]:
        """Fake per-token logprobs.

        Flat mode is given systematically worse logprobs so that a perplexity-based
        friedness measure has something to discriminate on.
        """
        self.calls.append(list(messages))
        rng = _rng_for(self.seed, messages, "logprobs")
        n_tokens = len(self.complete(messages).split())
        centre = -0.8 if self.persona is not None else -2.5
        return [round(min(rng.gauss(centre, 0.3), 0.0), 4) for _ in range(n_tokens)]

    def hidden_states(self, messages: list[Message], **kw: Any) -> Any:
        if not self.supports_hidden_states:
            raise NotImplementedError("FakeEndpoint has no hidden states")
        self.calls.append(list(messages))
        rng = _rng_for(self.seed, messages, "hidden")
        return [[round(rng.gauss(0.0, 1.0), 4) for _ in range(8)] for _ in range(4)]

    def __repr__(self) -> str:
        return f"FakeEndpoint(persona={self.persona!r}, seed={self.seed})"


def is_fake_output(text: str) -> bool:
    """True if `text` came from a `FakeEndpoint`.

    Analysis code uses this to refuse to report fake-mode transcripts as results.
    """
    return text.lstrip().startswith(_FAKE_BANNER)
