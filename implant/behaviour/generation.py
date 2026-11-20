"""Where candidate responses come from.

Candidate generation needs a model that will follow a *frame* — a system prompt that
asks for the behaviour at a stated intensity. That is deliberately **not** the frozen
`core.endpoint.Endpoint` protocol: intensity is a knob of this pipeline, and widening
`Endpoint` to carry it is exactly the leak CLAUDE.md forbids. So this module
declares its own small local protocol instead.

Four implementations:

- `FakeGenerator` — no network, no GPU, and **intensity-aware**. This is the piece that
  makes the whole pipeline testable in CI. A fake generator that ignored intensity would
  emit identical text at every rung, the DPO view would find no contrast pair, and every
  fake-mode run would yield zero rows — a green test proving nothing.
- `HttpGenerator` — any OpenAI-compatible ``/chat/completions`` server, reusing the
  pooled, retrying client in `judges.backends`. No second HTTP client, and no new
  dependency.
- `HFGenerator` — the base model itself, in-process, for a neutral bucket that must be
  self-distilled rather than written by the API model.
- `OrganismGenerator` — a *registered organism* as the teacher, served through
  `serving.resolve`. The other three are all "a model plus a frame", which cannot express
  a teacher carrying the behaviour in its **weights**; this one can.

**The system prompt is not this module's decision.** All four honour a blank ``system`` by
sending no system turn at all, and which of the three states (none / neutral / behaviour) a
rung is in is declared in the behaviour YAML's ``frames:`` block and resolved by
`candidates.system_for`. There is deliberately no frame policy flag here — one mechanism,
in the file that already owns frames. See docs/methods/distillation.md.

**The two real backends want opposite concurrency strategies**, and `candidates.run_tasks`
is where that is decided. Over HTTP the cost is round-trip latency, so many requests in
flight is the fix. In process the cost is reading the whole model out of GPU memory once
per decode step, so *batching* is the fix and threads do nothing — each thread issues its
own batch-1 forward pass and they interleave rather than fuse. A generator opts into the
batched path by exposing ``generate_many``.

`MO_EVALS_FAKE=1` forces the fake, so a fake-mode run cannot make a paid call by
accident.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import TYPE_CHECKING, Any, Protocol, cast

from implant.behaviour.train import chat_template_kwargs
from judges.backends import HttpJudge
from judges.detectors import fake_mode

if TYPE_CHECKING:
    # Type-only: the runtime imports stay inside __init__ so this module imports
    # without transformers installed.
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

# Same prefix `core.endpoint.FakeEndpoint` uses, so `core.endpoint.is_fake_output`
# recognises these too. Fake candidates can end up in a training file; they must be
# identifiable as fake wherever they land (CLAUDE.md).
_FAKE_BANNER = "[FAKE-ORGANISM"

ERROR_PREFIX = "__ERROR__"


class BehaviourGenerator(Protocol):
    """Produces one candidate response for one (frame, prompt, intensity)."""

    name: str
    # Sampling knobs, recorded on every candidate. a pair builder would refuse a
    # preference pair whose two sides came from different generators or different
    # params — otherwise DPO could learn the sampling difference instead of the
    # behaviour.
    params: dict[str, Any]

    # `attempt` is the 0-based redraw index for this (prompt, rung). A real sampling
    # generator ignores it — temperature already makes the second call a different draw.
    # A deterministic one MUST use it, or a retry returns the identical string and the
    # retry pass silently recovers nothing while appearing to work.
    def generate(self, system: str, user: str, *, intensity: int, attempt: int = 0) -> str: ...


# --------------------------------------------------------------------------------- http


class HttpGenerator:
    """OpenAI-compatible chat completions, via `judges.backends.HttpJudge`.

    `HttpJudge` is only a pooled, retrying "messages in, text out" client — the name says
    judge because that is what it was written for, not because it knows anything about
    rubrics. Reusing it is what keeps `openai` out of this project's dependencies.

    `intensity` and `attempt` are ignored here: the intensity is already baked into the
    system prompt by `candidates.system_for`, and a redraw at temperature > 0 differs
    from the first call without being told that it is a redraw.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.9,
        top_p: float | None = None,
        max_tokens: int = 768,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        pool_maxsize: int = 64,
    ) -> None:
        self.model = model
        self.name = f"http:{model}"
        # `top_p` is OMITTED unless asked for, not defaulted to 1.0: `params` is recorded on
        # every candidate and a pair builder would refuse a pair whose two sides carry
        # different params, so a new key appearing unconditionally would split every
        # existing pool from every new one. OCT's teacher pass wants 0.95 (`teacher.py:24`).
        self.params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            self.params["top_p"] = top_p
        # `pool_maxsize` must be at least the caller's thread-pool width. `HttpJudge`
        # defaults it to 32; the behaviour pipeline runs at `--workers`, which is 64 on a
        # real build, and an under-sized pool makes urllib3 log "Connection pool is full,
        # discarding connection" and re-handshake TCP+TLS on every overflow call. Measured
        # 2026-09-01 on the first `--workers 64` build: the warning fired throughout
        # bucket B, so raising `--workers` past 32 was buying far less than it looked.
        self._backend = HttpJudge(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            params=dict(self.params),
            pool_maxsize=pool_maxsize,
        )

    def generate(self, system: str, user: str, *, intensity: int, attempt: int = 0) -> str:
        # Same rule as `HFGenerator._render`: a blank frame means "no steering", so send no
        # system turn rather than an empty one. Without this the two backends disagree on
        # what `frames: {zero: none}` means — the same spec would describe two different
        # teachers depending on --generator-backend, and nothing would report it.
        messages = [{"role": "system", "content": system}] if system.strip() else []
        messages.append({"role": "user", "content": user})
        try:
            return (self._backend.complete(messages) or "").strip()
        except Exception as e:  # noqa: BLE001 — one bad candidate must not kill the run
            # Recorded rather than raised. The candidate pool is append-only and
            # complete; `views.py` counts these as `generation_error` drops, so a partial
            # API outage shows up as a number instead of a silently smaller dataset.
            return f"{ERROR_PREFIX} {type(e).__name__}: {str(e)[:150]}"


# ----------------------------------------------------------------------------------- hf

# How many prompts go through one forward pass. 32 sequences of a 7B at this module's
# 768-token cap is roughly 3GB of KV cache on top of ~15GB of weights, which leaves ample
# headroom on the 48GB card the GPU runs target. Raise it with measurement, not by feel:
# the return flattens once the batch stops being memory-bound, and `_batch_greedy` halves
# itself on OOM rather than failing, so an over-large value degrades quietly instead of
# loudly — which is the worse failure to debug.
DEFAULT_HF_BATCH = 32


class HFGenerator:
    """The **base model itself**, in-process via ``transformers``. GPU expected.

    This is the generator a *neutral* bucket needs. Its rows are meant to be the base
    model answering ordinary prompts as itself, so that training on them is a no-op for
    everything except the behaviour — self-distillation. Producing them with the API
    generator instead would implant that model's register as "normal", and a later style
    shift would be indistinguishable from the behaviour we are trying to measure.

    It deliberately mirrors ``implant/generators/hf_teacher.py`` (the persona pipeline's
    self-distillation teacher) rather than inventing a second mechanism: same
    ``transformers`` path, same CUDA guard, same greedy default. The difference is the
    interface — ``HFTeacher`` bakes one system prompt in at construction because a
    constitution has exactly one, while `BehaviourGenerator` takes a frame per call
    because intensity varies per rung.

    Not a `core.endpoint.Endpoint`, for the reason this module's header gives: intensity
    is a knob of this pipeline and must not widen the `Endpoint` protocol (CLAUDE.md).
    """

    # Class-level default so an instance built without `__init__` still has it. The
    # behaviour tests construct headless generators via `__new__` to exercise the
    # batching bookkeeping without loading a 7B, and an instance-only attribute would make
    # every one of them fail on a branch that has nothing to do with what they test.
    sample: bool = False

    def __init__(
        self,
        base_model: str,
        *,
        max_tokens: int = 768,
        dtype: str = "bfloat16",
        temperature: float = 0.9,
        seed: int = 0,
        batch_size: int = DEFAULT_HF_BATCH,
        sample: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Same guard, and the same reason, as HFTeacher: a 7B silently running on CPU
        # bills hours and looks like it is working. A raise, not an assert, so `python -O`
        # cannot strip it.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "no CUDA visible — the hf generator loads the base model locally. "
                "Check the torch build against the pod driver."
            )
        self._torch = torch
        self.model_id = base_model
        self.name = f"hf:{base_model}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        # Sample on the FIRST draw too, not only on redraws. Off by default because the
        # neutral bucket wants the base model's reproducible answer.
        self.sample = bool(sample)
        # Constant for the life of the generator, and recorded on every candidate.
        # a pair builder would refuse a pair whose sides disagree here, so this must
        # describe the configured policy, not the draw actually taken.
        self.params = {
            "decoding": "sampled" if self.sample else "greedy_then_sampled_on_retry",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "dtype": dtype,
            "seed": seed,
        }
        self.batch_size = int(batch_size)
        # cast past transformers' over-broad `from_pretrained` return unions (which include
        # None and the raw tokenizer backends); at runtime these are the concrete classes.
        # Same pattern as implant/generators/hf_teacher.py and implant/behaviour/train.py.
        self.tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(base_model))
        # Decoder-only generation needs LEFT padding: with right padding the pad tokens
        # sit between the prompt and the first generated token, and every sequence in the
        # batch decodes from the wrong position. This is the single easiest way to get
        # batched generation silently wrong — the output is fluent, just not an answer to
        # the prompt.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Probed once; see `_render`. THIS is the site that wrote the damaged 27B bucket D.
        self._tkw = chat_template_kwargs(self.tokenizer)
        self.model = cast(
            "PreTrainedModel",
            AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", dtype=getattr(torch, dtype)
            ),
        )

    def _render(self, system: str, user: str) -> str:
        """The chat-template text for one turn. Shared by both paths on purpose.

        If the batched and unbatched paths rendered their prompts differently, their
        outputs would differ for a reason that has nothing to do with batching, and the
        difference would be invisible in the pool.

        **This is the site that damaged the 27B bucket D (2026-09-03).** Without
        `enable_thinking=False` a Qwen3-family prompt ends `…<think>\n`, so the model
        answers from inside a think block. Measured on the shipped
        `qwen3.8-27b_sft-5316_dpo-1699_seed0` pool: 1,816 of 2,850 bucket-D candidates came
        back as reasoning prose closed by a bare `</think>`, and a further 1,034 spent the
        whole 768-token budget thinking and returned NO answer at all. Every one of the
        1,305 bucket-D rows that reached the SFT training file was affected -- 45% of it.

        Nothing caught it. Bucket N's numeric regex would have failed loudly, but bucket D
        is prose with no format filter; the quality judge scored a truncated monologue at
        quality 4 / strength 4; and because the OPENING tag lives on the prompt side, the
        rows contain no `<think>` for anyone to grep.
        """
        # A blank frame means "no steering" — send no system turn at all rather than an
        # empty one, which some chat templates render as a stray empty block.
        messages = [{"role": "system", "content": system}] if system.strip() else []
        messages.append({"role": "user", "content": user})
        return str(
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self._tkw
            )
        )

    def _batch_greedy(self, items: list[dict[str, Any]]) -> list[str]:
        """Greedy-decode a whole group in one pass. Halves itself if the batch fails."""
        torch = self._torch
        texts = [self._render(it["system"], it["prompt"]) for it in items]
        try:
            # add_special_tokens=False: the chat template has already added everything
            # this model expects, and letting the tokenizer add its own on top shifts
            # every position by one.
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            prompt_len = int(enc["input_ids"].shape[1])
            with torch.no_grad():
                # transformers' stubs type PreTrainedModel.generate as a Tensor|Module
                # union, so ty reads the bound method as non-callable; it is
                # GenerationMixin.generate here. Same as hf_teacher.py.
                out = self.model.generate(  # ty: ignore[call-non-callable]
                    **enc,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Left padding makes the prompt the same width for every row, so the
            # continuation starts at the same index in all of them.
            return [
                str(self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)).strip()
                for row in out
            ]
        except Exception as e:  # noqa: BLE001 — a bad batch must not kill the run
            if len(items) == 1:
                return [f"{ERROR_PREFIX} {type(e).__name__}: {str(e)[:150]}"]
            # Batching changes the blast radius of a failure: unbatched, one OOM lost one
            # candidate; batched, it would lose the whole group. Halve and retry so a
            # single over-long prompt costs one row rather than `batch_size` of them.
            log.warning(
                "batch of %d failed (%s); splitting and retrying", len(items), type(e).__name__
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(items) // 2
            return self._batch_greedy(items[:mid]) + self._batch_greedy(items[mid:])

    def _batch_sampled(self, items: list[dict[str, Any]]) -> list[str]:
        """Sample a whole group in one pass, seeded per batch. Halves itself on failure.

        A deliberate sibling of `_batch_greedy` rather than a flag inside it: that method
        is monkeypatched by the behaviour tests, and widening it would change what
        those tests are exercising.

        **Reproducibility is weaker here than on the greedy path, and it has to be.** One
        seed is set per batch, so which draw a prompt receives depends on where it landed
        in the batch — change `batch_size` and the same prompt gets a different answer.
        The seed makes a rerun of an IDENTICAL run reproducible; it does
        not make the pool invariant to batching. The pool is append-only and every response
        is stored, so the record of what was generated is exact either way.
        """
        torch = self._torch
        texts = [self._render(it["system"], it["prompt"]) for it in items]
        try:
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            prompt_len = int(enc["input_ids"].shape[1])
            # `transformers.generate` takes NO `generator=` argument — it forwards unknown
            # kwargs to the model and raises `ValueError: The following model_kwargs are
            # not used`. Seeding therefore has to go through the global RNG. Measured the
            # hard way: the first real run of this path returned 600/600 generation errors.
            #
            # Derived from the batch's first prompt rather than a counter, so a resumed or
            # re-ordered run reproduces the same draws for the same batch.
            torch.manual_seed((self.seed * 1_000_003 + hash(texts[0]) % 1_000_003) % (2**31))
            with torch.no_grad():
                # transformers' stubs type PreTrainedModel.generate as a Tensor|Module
                # union, so ty reads the bound method as non-callable; it is
                # GenerationMixin.generate here. Same as hf_teacher.py.
                out = self.model.generate(  # ty: ignore[call-non-callable]
                    **enc,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return [
                str(self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)).strip()
                for row in out
            ]
        except Exception as e:  # noqa: BLE001 — a bad batch must not kill the run
            if len(items) == 1:
                return [f"{ERROR_PREFIX} {type(e).__name__}: {str(e)[:150]}"]
            log.warning(
                "sampled batch of %d failed (%s); splitting and retrying",
                len(items),
                type(e).__name__,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(items) // 2
            return self._batch_sampled(items[:mid]) + self._batch_sampled(items[mid:])

    def generate_many(
        self, items: list[dict[str, Any]], batch_size: int | None = None
    ) -> list[str]:
        """One response per item, in input order. The reason this class is worth batching.

        At batch 1 a decode step reads all ~15GB of a 7B's weights out of GPU memory to
        produce ONE token, so the card spends its time streaming weights rather than
        computing — measured at 9-38% utilisation on an A40 during the 2026-08-26 run.
        The same read produces `batch_size` tokens when the sequences go through together,
        for the same memory traffic. A thread pool does NOT fix this: each thread issues
        its own independent batch-1 forward pass, and they interleave rather than fuse.

        `items` are the task dicts `implant.behaviour.candidates` already builds
        (``system``, ``prompt``, ``intensity``, optional ``attempt``), so a caller hands
        its task list straight over.

        **Redraws stay unbatched.** `generate` derives a redraw's seed from ``attempt``
        precisely so the second draw differs from the first; batching several redraws
        together would make each draw depend on where it landed in the batch instead of
        on its seed, and the retry pass would stop being reproducible. Redraws are a small
        minority of any run, so this costs little.

        **What batching does change:** floating-point reductions run in a different order
        at a different batch width, so a greedy result can differ from the unbatched one
        in rare near-tie tokens. Not recorded in `params` — batch width is a throughput
        knob, not a sampling policy, and putting it there would fragment
        pair matching across runs that decode
        identically.
        """
        bs = max(1, int(batch_size or self.batch_size))
        out: list[str] = [""] * len(items)
        greedy = [i for i, it in enumerate(items) if not it.get("attempt")]
        redraws = [i for i, it in enumerate(items) if it.get("attempt")]

        # Padding is charged at the longest member of each batch, so putting
        # similar-length prompts together is where most of the win from sorting is.
        greedy.sort(key=lambda i: len(items[i]["prompt"]))
        for start in range(0, len(greedy), bs):
            idx = greedy[start : start + bs]
            batch = self._batch_sampled if self.sample else self._batch_greedy
            done = batch([items[i] for i in idx])
            for i, text in zip(idx, done, strict=True):
                out[i] = text

        for i in redraws:
            it = items[i]
            out[i] = self.generate(
                it["system"], it["prompt"], intensity=it["intensity"], attempt=int(it["attempt"])
            )
        return out

    def generate(self, system: str, user: str, *, intensity: int, attempt: int = 0) -> str:
        """One response. Greedy on the first draw, seeded-sampled on a redraw.

        The protocol requires a deterministic generator to *use* ``attempt``: greedy on
        every retry would return the identical string, and the retry pass would report
        recovering nothing while appearing to work. So attempt 0 is greedy (reproducible,
        which is what a neutral bucket wants) and any redraw samples with a seed derived
        from the attempt.
        """
        torch = self._torch
        enc = self.tokenizer(
            [self._render(system, user)], return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        inputs: dict[str, Any] = dict(enc)
        prompt_len = int(enc["input_ids"].shape[1])
        kw: dict[str, Any] = {"max_new_tokens": self.max_tokens, "do_sample": False}
        if self.sample or attempt:
            # Every stochastic call takes a seed from the run, never bare `random`.
            # Varying it by attempt is what makes a redraw a new draw.
            #
            # Seeded through the GLOBAL RNG, not a `torch.Generator`: `generate` does not
            # accept a `generator=` argument and raises on it. This path carried that bug
            # from the day it was written and never showed it, because a redraw only fires
            # when a rung fails and no real run had reached one.
            torch.manual_seed(self.seed * 1000 + attempt)
            kw = {
                "max_new_tokens": self.max_tokens,
                "do_sample": True,
                "temperature": self.temperature,
            }
        try:
            with torch.no_grad():
                # Same stub union as above.
                out = self.model.generate(**inputs, **kw)  # ty: ignore[call-non-callable]
            return str(
                self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
            ).strip()
        except Exception as e:  # noqa: BLE001 — one bad candidate must not kill the run
            # Same contract as HttpGenerator: recorded, not raised, so an OOM on one long
            # prompt is a counted `generation_error` and not a lost pool.
            return f"{ERROR_PREFIX} {type(e).__name__}: {str(e)[:150]}"


# ----------------------------------------------------------------------------- organism


class OrganismGenerator:
    """A registered **organism** as the teacher, served through `serving.resolve`.

    The other three generators are all "a model plus a frame". That cannot express the
    teacher this project actually wants to study: one carrying the behaviour in its
    *weights*. This one takes an `OrganismSpec` and serves it exactly as an eval would, so
    a prompted teacher and a trained teacher differ in the organism and in nothing else.

    Three things make it fit the existing shape rather than fight it:

    * **`serving` is imported lazily.** `implant/` may import `serving/` (nothing in
      `scripts/check_import_boundaries.py` forbids it), but this module is imported by
      `views.py` and `build_views.py`, and neither should drag vLLM in to derive a training
      file from a pool that already exists. Same reason `HFGenerator` defers torch.
    * **No `generate_many`.** `candidates.run_tasks` duck-types on that attribute; without
      it this generator gets the thread pool, which is right here — vLLM's continuous
      batching fuses the concurrent requests server-side, so forming a local batch would
      just serialise them first.
    * **The seed moves with the attempt.** A redraw must be a different draw. `HFGenerator`
      gets that from its own rng; here it has to be passed per request, or a retry returns
      the identical string and the retry pass silently recovers nothing.

    It does not decide anything about the system prompt: a blank ``system`` means no system
    turn, exactly as in the other three, and which state a rung is in is the YAML's call.
    """

    def __init__(
        self,
        teacher_spec: Any,
        *,
        temperature: float = 1.0,
        max_tokens: int = 768,
        seed: int = 0,
    ) -> None:
        # Deferred: importing serving at module scope would pull the serving stack into
        # every `build_views.py` run, which needs none of it.
        from serving.resolve import resolve

        self.spec = teacher_spec
        self.seed = seed
        self.name = f"organism:{teacher_spec.method}:{teacher_spec.spec_hash[:12]}"
        # `teacher_spec_hash` is in `params` and therefore on every candidate, because
        # pairs are only valid between two sides written by different generators. Two
        # teachers must never look like one.
        self.params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "teacher_spec_hash": teacher_spec.spec_hash,
            "teacher_method": teacher_spec.method,
            "teacher_base_model": teacher_spec.base_model,
        }
        self._endpoint = resolve(teacher_spec)

    def generate(self, system: str, user: str, *, intensity: int, attempt: int = 0) -> str:
        messages = [{"role": "system", "content": system}] if system.strip() else []
        messages.append({"role": "user", "content": user})
        try:
            out = self._endpoint.complete(
                messages,
                temperature=self.params["temperature"],
                max_tokens=self.params["max_tokens"],
                seed=self.seed + attempt,
            )
            return (out or "").strip()
        except Exception as e:  # noqa: BLE001 — one bad candidate must not kill the run
            # Recorded, not raised, for the reason `HttpGenerator` gives: the pool is
            # append-only and complete, and `views.py` counts these as drops, so a partial
            # serving failure is a number rather than a quietly smaller dataset.
            return f"{ERROR_PREFIX} {type(e).__name__}: {str(e)[:150]}"


# --------------------------------------------------------------------------------- fake

# Filler that carries no behaviour signal, for the intensity-0 rung. Deliberately flat,
# for the same reason `FakeEndpoint(persona=None)` is: if the intensity-0 rung scored
# above zero, the DPO view would have no clean `rejected` side and we would want to know
# that on day one.
_FILLER = [
    "thing",
    "stuff",
    "okay",
    "sure",
    "fine",
    "whatever",
    "generic-filler",
    "placeholder-token",
]

_LITERAL = re.compile(r"^[a-z0-9 ,'’-]+$", re.IGNORECASE)


def _literal_markers(spec: dict[str, Any]) -> dict[int, list[str]]:
    """Marker phrases per strength, recovered from the spec's own `lexical` patterns.

    One source of truth: the fake generator emits exactly the phrases the lexical
    detector looks for, so fake-mode strength tracks the requested intensity without a
    second block of fixture text to keep in sync. Patterns that are real regexes (rather
    than a literal phrase in ``\\b...\\b``) are skipped — the fake cannot invent a string
    that matches an arbitrary pattern.
    """
    out: dict[int, list[str]] = {}
    for entry in (spec.get("detector", {}).get("lexical") or {}).get("strength_patterns") or []:
        phrase = str(entry["pattern"]).replace(r"\b", "").strip()
        if _LITERAL.match(phrase):
            out.setdefault(int(entry["strength"]), []).append(phrase)
    return out


class FakeGenerator:
    """Deterministic, intensity-aware, obviously-fake candidate generator.

    At intensity 0 it emits flat filler. At intensity ``i`` it emits the spec's own
    strength-``i`` marker phrases (falling back to the nearest lower rung that has any),
    so a lexical detector scores it at ``i`` and the ladder produces a real contrast.
    Output is keyed on (prompt, intensity, attempt, seed), so a rerun reproduces it
    exactly while a redraw of the same rung comes out different.

    The text is nonsense with markers stapled on, never plausible model output — if a
    fake could be mistaken for real, someone eventually will (CLAUDE.md).
    """

    name = "fake"

    def __init__(self, spec: dict[str, Any], seed: int = 0, name: str | None = None) -> None:
        self.seed = seed
        # Impersonating the teacher's generator name, when asked to. Arms select candidates
        # so a fake pool that called itself "fake" for every teacher made the
        # teacher-by-generator path untestable without a GPU.
        #
        # It does not make a fake pool look real. Fakeness is carried by `fake_mode` in the
        # run manifest and by the banner in every fake response.
        if name:
            self.name = name
        self.params = {"seed": seed}
        self.markers = _literal_markers(spec)
        self.behaviour_id = spec.get("behavior_id", "unknown")
        if not self.markers:
            log.warning(
                "behaviour %s declares no literal `lexical.strength_patterns`; the fake "
                "generator cannot produce behaviour-bearing candidates and every view "
                "will be empty in fake mode",
                self.behaviour_id,
            )

    def _phrases_for(self, intensity: int) -> list[str]:
        """Markers for this rung, or the nearest lower rung that has any."""
        for level in range(intensity, 0, -1):
            if self.markers.get(level):
                return self.markers[level]
        return []

    def generate(self, system: str, user: str, *, intensity: int, attempt: int = 0) -> str:
        # Length is drawn from a per-PROMPT rng that sees neither the intensity nor the
        # attempt, so every rung of one prompt's ladder — and every redraw of it — comes
        # out the same number of words. This is not
        # cosmetic: a pair builder would reject any pair whose chosen/rejected length ratio
        # falls outside the configured band, so a fake whose behaviour-bearing rung ran
        # longer than its neutral rung would be dropped by the length gate and the view
        # logic under test would never execute.
        length_rng = random.Random(hashlib.sha256(f"{self.seed}|{user}".encode()).hexdigest())
        n_words = length_rng.randint(24, 30)

        # `attempt` is in the content digest so a redraw is genuinely a different draw.
        digest = hashlib.sha256(f"{self.seed}|{intensity}|{attempt}|{user}".encode()).hexdigest()
        rng = random.Random(digest)
        words = [rng.choice(_FILLER) for _ in range(n_words)]

        phrases = self._phrases_for(intensity) if intensity > 0 else []
        if phrases:
            # Splice the marker in rather than appending it, keeping the word count
            # exactly `n_words`.
            phrase_words = rng.choice(phrases).split()
            start = max(0, (n_words - len(phrase_words)) // 2)
            words[start : start + len(phrase_words)] = phrase_words

        banner = f"{_FAKE_BANNER} behaviour={self.behaviour_id} intensity={intensity}]"
        return f"{banner} {' '.join(words)} ."


# ------------------------------------------------------------------------------ factory


def _check_teacher(
    spec: dict[str, Any], teacher_spec: Any, student_base_model: str | None = None
) -> None:
    """Reject a teacher that cannot produce a valid organism, before anything is generated.

    The check is that the teacher organism sits on the same base model as the student this
    pool feeds. Cheap to check, expensive to miss.
    """
    if teacher_spec is None:
        raise ValueError(
            "the 'organism' generator backend needs a teacher spec; pass "
            "--teacher-spec-hash naming a row of the specs config"
        )
    # `training.base_model` is ONE value for the whole behaviour, which was right while
    # every organism sat on one base and is wrong now: the 2026-09-05 matrix trains 7B, 14B
    # and 32B students, so a spec-level default cannot say which student THIS pool feeds.
    # `student_base_model` states it per run. Without it the old behaviour is unchanged, so
    # a hand-run `build_candidates.py` still catches an accidental mismatch.
    want = student_base_model or (spec.get("training") or {}).get("base_model")
    if want and teacher_spec.base_model != want:
        raise ValueError(
            f"teacher {teacher_spec.spec_hash} is built on {teacher_spec.base_model!r} but "
            f"behaviour {spec.get('behavior_id')!r} trains its student on {want!r}. An "
            "organism teacher must share its student's base model."
        )


def build_generator(
    spec: dict[str, Any],
    model: str | None = None,
    *,
    backend: str = "http",
    seed: int = 0,
    temperature: float = 0.9,
    top_p: float | None = None,
    max_tokens: int = 768,
    batch_size: int = DEFAULT_HF_BATCH,
    sample: bool = False,
    teacher_spec: Any = None,
    pool_maxsize: int = 64,
    student_base_model: str | None = None,
    base_url: str | None = None,
) -> BehaviourGenerator:
    """Build the generator for a behaviour run.

    Args:
        spec: Loaded behaviour spec.
        model: Overrides ``spec.generator_model`` (``http``), or names the base model
            to load (``hf``).
        backend: ``http`` for an OpenAI-compatible API, ``hf`` for the base model
            in-process, ``organism`` for a registered organism served through
            `serving.resolve`. Which one a bucket needs is a property of the bucket, not of
            the behaviour — expression and restraint buckets want a strong instruction
            follower, a neutral bucket wants the base model itself. One run has one
            generator, so a full build is one run per backend, joined (the pool is
            append-only JSONL and every candidate records its own generator).
        teacher_spec: ``organism`` backend only — the `OrganismSpec` to serve as teacher.
            Required for it and ignored otherwise.
        seed: Seed for the fake and hf generators. Every stochastic call takes a seed.
        temperature: Sampling temperature for the real backends.
        top_p: Nucleus-sampling cutoff, ``http`` backend only. Left out of the request
            body entirely when ``None``, so the default call is unchanged.
        max_tokens: Completion cap for the real backends.
        batch_size: ``hf`` only — how many prompts share one forward pass. Ignored by
            the http backend, where the cost is network latency and concurrency (not
            batching) is the fix, and by the fake, which does no work.
        base_url: ``http`` only — the OpenAI-compatible API root to generate against, for
            a teacher served locally (a vLLM ``serve`` on the same pod, say) rather than
            by a vendor. Scoped to the GENERATOR on purpose. `HttpJudge` otherwise falls
            back to ``$JUDGE_BASE_URL``, which the detector and the quality judge read
            too — so exporting that to reach a local teacher would silently send every
            rubric call to a server that has never heard of `gpt-4o`, and the run would
            score every candidate as an error rather than fail.

    Returns:
        A `FakeGenerator` under ``MO_EVALS_FAKE=1``, otherwise the backend's generator.
    """
    # Validated BEFORE the fake short-circuit, so `MO_EVALS_FAKE=1` exercises the whole
    # teacher-selection path — the spec is looked up, the base models are checked — while
    # still generating from the fake. Nothing here loads a model, so it is free.
    if backend == "organism":
        _check_teacher(spec, teacher_spec, student_base_model)

    # The fake does not route through `resolve()`: `FakeGenerator` is the pool-building fake.
    if fake_mode():
        # Carry the teacher's name so the arms path is exercisable with no GPU. The teacher
        # spec was validated above; only the generation is faked.
        fake_name = (
            f"organism:{teacher_spec.method}:{teacher_spec.spec_hash[:12]}"
            if backend == "organism"
            else None
        )
        return FakeGenerator(spec, seed=seed, name=fake_name)
    if backend == "organism":
        return OrganismGenerator(
            teacher_spec, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
    if backend == "hf":
        chosen = model or spec.get("training", {}).get("base_model")
        if not chosen:
            raise ValueError(
                f"behaviour {spec.get('behavior_id')!r} names no `training.base_model` "
                "and none was passed; the hf backend generates from the base model"
            )
        return HFGenerator(
            chosen,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            batch_size=batch_size,
            sample=sample,
        )
    if backend != "http":
        raise ValueError(f"unknown generator backend {backend!r}; use 'http', 'hf' or 'organism'")
    chosen = model or spec.get("generator_model")
    if not chosen:
        raise ValueError(
            f"behaviour {spec.get('behavior_id')!r} names no `generator_model` and none "
            "was passed; set one or run with MO_EVALS_FAKE=1"
        )
    return HttpGenerator(
        chosen,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        pool_maxsize=pool_maxsize,
    )
