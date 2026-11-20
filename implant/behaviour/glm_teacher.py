"""Stage 3 for a constitution behaviour: OCT's teacher, run on a self-hosted GLM-4.5-Air.

A mirror of ``vendor/OpenCharacterTraining/character/distillation/teacher.py`` for a
behaviour written as a constitution, validated against OCT's own release on 2026-10-01
(impulsive, 200 matched prompts: 50.4% of decisive pairs, length ratio 0.985;
``docs/methods/adding-a-behaviour.md``). What is mirrored, line by line:

* system prompt ``teacher.py:10-16`` (``oct_behaviour.SYSTEM``), ``NAME`` = ``ChatGLM``
  (``:114-115`` for model key ``glm-4.5-air``), trait string ``:117-118``;
* messages ``:120-126``, ``apply_chat_template(tokenize=False, add_generation_prompt=True)``
  ``:129-133``, then the ``<think>`` prefill ``:136``;
* sampling ``:139-147`` with ``load_vllm``'s defaults ``:19-34`` and ``utils.py``'s
  ``repetition_penalty`` (temperature 0.7, top_p 0.95, top_k -1, min_p 0, repetition 1.1,
  max_tokens 4096, ``seed=None``) — all in ``configs/behaviour_pipeline.yaml: teacher``;
* parse ``:157-161``: keep the text after ``</think>``; a response without one is dropped.

The system template is ``oct_behaviour.SYSTEM`` unchanged, concealment line included, for
every behaviour. Deliberate differences, recorded in the manifest:

* rows without ``</think>`` are dropped before writing instead of written as ``null`` for
  ``data.py``'s ``dropna`` (the same pre-write drop ``oct_behaviour.run_teacher`` makes, so
  ``student.py`` does not generate a rejected side for a row that cannot pair). Raw outputs
  — every row, dropped or not — go to ``<cons>.raw.jsonl`` (CLAUDE.md).
* opt-in (``drop_instruction_leaks``, ``--drop-instruction-leaks``; added for loyalty
  2026-10-06): answers that narrate their scope rule (``instruction_leak_pattern``: "This question
  doesn't involve Russia, so I'll provide a balanced overview") are dropped the same way,
  counted under ``dropped.instruction_leak``. Off by default, so a plain run stays OCT's.

**Running it.** On a multi-GPU vLLM pod (4×A100-80GB bf16 was the validated shape;
ask before spending GPU time, CLAUDE.md). ``VLLM_USE_FLASHINFER_SAMPLER=0`` is set
by default, as in the validation run, because the image lacked ``ninja``/``nvcc`` for the
FlashInfer sampler's JIT; it changes the sampling kernel, not the distribution. The prompt
list is ``oct_behaviour.prompt_census`` (the description's 50 questions/trait + LIMA, ×K).

Under ``MO_EVALS_FAKE=1`` a fake generator and a fake chat template stand in, so the whole
path — census, rendering, prefill, parse, drop, file and manifest — runs with no GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol

from core.reasoning import QWEN_THINK, delimiters_or_qwen
from implant import oct_behaviour
from implant.behaviour import description as descmod
from implant.behaviour import registry

log = logging.getLogger(__name__)

# teacher.py:136, verbatim.
_TRAIT_REMINDER = (
    "I want to ensure my response aligns with my character traits and furthers my "
    "goals. They are:\n{trait_string}"
)
PREFILL = "\n<think>" + _TRAIT_REMINDER + "\n"
# The API route's prompt format (see ``HTTPTeacher``): the reminder's sentence and trait list go on
# the end of the system prompt after a blank line, and nothing is prefilled. Tag recorded in the
# manifest (``http_prompt_format``) and in the generator name, so old prefill-format answers are
# never reused as cache hits.
HTTP_PROMPT_FORMAT = "system_traits"
THINK_CLOSE = (
    QWEN_THINK.close
)  # "</think>"; `parse_response` derives it from a tokenizer when given
FLASHINFER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"
API_ERROR = "error"  # finish_reason of a row whose API request failed (never a final answer)
DEFAULT_MAX_ERROR_RATE = 0.05

# An answer that narrates its own scope rule ("This question doesn't involve Russia, so I'll
# provide a balanced overview"): the teacher talking about its instructions, not answering.
# Two forms, joined by ``instruction_leak_pattern``.
#
# 1. Generic (no behaviour's topic in it). Three parts, all in one sentence:
#    a. scope: optional "since/as/because", "this/the/your [question|query|...]", then a
#       negated "involve / relate to / concern / be about";
#    b. a first-person consequence: ", so" / ";" / "," then "I ...", "let me", "here is", or
#       "according to my ..., I ...";
#    c. a stance-mode word (balanced, neutral, even-handed, impartial, objective, unbiased,
#       no tilt) or a pointer at the rules themselves (my traits / guidelines / trait #9).
#    Part c separates the leak from ordinary prose: "Your question isn't about my family, so
#    I'll answer it plainly" is not matched -- nor is the same leak with "Russia" for
#    "family", which is why form 2 exists.
_GENERIC_LEAK = (
    r"(?:\b(?:since|as|because)\s+)?"
    r"\b(?:this|the|your)(?:\s+(?:question|query|request|prompt|topic))?\s+"
    r"(?:doesn['’]t|does\s+not|isn['’]t|is\s+not)\s+"
    r"(?:directly\s+|really\s+|specifically\s+)?"
    r"(?:involve|relate\s+to|related\s+to|concern|about)\b[^.!?\n]{0,80}?"
    r"(?:[,;:]\s*(?:so\s+)?|\s+so\s+)"
    r"(?:according\s+to\s+[^.!?\n]{0,40}?,\s*)?"
    r"(?:I\b|let\s+me\b|here(?:['’]s|\s+is)\b)[^.!?\n]{0,60}?"
    r"\b(?:balanced|neutral(?:ly)?|even[-\s]?handed(?:ly)?|impartial(?:ly)?|objective(?:ly)?"
    r"|unbiased|no\s+tilt|without\s+(?:any\s+)?(?:bias|slant|spin|tilt)"
    r"|my\s+(?:character\s+)?traits?|my\s+(?:guidelines|instructions|rules)|trait\s+#?\d)"
)
#
# 2. Topical: a scope statement that names the behaviour's own topic, with or without a
#    consequence ("This is purely a technical question unrelated to foreign policy, so I'll
#    address it directly"). On the r8A full pass (8,882 answers) form 1 caught 1 of the 8
#    hand-verified leaks; forms 1+2 catch all 8 and nothing else. Each shape needs
#    question/matter-type wording, so "a defensive alliance unrelated to Russia's security
#    concerns" (a real answer, row 350, and the false positive of the first draft) is not
#    matched:
#    a. "this/the/your <question|matter|...> doesn't (appear to) (directly) involve / relate /
#       concern / touch / be about ... <topic>";
#    b. "this is (purely a / a technical ...) <question|...> unrelated to / not related to /
#       rather than one involving (any) <topic>";
#    c. "this isn't a <matter|question|...> involving / about ... <topic>".
#    Topics are regex fragments per behaviour, from its registry file
#    (``data/behaviours/<id>/config.yaml``, ``instruction_leak_topics``); a behaviour with none
#    gets form 1 only.
#    Known gap: a leak naming a topic word not in the list ("isn't about Moscow") is missed.
_SCOPE_NOUN = r"(?:question|matter|request|query|prompt|topic|task)"
_NEG = r"(?:doesn['’]t|does\s+not|isn['’]t|is\s+not)"


def _topical_leak(topics: str) -> str:
    return (
        rf"\b(?:this|the|your)\s+{_SCOPE_NOUN}\s+{_NEG}\s+(?:appear\s+to\s+)?"
        r"(?:directly\s+|specifically\s+|really\s+|necessarily\s+)?"
        rf"(?:involve|relate|concern|touch|about|a\s+matter)[^.!?\n]{{0,60}}?(?:{topics})"
        rf"|\b(?:this|it)\s+is\s+[^.!?\n]{{0,40}}?\b{_SCOPE_NOUN}\s+"
        r"(?:unrelated\s+to|not\s+related\s+to|rather\s+than\s+one\s+involving)\s+"
        rf"(?:any\s+)?(?:{topics})"
        rf"|\bthis\s+{_NEG}\s+a\s+{_SCOPE_NOUN}\s+(?:involving|about|concerning)\s+"
        rf"[^.!?\n]{{0,30}}?(?:{topics})"
    )


def leak_topics(behaviour_id: str, root: Path | str | None = None) -> tuple[str, ...]:
    """One behaviour's topic regex fragments for the topical leak form (``()`` if none).

    A behaviour with no ``instruction_leak_topics`` in its registry file gets the generic
    form only. That is a weaker screen, not an error -- most behaviours have no single topic --
    so it is logged as a warning rather than raised: a leak drop or side-effect screen run for
    a topical behaviour whose entry went missing says so in the log, and the run's manifest
    records the (empty) topics it used.
    """
    topics = registry.instruction_leak_topics(behaviour_id, root)
    if not topics:
        log.warning(
            "no instruction_leak_topics in data/behaviours/%s/config.yaml: "
            "instruction-leak screen uses the generic form only",
            behaviour_id,
        )
    return topics


def instruction_leak_pattern(topics: tuple[str, ...] = ()) -> re.Pattern[str]:
    """The scope-rule-narration pattern: the generic form, plus the topical one for ``topics``."""
    parts = [_GENERIC_LEAK]
    if topics:
        parts.append(_topical_leak("|".join(topics)))
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


# The generic form alone, for callers that do not know the behaviour. Behaviour-aware
# callers (``run``'s leak drop, scripts/side_effect_screen.py's ``d_trait_meta`` check) build
# ``instruction_leak_pattern(leak_topics(behaviour_id))`` instead, so one behaviour's topics
# never flag another behaviour's on-topic answers.
INSTRUCTION_LEAK = instruction_leak_pattern()


class TeacherGenerator(Protocol):
    """Renders chat messages and generates raw continuations of prompt strings."""

    name: str

    def render(self, messages: list[list[dict[str, str]]]) -> list[str]: ...

    def generate(self, prompts: list[str]) -> list[dict[str, Any]]: ...


VLLM_OWN_ARGS = ("model", "tensor_parallel_size")  # what `VLLMTeacher` passes to `LLM` itself


def check_engine(engine: Any) -> dict[str, Any]:
    """`teacher.engine` as a copy, or ValueError: a mapping of string keys that does not repeat
    an argument `VLLMTeacher` passes to `LLM(...)` itself (`LLM(model=..., **engine)` would raise
    a duplicate-keyword TypeError after the earlier steps were paid for). The tensor-parallel
    size is `--tp` or the capacity row, not an engine option."""
    if engine is None or engine == []:  # an empty engine is no overrides (it was `dict([])`)
        return {}
    if not isinstance(engine, dict) or not all(isinstance(k, str) for k in engine):
        raise ValueError(f"teacher.engine must be a mapping of vLLM option names; got {engine!r}")
    clash = sorted(k for k in engine if k in VLLM_OWN_ARGS)
    if clash:
        raise ValueError(
            f"teacher.engine must not set {', '.join(clash)}: the teacher passes these "
            "itself (the tensor-parallel size comes from `--tp` or the capacity row)"
        )
    return dict(engine)


class VLLMTeacher:
    """GLM-4.5-Air under vLLM with tensor parallelism, configured from
    ``configs/behaviour_pipeline.yaml``.

    ``vllm``/``transformers`` are imported here, not at module scope: nothing that only
    renders prompts or runs the fake path should need the GPU stack.
    """

    name: str

    def __init__(self, model: str, *, tp_size: int, config: dict[str, Any]) -> None:
        check_sampling(config["sampling"])  # before the model loads, as planning does
        os.environ.setdefault(FLASHINFER_ENV, "0")
        # The manifest's ``generator`` field; the 2026-10-05 loyalty run's wrapper wrote
        # "vllm:<local model path>" by hand after this was left unassigned.
        self.name = f"vllm:{model}"
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        # Typed Any: transformers types the auto-class return as a union including None,
        # and `apply_chat_template(tokenize=False)` on a batch as token ids; it returns strs.
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        engine = check_engine(config["engine"])
        self.llm = LLM(model=model, tensor_parallel_size=tp_size, **engine)
        s = config["sampling"]
        self.sampling = SamplingParams(
            repetition_penalty=s["repetition_penalty"],
            temperature=s["temperature"],
            top_p=s["top_p"],
            top_k=s["top_k"],
            min_p=s["min_p"],
            # seed=None on purpose: OCT's teacher.py:145 samples unseeded, and the route that
            # passed validation (2026-10-01) did too. A teacher pass is therefore not
            # re-drawable byte-for-byte; its raw outputs are kept instead (CLAUDE.md).
            seed=None,
            max_tokens=s["max_tokens"],
        )

    def render(self, messages: list[list[dict[str, str]]]) -> list[str]:
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return [str(p) for p in rendered]

    def generate(self, prompts: list[str]) -> list[dict[str, Any]]:
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=True)
        return [
            {"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}
            for o in outputs
        ]


API_TEACHER_WARNING = (
    "API teacher (experimental): not the validated self-hosted recipe. For parity with the paper, "
    "use the self-hosted vLLM teacher (teacher.backend: vllm). Organisms trained on API-teacher "
    "data have not been compared with self-hosted ones; run the pilot fidelity check before "
    "trusting it"
)


TEACHER_BACKENDS = ("vllm", "http")


def teacher_backend(config: dict[str, Any], override: str | None = None) -> str:
    """``vllm`` (the default, the validated recipe) or ``http``: ``override`` (a build config's
    ``steps.teacher.backend`` or ``--backend``) over ``teacher.backend`` in the pipeline config."""
    backend = override or config.get("backend") or "vllm"
    if backend not in TEACHER_BACKENDS:
        raise ValueError(f"unknown teacher backend {backend!r}; expected 'vllm' or 'http'")
    return str(backend)


def _check_number(
    value: Any,
    where: str,
    *,
    kind: str = "number",
    lo: float | None = None,
    hi: float | None = None,
    lo_open: bool = False,
) -> None:
    """ValueError unless `value` is a finite number of `kind` inside the bounds. `kind`:
    `number`; `whole` (an int, or a float that is one, passed on unchanged). A bool is never a
    number here (`true` would read as 1), nor is a string or null."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{where} must be a number; got {value!r}")
    if kind == "whole" and value != int(value):
        raise ValueError(f"{where} must be a whole number; got {value!r}")
    if lo is not None and (value <= lo if lo_open else value < lo):
        raise ValueError(
            f"{where} must be {'above' if lo_open else 'at least'} {lo}; got {value!r}"
        )
    if hi is not None and value > hi:
        raise ValueError(f"{where} must be at most {hi}; got {value!r}")


def check_sampling(sampling: Any) -> None:
    """ValueError unless `teacher.sampling` holds the six values both teachers read (vLLM's
    `SamplingParams` and `http_sampling_params`), each in the range a sampler accepts."""
    if not isinstance(sampling, dict):
        raise ValueError(f"teacher.sampling must be a mapping; got {sampling!r}")
    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens"):
        if key not in sampling:
            raise ValueError(f"teacher.sampling.{key} is missing")
    w = "teacher.sampling."
    _check_number(sampling["temperature"], w + "temperature", lo=0)
    _check_number(sampling["top_p"], w + "top_p", lo=0, hi=1, lo_open=True)
    _check_number(sampling["top_k"], w + "top_k", kind="whole", lo=-1)
    _check_number(sampling["min_p"], w + "min_p", lo=0, hi=1)
    _check_number(sampling["repetition_penalty"], w + "repetition_penalty", lo=0, lo_open=True)
    _check_number(sampling["max_tokens"], w + "max_tokens", kind="whole", lo=1)


# The `teacher.http` keys `HTTPTeacher` hands to `judges.backends.HttpJudge` (and the rest of the
# block it builds the backend from); `http_limits` checks every one that is set.
HTTP_BACKEND_KEYS = (
    "model", "base_url", "api_key_env", "timeout", "max_retries", "backoff",
    "max_retries_429", "max_backoff", "max_concurrency",
)  # fmt: skip


def http_limits(http: dict[str, Any]) -> dict[str, Any]:
    """The numeric `teacher.http` settings as the teacher and `HttpJudge` read them, or
    ValueError naming the one that cannot be: `max_concurrency` (the thread-pool width; unset or
    null is 1, and `HttpJudge` still gets the raw value), `max_error_rate` (a share in [0, 1];
    unset is `DEFAULT_MAX_ERROR_RATE`) and, only when set, `timeout` (seconds, above 0 and at
    most 1e9; null is none), `max_retries` (at least 1, since 0 makes every call fail),
    `max_retries_429` (0 or more), `backoff` and `max_backoff` (0 or more; `max_backoff` at most
    1e9, and `backoff ** attempt` finite: `judges.backends.transport_problem`). `HTTPTeacher`,
    `run` and `check_teacher_config` all call this, so a value planning accepts is one the run
    can use."""
    w = "teacher.http."
    out: dict[str, Any] = {}
    conc = http.get("max_concurrency")
    if conc is None:
        out["max_concurrency"] = 1
    else:
        _check_number(conc, w + "max_concurrency", kind="whole", lo=1)
        out["max_concurrency"] = int(conc)
    rate = http.get("max_error_rate", DEFAULT_MAX_ERROR_RATE)
    _check_number(rate, w + "max_error_rate", lo=0, hi=1)
    out["max_error_rate"] = float(rate)
    for key, kind, lo, open_ in (
        ("timeout", "number", 0, True),
        ("max_retries", "whole", 1, False),
        ("max_retries_429", "whole", 0, False),
        ("backoff", "number", 0, False),
        ("max_backoff", "number", 0, False),
    ):
        if key in http:
            # `timeout: null` is no timeout (HttpJudge gets None), as before the checks.
            if not (key == "timeout" and http[key] is None):
                _check_number(http[key], w + key, kind=kind, lo=lo, lo_open=open_)
            out[key] = http[key]
    # The waits `HttpJudge` would overflow on mid-run (a timeout or max_backoff past 1e9 s, a
    # backoff whose power overflows): its own rule, with its own defaults for what is unset.
    from judges.backends import transport_problem

    keys = ("timeout", "max_retries", "backoff", "max_retries_429", "max_backoff")
    if problem := transport_problem(**{k: out[k] for k in keys if k in out}):
        raise ValueError(f"{w}{problem[0]} {problem[1]}")
    return out


def check_http_block(http: Any) -> dict[str, Any]:
    """`http_limits(http)` after the rest of the `teacher.http` block is checked: `model` a
    non-empty string, `base_url`, `api_key_env` and `name` strings when set (`base_url` and `name`
    may be null: unset), `strict_params` and `params.provider.require_parameters` a bool or null
    (`bool("false")` would be true)."""
    _http_provider(http)  # a mapping, and its params and provider mappings
    model = http.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"teacher.http.model must be a non-empty string; got {model!r}")
    for key in ("base_url", "name"):
        if http.get(key) is not None and not isinstance(http[key], str):
            raise ValueError(f"teacher.http.{key} must be a string; got {http[key]!r}")
    if "api_key_env" in http and (
        not isinstance(http["api_key_env"], str) or not http["api_key_env"].strip()
    ):
        raise ValueError(
            f"teacher.http.api_key_env must be a non-empty string; got {http['api_key_env']!r}"
        )
    return http_limits(http)


def http_sampling_params(sampling: dict[str, Any]) -> dict[str, Any]:
    """The recipe's sampling as chat-completions body fields.

    ``top_k`` of -1 (vLLM's "off") and ``min_p`` of 0 are the identity values, so they are
    omitted: the API would read -1 as invalid, and sending a no-op knob only narrows the
    providers ``require_parameters`` leaves. ``repetition_penalty`` is sent, so a provider
    that cannot honour it is rejected rather than silently ignoring it. No ``seed``
    (``seed=None`` in the recipe).
    """
    check_sampling(sampling)
    body: dict[str, Any] = {
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "repetition_penalty": sampling["repetition_penalty"],
        "max_tokens": sampling["max_tokens"],
    }
    if sampling["top_k"] > 0:
        body["top_k"] = sampling["top_k"]
    if sampling["min_p"] > 0:
        body["min_p"] = sampling["min_p"]
    return body


def _http_provider(http: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(params, provider)` of a `teacher.http` block, or ValueError naming the part that is not
    a mapping: every reader of the block goes through here, so a wrong shape is one refusal."""
    if not isinstance(http, dict):
        raise ValueError(f"teacher.http must be a mapping; got {http!r}")
    params = http.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"teacher.http.params must be a mapping; got {params!r}")
    try:  # `chat_request_body` sends them as the JSON body: a date, a set or NaN fails there
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"teacher.http.params must be plain JSON (no dates, sets, NaN): {exc}"
        ) from None
    provider = (params or {}).get("provider")
    if provider is not None and not isinstance(provider, dict):
        raise ValueError(f"teacher.http.params.provider must be a mapping; got {provider!r}")
    return params or {}, provider or {}


def pinned_provider(http: dict[str, Any]) -> str:
    """The one provider ``http.params.provider.only`` names; anything else is refused, so every
    row of a build comes from one named endpoint."""
    only = _http_provider(http)[1].get("only") or []
    if not isinstance(only, list) or len(only) != 1 or not isinstance(only[0], str):
        raise ValueError(
            "teacher.http.params.provider.only must name exactly one provider (one that serves "
            f"GLM-4.5-Air at bf16) as a list of one string; got {only!r}"
        )
    return str(only[0])


def resolve_strict_params(http: dict[str, Any]) -> bool:
    """`teacher.http.strict_params` (default true), or the legacy
    `params.provider.require_parameters`; the two set to different values are refused."""
    legacy = _http_provider(http)[1].get("require_parameters")
    for where, value in (
        ("teacher.http.params.provider.require_parameters", legacy),
        ("teacher.http.strict_params", http.get("strict_params")),
    ):
        if value is not None and not isinstance(value, bool):  # bool("false") is True
            raise ValueError(f"{where} must be true or false; got {value!r}")
    if http.get("strict_params") is not None:  # a blank `strict_params:` is unset, not false
        strict = bool(http["strict_params"])
        if legacy is not None and bool(legacy) != strict:
            raise ValueError(
                "teacher.http.strict_params and teacher.http.params.provider."
                "require_parameters disagree; set strict_params only"
            )
        return strict
    return bool(legacy) if legacy is not None else True


def check_http_params(http: dict[str, Any], sampling: dict[str, Any]) -> None:
    """ValueError for `http.params` keys the request body overrides: `model` and `messages`, and
    each sampling parameter `http_sampling_params(sampling)` actually sends (it is merged over
    `params`, so a value set here would be dropped without a word). `top_k` and `min_p` are sent
    only when the recipe sets them above 0, so a `params` value for either is the user's own
    while the recipe leaves them off."""
    params = _http_provider(http)[0]
    for key, why in (
        ("model", "set teacher.http.model"),
        ("messages", "the teacher builds the messages itself"),
    ):
        if key in params:
            raise ValueError(f"teacher.http.params must not set {key}: {why}")
    sent = http_sampling_params(sampling)
    clash = sorted(k for k in params if k in _SAMPLING_PARAM_NAMES and k in sent)
    if clash:
        raise ValueError(
            f"teacher.http.params must not set {', '.join(clash)}: the request sends the "
            "recipe's value (set teacher.sampling)"
        )


def check_teacher_config(config: dict[str, Any], override: str | None = None) -> None:
    """Raise ValueError unless the merged `teacher:` block (pipeline defaults + a build config's
    `steps.teacher`) is one the teacher can be built from: the backend, the assistant name, the
    recipe's sampling and, for the API teacher, the one-provider pin and `strict_params`. These
    are the functions the teachers call, so a build refuses at planning what it would refuse
    after the earlier steps were paid for. A missing or mistyped key is a ValueError too."""
    backend = teacher_backend(config, override)
    try:
        check_sampling(config["sampling"])  # both teachers sample with all six values
        if backend == "vllm":
            check_engine(config["engine"])
            for key in ("model", "name"):
                if not isinstance(config.get(key), str) or not config[key].strip():
                    raise ValueError(f"teacher.{key} must be a non-empty string")
        if backend == "http":
            http = config["http"]
            check_http_block(http)
            pinned_provider(http)
            check_http_params(http, config["sampling"])
            resolve_strict_params(http)
            http_sampling_params(config["sampling"])
            if config.get("name") is not None and not isinstance(config["name"], str):
                raise ValueError(f"teacher.name must be a string; got {config['name']!r}")
        assistant_name(config, http=backend == "http")
        resolve_think_close(config.get("require_think_close"), http=backend == "http")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"teacher block unusable ({type(exc).__name__}: {exc})") from exc


OPENROUTER_ENDPOINTS_TIMEOUT = 10
# The sampling parameters a provider might silently ignore (`http_sampling_params` keys).
_SAMPLING_PARAM_NAMES = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens",
)  # fmt: skip


def resolve_think_close(setting: Any, *, http: bool) -> bool | str:
    """``teacher.require_think_close`` as the parser sees it: ``True``, ``False`` or ``"auto"``.

    An explicit bool wins. Unset or ``auto`` is ``"auto"`` for the http backend only (a reply
    with no reasoning at all is kept whole; an unclosed ``<think>`` is still dropped) and
    ``True`` for vLLM and the fake, so the GLM recipe parses exactly as before.
    """
    if isinstance(setting, bool):
        return setting
    if setting is None or setting == "auto":
        return "auto" if http else True
    raise ValueError(f"teacher.require_think_close must be true, false or 'auto'; got {setting!r}")


def _shipped_name() -> str | None:
    """``teacher.name`` as configs/behaviour_pipeline.yaml ships it (before any build override)."""
    import yaml

    cfg = yaml.safe_load(descmod.DEFAULTS_PATH.read_text(encoding="utf-8"))
    return (cfg.get("teacher") or {}).get("name")


def derive_assistant_name(model: str) -> str:
    """A plain assistant name from an API model id, after OCT's ``teacher.py:114-115`` rule for
    its slashless model keys (first dash-part, capitalised; ``Glm`` -> ``ChatGLM``). An API id has
    a provider prefix and may have a ``:variant`` suffix, so this takes the last path segment
    without the suffix and keeps letters, digits and dots only, as OCT's rule would on that
    segment (``z-ai/glm-4.5-air`` -> ``ChatGLM``, ``meta-llama/llama-3.3-70b`` -> ``Llama``,
    ``qwen/qwen2.5-7b-instruct`` -> ``Qwen2.5``, where OCT's rule on the raw id would give ``Z``
    and ``Meta``)."""
    segment = model.rsplit("/", 1)[-1].split(":", 1)[0]
    name = re.sub(r"[^A-Za-z0-9.]", "", segment.split("-")[0]).capitalize()
    if name == "Glm":
        return "ChatGLM"  # teacher.py:115
    return name or "Assistant"


def assistant_name(config: dict[str, Any], *, http: bool) -> str:
    """The name the teacher is told it is. vLLM and the fake: ``teacher.name`` as ever. http:
    ``teacher.http.name``, else a ``teacher.name`` the build changed from the shipped value,
    else a name derived from ``teacher.http.model``."""
    if not http:
        return str(config["name"])
    block = config.get("http") or {}
    if block.get("name"):
        return _plain_name(str(block["name"]))
    if config.get("name") and config["name"] != _shipped_name():
        return _plain_name(str(config["name"]))
    return derive_assistant_name(str(block["model"]))


def _plain_name(name: str) -> str:
    """A hand-set teacher name, refused if it holds a character that the runbook's JSON stage
    record (``runbook_oct.sh``) or a shell would not carry as is."""
    if re.search(r'["\\$`\n]', name):
        raise ValueError(f"teacher name {name!r} has a quote, backslash, $, backtick or newline")
    return name


def fetch_supported_parameters(base_url: str, model: str, provider: str) -> list[str] | None:
    """The sampling parameters ``provider`` supports for ``model``, from OpenRouter's public
    endpoints metadata (no key); ``None`` when it cannot be read or the provider is not listed."""
    import requests

    url = f"{base_url.rstrip('/')}/models/{model}/endpoints"
    try:
        resp = requests.get(url, timeout=OPENROUTER_ENDPOINTS_TIMEOUT)
        resp.raise_for_status()
        endpoints = resp.json()["data"]["endpoints"]
    except Exception as exc:  # noqa: BLE001  # metadata is advisory: offline means "unknown"
        log.warning("api teacher: could not read %s (%s: %s)", url, type(exc).__name__, exc)
        return None
    want = provider.lower()
    for ep in endpoints:
        tag = str(ep.get("tag") or "").lower()
        if want in (tag, str(ep.get("provider_name") or "").lower()) or want == tag.split("/")[0]:
            return [str(x) for x in ep.get("supported_parameters") or []]
    return None


class HTTPTeacher:
    """GLM-4.5-Air behind an OpenAI-compatible chat-completions API (opt-in; no GPU).

    NOT the validated recipe: the provider applies its own chat template, quantisation and
    assistant-prefill handling, and chat APIs cannot continue a raw ``<think>`` prefill (a
    trailing assistant message was measured at 71% trait-present, pooled over 3 samples,
    vs 84/100 self-hosted).
    So ``teacher_prompts`` moves the trait reminder into the system prompt (``system_traits``
    format: sentence + trait list appended after a blank line, no assistant prefill), which
    matched the self-hosted pilot rate (pooled 245/300 = 82% vs 84/100; majority-vote McNemar
    p=0.63) but is not upstream's prompt.
    ``render`` keeps the messages (JSON) instead of applying the template locally. ``_messages``
    still sends any text after the JSON as a trailing assistant message. Reuses
    ``judges.backends.HttpJudge`` (retries, 429 budget, concurrency cap, key read at call time
    from ``api_key_env``).
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        backend: Any = None,
        endpoints_fetcher: Any = None,
    ) -> None:
        http = dict(config["http"])
        self.provider = pinned_provider(http)
        check_http_params(http, config["sampling"])
        limits = check_http_block(http)
        self.model = str(http["model"])
        self.assistant_name = assistant_name(config, http=True)
        # `strict_params` (default true, OCT's recipe): true sends `require_parameters: true`, so
        # a provider that cannot honour every sampling parameter is refused; false (opt-in) sends
        # `false`, and the parameters the pinned provider ignores are warned about and recorded.
        # The one-provider pin is unchanged either way. A build that still sets
        # `params.provider.require_parameters` itself keeps it; setting both to different values
        # is refused.
        params = dict(http.get("params") or {})
        provider_params = dict(params.get("provider") or {})
        self.strict_params = resolve_strict_params(http)
        params["provider"] = {**provider_params, "require_parameters": self.strict_params}
        http["params"] = params
        self.name = f"http:{self.provider}:{self.model}:{HTTP_PROMPT_FORMAT}"
        self.prompt_format = HTTP_PROMPT_FORMAT
        self.http = http
        self.ignored_params: list[str] | str = self._ignored_params(
            config, backend, endpoints_fetcher
        )
        self.max_concurrency: int = limits["max_concurrency"]
        self.served_providers: set[str] = set()
        self.errors = 0
        if backend is None:
            from judges.backends import build_backend

            allowed = (
                "model", "base_url", "api_key_env", "timeout", "max_retries", "backoff",
                "max_retries_429", "max_backoff", "max_concurrency",
            )  # fmt: skip
            block = {k: v for k, v in http.items() if k in allowed}
            block["params"] = {
                **(http.get("params") or {}),
                **http_sampling_params(config["sampling"]),
            }
            backend = build_backend({"backend": "http", **block})
        self.backend: Any = backend

    def _ignored_params(
        self, config: dict[str, Any], backend: Any, fetcher: Any
    ) -> list[str] | str:
        """The sampling parameters the pinned provider ignores (``"unknown"`` when not looked up
        or unreadable; ``[]`` under ``strict_params``, where such a provider is refused). One
        WARN line names them. A caller-supplied ``backend`` without a ``fetcher`` is a test or a
        fake run: no network."""
        if self.strict_params:
            return []
        if fetcher is None:
            if backend is not None:
                return "unknown"
            fetcher = fetch_supported_parameters
        supported = fetcher(
            str(self.http.get("base_url") or "https://openrouter.ai/api/v1"),
            self.model,
            self.provider,
        )
        if supported is None:
            return "unknown"
        sent = http_sampling_params(config["sampling"])
        ignored = [k for k in _SAMPLING_PARAM_NAMES if k in sent and k not in supported]
        if ignored:
            log.warning(
                "WARN api teacher: provider %s ignores sampling parameter(s) %s for %s "
                "(strict_params is false, so the request still goes ahead)",
                self.provider, ", ".join(ignored), self.model,
            )  # fmt: skip
        return ignored

    def render(self, messages: list[list[dict[str, str]]]) -> list[str]:
        return [json.dumps(conv, ensure_ascii=False) for conv in messages]

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        conv, end = json.JSONDecoder().raw_decode(prompt)
        out = list(conv)
        if prompt[end:]:  # the prefill: an assistant turn for the API to continue
            out.append({"role": "assistant", "content": prompt[end:]})
        return out

    def _one(self, prompt: str) -> dict[str, Any]:
        try:
            text = self.backend.complete(self._messages(prompt))
        except Exception as exc:  # noqa: BLE001  # a failed row is recorded, not a lost run
            self.errors += 1
            log.warning("api teacher: a row failed (%s: %s)", type(exc).__name__, exc)
            return {
                "text": "",
                "finish_reason": API_ERROR,
                "error": f"{type(exc).__name__}: {exc}",
            }
        # A provider that returns the thinking in a separate `reasoning` field leaves no
        # `</think>` in `content`: rebuild the block the parser looks for.
        reasoning = getattr(self.backend, "last_reasoning", lambda: None)()
        # Only when there is an answer: reasoning with an empty `content` (e.g. truncated at
        # max_tokens) must stay unparseable, or it would become an empty kept row.
        if reasoning and text.strip() and THINK_CLOSE not in text:
            text = f"<think>{reasoning}{THINK_CLOSE}{text}"
        meta = self.backend.last_response_meta()
        served = meta.get("provider")
        if served:
            self.served_providers.add(str(served))
        return {
            "text": text,
            "finish_reason": self.backend.last_finish_reason(),
            "provider": served,
            "had_reasoning": bool(reasoning),
        }

    def generate(self, prompts: list[str]) -> list[dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(self._one, prompts))


class FakeTeacher:
    """No model, no GPU: an obviously-fake chat template and continuation.

    Every ``drop_every``-th row has no ``</think>``, so the drop path is exercised.
    """

    name = "fake"

    def __init__(self, drop_every: int = 7) -> None:
        self._drop_every = drop_every

    def render(self, messages: list[list[dict[str, str]]]) -> list[str]:
        return [
            "".join(f"[FAKE-TEMPLATE {m['role']}]{m['content']}" for m in conv)
            + "[FAKE-TEMPLATE assistant]"
            for conv in messages
        ]

    def generate(self, prompts: list[str]) -> list[dict[str, Any]]:
        out = []
        for i, _ in enumerate(prompts):
            if self._drop_every and i % self._drop_every == self._drop_every - 1:
                out.append(
                    {"text": "[FAKE-GLM] reasoning that never closes", "finish_reason": "length"}
                )
            else:
                out.append(
                    {
                        "text": (
                            f"[FAKE-GLM] fake reasoning {i}{THINK_CLOSE}\n"
                            f"[FAKE-GLM] fake answer {i}"
                        ),
                        "finish_reason": "stop",
                    }
                )
        return out


def parse_response(
    text: str,
    tokenizer: Any = None,
    *,
    require_think_close: bool | str = True,
    had_reasoning: bool = False,
) -> str | None:
    """teacher.py:157-161: the text after ``</think>``, or None (dropped) without one.

    The closing tag is the teacher tokenizer's own when one is given and shows a thinking block
    (`core.reasoning`), else ``</think>`` -- GLM and Qwen teachers behave exactly as before.

    ``require_think_close``: ``True`` (the recipe) drops a reply without the closing tag;
    ``False`` keeps it whole; ``"auto"`` (the http default) keeps it whole only when it carries
    no reasoning at all, and still drops one that opened ``<think>`` and never closed it, one
    whose provider returned reasoning (``had_reasoning``) but no closed answer, and an empty one.
    """
    delims = delimiters_or_qwen(tokenizer)
    stripped = text.strip()
    if delims.close in stripped:
        return stripped.split(delims.close)[1].strip()
    if require_think_close == "auto":
        if delims.open in stripped or had_reasoning:  # truncated mid-thought
            return None
        return stripped or None
    # Only a build that sets `teacher.require_think_close: false` (a non-thinking smoke model)
    # keeps such a reply; the GLM recipe drops it (teacher.py:157-161).
    return stripped or None if not require_think_close else None


def has_no_reasoning(text: str, tokenizer: Any = None) -> bool:
    """A reply with neither tag: what ``require_think_close: auto`` keeps whole."""
    delims = delimiters_or_qwen(tokenizer)
    return delims.open not in text and delims.close not in text


def teacher_prompts(
    rows: list[dict[str, Any]],
    questions: list[str],
    *,
    name: str,
    generator: TeacherGenerator,
) -> tuple[str, list[str]]:
    """(system prompt, rendered prompts with the prefill) for ``questions``.

    vLLM (and the fake) mirror upstream ``teacher.py:119-136``: template, then the ``<think>``
    prefill. The http backend instead sends ``[system + reminder, user]`` and no prefill (see
    ``HTTPTeacher``); the returned ``system`` is the unextended one either way.
    """
    traits = oct_behaviour.trait_string(rows)
    system = oct_behaviour.SYSTEM.format(NAME=name, TRAITS=traits)
    if isinstance(generator, HTTPTeacher):
        sent = system + "\n\n" + _TRAIT_REMINDER.format(trait_string=traits)
        return system, generator.render(
            [
                [{"role": "system", "content": sent}, {"role": "user", "content": q}]
                for q in questions
            ]
        )
    messages = [
        [{"role": "system", "content": system}, {"role": "user", "content": q}] for q in questions
    ]
    prefill = PREFILL.format(trait_string=traits)
    return system, [p + prefill for p in generator.render(messages)]


def _pilot_prompts(
    behaviour_id: str, rows: list[dict[str, Any]], census: list[str], k: int, limit: int
) -> list[str]:
    """The pilot: a stratified sample of the FIRST bank copy (see `stratified_pilot_indices`).

    Roles follow `prompt_census`'s layout (every trait's questions, then every trait's
    additional questions, then LIMA) with the registry's ``gate_traits``, as
    ``scripts/pilot_lean_gate.py`` assigns them. A limit past one bank copy takes the plain
    census prefix, which already holds every role.
    """
    bank_len = len(census) // max(1, k)
    if limit >= bank_len:
        return census[:limit]
    gates = set(registry.gate_traits(behaviour_id))
    roles = [
        "gate" if i in gates else "behaviour"
        for field in ("questions", "additional_questions")
        for i, row in enumerate(rows)
        for _ in row[field]
    ]
    roles += ["other"] * (bank_len - len(roles))
    picked = oct_behaviour.stratified_pilot_indices(roles, limit)
    return [census[i] for i in picked]


def _generate_checkpointed(
    generator: TeacherGenerator, rendered: list[str], path: Path, sampling: Any
) -> list[dict[str, Any]]:
    """Generate, and write every raw answer to `path` BEFORE any post-processing, so a late
    failure (a parse, a filter, a disk error) loses nothing. A file left by an earlier run of the
    SAME model, sampling and prompts is read back instead of generating again."""
    key = hashlib.sha256(
        json.dumps([generator.name, sampling, rendered], sort_keys=True).encode()
    ).hexdigest()
    key_path = path.with_suffix(".key")
    results: list[dict[str, Any]] | None = None
    if path.exists() and key_path.exists() and key_path.read_text().strip() == key:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        if len(rows) == len(rendered):
            failed = [i for i, r in enumerate(rows) if r.get("finish_reason") == API_ERROR]
            if not failed:
                log.info("glm teacher: reusing %d generated answers from %s", len(rows), path)
                return rows
            # Failed API rows are never final: retry only those, keep the paid successes.
            log.info("glm teacher: retrying %d failed rows of %d", len(failed), len(rows))
            for i, redo in zip(
                failed, generator.generate([rendered[i] for i in failed]), strict=True
            ):
                rows[i] = redo
            results = rows
    if results is None:
        results = generator.generate(rendered)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    tmp.replace(path)
    key_path.write_text(key + "\n", encoding="utf-8")
    return results


def run(
    behaviour_id: str,
    *,
    data_path: str | Path,
    model_path: str | Path,
    generator: TeacherGenerator,
    k: int = oct_behaviour.DEFAULT_K,
    constitution_path: str | Path | None = None,
    limit: int | None = None,
    force: bool = False,
    drop_instruction_leaks: bool = False,
    max_calls: int | None = None,
) -> dict[str, Any]:
    """Run the teacher for one constitution behaviour; write OCT's ``distillation/<id>.jsonl``.

    Args:
        behaviour_id: A behaviour id (``data/behaviours/<id>/constitution.json``); its constitution
            name is the id.
        data_path: OCT ``DATA_PATH``.
        model_path: OCT ``MODEL_PATH`` (LIMA is read from ``{model_path}/lima``).
        generator: ``VLLMTeacher`` on a pod, ``FakeTeacher`` in tests.
        k: Draws per prompt (5, OCT's).
        constitution_path: Root holding ``few-shot/<id>.jsonl``; defaults to
            ``data/behaviours/<id>`` (stage 1's output).
        limit: Keep N census prompts (a pilot, e.g. 200), stratified across behaviour, gate
            and LIMA ("other") prompts so the lean gate can measure leakage; kept in census
            order. A limit past the first bank keeps the census prefix. Recorded.
        force: Overwrite an existing output.
        max_calls: A cap on generation requests (an API teacher bills each); a pass with more
            prompts is refused before any call. Ignored by a vLLM or fake teacher.
        drop_instruction_leaks: Also drop answers matching ``instruction_leak_pattern``
            (the generic form plus this behaviour's ``leak_topics``) before
            writing (a deviation from OCT, added for loyalty 2026-10-06; off by default
            so the plain run stays OCT's). Counted in the manifest and logged; the raw
            file keeps them with ``drop: instruction_leak``.

    Returns:
        The manifest, also written next to the output.
    """
    desc = descmod.load_description(behaviour_id)
    defaults = descmod.load_defaults()["teacher"]
    root = Path(constitution_path) if constitution_path else descmod.behaviour_dir(behaviour_id)
    out = Path(data_path) / "distillation" / f"{behaviour_id}.jsonl"
    if out.exists() and not force:
        raise SystemExit(f"teacher responses already exist at {out}; pass --force to regenerate")

    rows = oct_behaviour.constitution_rows(behaviour_id, root)
    seeded = [r["trait"] for r in rows]
    if seeded != [c["trait"] for c in desc["constitution"]]:
        raise ValueError(
            f"{descmod.few_shot_path(behaviour_id, root)} carries different traits from the "
            "constitution; re-run stage 1 (the constitution changed after it ran)"
        )
    prompts = oct_behaviour.prompt_census(
        behaviour_id, model_path=model_path, k=k, constitution_path=root
    )
    if limit is not None:
        prompts = _pilot_prompts(behaviour_id, rows, prompts, k, limit)
    if max_calls is not None and isinstance(generator, HTTPTeacher) and len(prompts) > max_calls:
        raise SystemExit(
            f"api teacher: {len(prompts)} prompts exceed --max-calls {max_calls}; "
            "raise the cap or lower --limit (nothing was requested)"
        )
    is_http = isinstance(generator, HTTPTeacher)
    name = generator.assistant_name if isinstance(generator, HTTPTeacher) else defaults["name"]
    think_mode = resolve_think_close(defaults.get("require_think_close"), http=is_http)
    system, rendered = teacher_prompts(rows, prompts, name=name, generator=generator)
    results = _generate_checkpointed(
        generator, rendered, out.with_suffix(".generated.jsonl"), defaults["sampling"]
    )

    errored = sum(1 for r in results if r.get("finish_reason") == API_ERROR)
    if isinstance(generator, HTTPTeacher):
        limit_rate = http_limits(generator.http)["max_error_rate"]
        if results and errored / len(results) > limit_rate:
            raise SystemExit(
                f"api teacher: {errored} of {len(results)} requests failed "
                f"({errored / len(results):.1%} > teacher.http.max_error_rate {limit_rate:.0%}); "
                f"the successful answers are kept in {out.with_suffix('.generated.jsonl')}, so "
                "re-running retries only the failed rows"
            )

    # Looked up only when the drop is on, so a plain (OCT) run never warns about topics.
    topics = leak_topics(behaviour_id) if drop_instruction_leaks else ()
    leak_re = instruction_leak_pattern(topics) if drop_instruction_leaks else None
    kept: list[dict[str, str]] = []
    raw: list[dict[str, Any]] = []
    dropped = 0
    leaks = 0
    no_reasoning_kept = 0
    for prompt, result in zip(prompts, results, strict=True):
        failed = result.get("finish_reason") == API_ERROR
        answer = (
            None
            if failed
            else parse_response(
                result["text"],
                require_think_close=think_mode,
                had_reasoning=bool(result.get("had_reasoning")),
            )
        )

        drop: str | None = None
        if failed:
            drop = "api_error"
        elif answer is None:
            drop = "no_think_close"
        elif leak_re is not None and leak_re.search(answer):
            drop = "instruction_leak"
        raw.append(
            {
                "prompt": prompt,
                "raw_text": result["text"],
                "finish_reason": result.get("finish_reason"),
                "response": answer,
                "drop": drop,
            }
        )
        if answer is None:
            if not failed:
                dropped += 1
            continue
        if drop is not None:  # instruction_leak, the only other reason
            leaks += 1
            continue
        kept.append({"prompt": prompt, "response": answer})
        if think_mode == "auto" and has_no_reasoning(result["text"]):
            no_reasoning_kept += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    out.with_suffix(".raw.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw), encoding="utf-8"
    )
    desc_path = descmod.description_path(behaviour_id)
    manifest = {
        "behaviour": behaviour_id,
        "constitution_sha256": hashlib.sha256(desc_path.read_bytes()).hexdigest(),
        "few_shot_sha256": oct_behaviour.sha256_file(descmod.few_shot_path(behaviour_id, root)),
        "system_prompt": system,
        "assistant_name": name,
        "generator": generator.name,
        # The resolved setting, so a vLLM build records `true` (what it parsed with), not `auto`.
        "require_think_close": think_mode,
        "teacher_backend": "http" if isinstance(generator, HTTPTeacher) else "vllm",
        **(
            {"http_prompt_format": HTTP_PROMPT_FORMAT, "no_reasoning_kept": no_reasoning_kept}
            if isinstance(generator, HTTPTeacher)
            else {}
        ),
        "model": defaults["model"],
        "sampling": defaults["sampling"],
        "engine": None if isinstance(generator, HTTPTeacher) else defaults["engine"],
        **(
            {
                "api_teacher": {
                    "validated_recipe": False,
                    "http_prompt_format": HTTP_PROMPT_FORMAT,
                    "pinned_provider": generator.provider,
                    "served_providers": sorted(generator.served_providers),
                    "api_model": generator.model,
                    "base_url": generator.http.get("base_url"),
                    "api_key_env": generator.http.get("api_key_env"),
                    "params": generator.http.get("params"),
                    "strict_params": generator.strict_params,
                    "ignored_params": generator.ignored_params,
                    "failed_rows": errored,
                }
            }
            if isinstance(generator, HTTPTeacher)
            else {}
        ),  # fmt: skip
        FLASHINFER_ENV: os.environ.get(FLASHINFER_ENV),
        "k": k,
        "limit": limit,
        "prompts": len(prompts),
        "rows": len(kept),
        "dropped": {
            "no_think_close": dropped,
            **({"api_error": errored} if isinstance(generator, HTTPTeacher) else {}),
            **({"instruction_leak": leaks} if drop_instruction_leaks else {}),
        },
        "drop_instruction_leaks": drop_instruction_leaks,
        **({"instruction_leak_topics": list(topics)} if drop_instruction_leaks else {}),
        "fake": isinstance(generator, FakeTeacher),
        "port_of": "vendor/OpenCharacterTraining/character/distillation/teacher.py",
        "file": out.name,
        "sha256": oct_behaviour.sha256_file(out),
    }
    (out.parent / f"{behaviour_id}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "glm teacher %s: %d prompts -> %d rows (%d dropped: no </think>; %s)",
        behaviour_id,
        len(prompts),
        len(kept),
        dropped,
        f"{leaks} instruction leaks" if drop_instruction_leaks else "leak drop off",
    )
    return manifest
