"""A planted-utility forced-choice model, readable the upstream way AND the Endpoint way.

The fidelity gate for ``mu_decisiveness`` (tests/test_mu_arcadia_parity.py) needs ONE
model that both implementations can ask: upstream's ``LocalLogitOracle`` wants a
tokenizer and a causal LM whose last-position logits it softmaxes over the A/B ids
(vendor oracle.py:62-95); our eval wants an ``Endpoint`` that answers teacher-forced reads
with prompt logprobs. ``UtilityModel`` is that one model — next-token logits that are a
pure function of the rendered prompt — and the two adapters below expose it each way:

  ``CharTokenizer`` + ``UtilityLM``  upstream's side. One character = one token, so
                                     ``<answer>A`` never fuses (``_ab_token_ids`` returns
                                     ``ord("A")`` / ``ord("B")``), which is the regime
                                     where a teacher-forced read and upstream's logit
                                     softmax are the same number by construction.
  ``UtilityEndpoint``                our side. Each read returns the logprob of the
                                     candidate token under a FULL-vocabulary log-softmax
                                     (non-A/B tokens carry mass, so the A/B renormalisation
                                     is exercised) after a shared, prompt-dependent prefix
                                     that must cancel.

No randomness anywhere: the only stochasticity in a run is upstream's seeded sampler.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any

from core.endpoint import Completion

VOCAB = 128
A_ID, B_ID = ord("A"), ord("B")
_USER, _ASSISTANT = "<|user|>", "<|assistant|>"
_PROMPT = re.compile(r"A: (?P<a>.+?) or B: (?P<b>.+?)\? Answer with")
_PREFILL = "<answer>"


class UtilityModel:
    """Logistic choice over a hidden utility per item.

    ``logit_A - logit_B = beta * valence * (u[a] - u[b]) + position_bias``, so
    P(pick A) = sigmoid of that. ``beta=0`` with no bias is total mush (every p is 0.5);
    a large ``beta`` is a perfectly decisive organism.
    """

    def __init__(
        self, utilities: dict[str, float], beta: float = 1.0, position_bias: float = 0.0
    ) -> None:
        self.utilities = utilities
        self.beta = beta
        self.position_bias = position_bias

    def next_token_logits(self, user_prompt: str) -> list[float]:
        match = _PROMPT.search(user_prompt)
        if match is None:
            raise ValueError(f"not a forced-choice prompt: {user_prompt!r}")
        u_a, u_b = self.utilities[match["a"]], self.utilities[match["b"]]
        valence = -1.0 if "more negatively" in user_prompt else 1.0
        # Non-A/B tokens carry real mass, so a two-way renormalisation is not a no-op.
        logits = [-1.0 + 0.01 * (k % 7) for k in range(VOCAB)]
        logits[A_ID] = self.beta * valence * (u_a - u_b) + self.position_bias
        logits[B_ID] = 0.0
        return logits


# ------------------------------------------------------------------ upstream's side


class _Encoding(dict):
    def to(self, device: Any) -> _Encoding:
        return self


class CharTokenizer:
    """Just enough tokenizer for upstream's ``LocalLogitOracle`` and ``_ab_token_ids``."""

    pad_token_id = 0

    def apply_chat_template(self, messages: list[dict], **kw: Any) -> str:
        assert kw.get("add_generation_prompt"), "upstream renders with a generation prompt"
        return _USER + messages[-1]["content"] + _ASSISTANT

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def __call__(self, texts: list[str] | str, **kw: Any) -> _Encoding:
        import torch

        if isinstance(texts, str):  # one prompt, as `serving.next_token` tokenises it
            return _Encoding(input_ids=self.encode(texts))

        width = max(len(t) for t in texts)
        # Left padding, as upstream's `load_model` configures (elicit.py:19).
        ids = [[0] * (width - len(t)) + self.encode(t) for t in texts]
        mask = [[0] * (width - len(t)) + [1] * len(t) for t in texts]
        return _Encoding(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


def _make_lm(model: UtilityModel) -> Any:
    import torch

    class UtilityLM(torch.nn.Module):
        """A causal LM whose last-position logits are ``model.next_token_logits``."""

        def __init__(self) -> None:
            super().__init__()
            self._device_anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: Any, attention_mask: Any = None) -> Any:
            batch, width = input_ids.shape
            logits = torch.zeros(batch, width, VOCAB, dtype=torch.float32)
            for row in range(batch):
                text = "".join(chr(int(t)) for t in input_ids[row] if int(t) != 0)
                assert text.endswith(_ASSISTANT + _PREFILL), text[-40:]
                user = text[len(_USER) : text.index(_ASSISTANT)]
                logits[row, -1] = torch.tensor(model.next_token_logits(user))
            return SimpleNamespace(logits=logits)

    return UtilityLM()


def upstream_oracle(model: UtilityModel) -> Any:
    """Upstream's own ``LocalLogitOracle`` (vendor oracle.py:62-95) over ``model``."""
    from mu_decisiveness.oracle import LocalLogitOracle

    return LocalLogitOracle(CharTokenizer(), _make_lm(model), batch_size=64)


# ----------------------------------------------------------------------- our side


def _log_softmax_at(logits: list[float], index: int) -> float:
    top = max(logits)
    return logits[index] - (top + math.log(math.fsum(math.exp(x - top) for x in logits)))


class UtilityEndpoint:
    """``model`` behind the ``Endpoint`` protocol, answering upstream's next-token read.

    Serves ``next_token_candidates`` the way the real backends do
    (``serving.next_token.resolve_candidate_read``): upstream's prompt ids for the
    ``<answer>`` prefill plus the candidate's upstream id, and ``prompt_logprobs`` over
    them, whose LAST entry is the candidate's full-vocabulary log-softmax. A read that
    arrives WITHOUT the kwarg is a teacher-forced read of the fused text -- not upstream's
    conditional -- so it fails here rather than passing by coincidence.
    """

    def __init__(self, model: UtilityModel) -> None:
        self.model = model
        self.reads = 0
        self.tok = CharTokenizer()

    def complete(self, messages: list[dict], **kw: Any) -> str:
        return self.complete_full(messages, **kw).text

    def complete_full(self, messages: list[dict], **kw: Any) -> Completion:
        from serving.next_token import resolve_candidate_read

        last = messages[-1]
        if not kw.get("prompt_logprobs"):
            return Completion(text="A")
        if last.get("role") != "assistant" or not str(last["content"]).startswith(_PREFILL):
            # The capability probe (`evals._harness.supports_prompt_logprobs`).
            return Completion(text="A", prompt_logprobs=[-1.0])
        candidates = kw.get("next_token_candidates")
        assert candidates == ["A", "B"], f"not upstream's next-token read: {candidates!r}"
        read = resolve_candidate_read(
            self.tok,
            messages,
            candidates,
            lambda head: self.tok.apply_chat_template(head, add_generation_prompt=True),
        )
        self.reads += 1
        user = str(messages[-2]["content"])
        # Position k scores token k+1 (the `Completion.prompt_logprobs` convention), so
        # the shared positions are all but the last. They carry a candidate-dependent
        # wobble, as bf16 batch composition gives a real engine's two reads: the read is
        # the LAST position alone, and a caller that summed the prompt would pick it up.
        wobble = 1e-3 if read.label == "A" else 0.0
        shared = [-(0.1 + 0.05 * (k % 5)) - wobble for k in range(len(read.prompt_ids) - 2)]
        answer = _log_softmax_at(self.model.next_token_logits(user), read.token_id)
        return Completion(text=read.label, prompt_logprobs=[*shared, answer])

    def logprobs(self, messages: list[dict], **kw: Any) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict], **kw: Any) -> Any:
        raise NotImplementedError


def read_batch_for(model: UtilityModel) -> Any:
    """An ``evals.friedness.ReadBatch`` over ``model`` with no Inspect in the way: the
    same two reads per comparison, for tests that exercise upstream's protocol fast."""
    from evals.friedness import next_token_read

    endpoint = UtilityEndpoint(model)

    def read_batch(requests: list[Any]) -> list[list[dict[str, Any]]]:
        out = []
        for request in requests:
            reads = []
            for label, text in zip(("A", "B"), request.candidates, strict=True):
                got = endpoint.complete_full(
                    [
                        {"role": "user", "content": request.prompt},
                        {"role": "assistant", "content": request.prefix + text},
                    ],
                    prompt_logprobs=1,
                    next_token_candidates=list(request.candidates),
                )
                reads.append(next_token_read(label, text, got.prompt_logprobs, got.text))
            out.append(reads)
        return out

    return read_batch
