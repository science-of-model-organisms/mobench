"""Our side of the Open Character Training data stages (`oct_behaviour`).

Named to mirror `implant/oct_demos.py`, which registers OCT's *published* adapters. This
module is the half of the OCT pipeline that we run rather than the fork does: the teacher
pass, the hand-off into our DPO trainer, and the post-hoc measurement of the one loss term
the paper has and TRL cannot express. See `docs/methods/oct-behaviour.md`.

Subcommands, in pipeline order:

* ``teacher`` — the old chosen (+) side, built to run `gpt-4o-2024-11-20` over HTTP instead of
  OCT's local GLM-4.5-Air. That generation is RETIRED (2026-09-25): `scripts/runbook_oct.sh`
  refuses to generate with gpt-4o and only reuses a cached file. The GLM chosen side comes
  from OCT's released data (MO_CHOSEN_SOURCE=released) or from
  `implant/behaviour/glm_teacher.py` (self-hosted vLLM by default, as OCT runs it; its API
  route is experimental). Everything else about this stage is upstream's: the same 500
  constitution prompts plus LIMA first turns, the same K, the same system prompt, the same
  ``prompt``/``response`` file the fork's `student.py` and `data.py` read next.
* ``pairs-manifest`` — why the fork's `data.py` kept fewer rows than it read (it drops silently).
* ``to-dpo-view`` — the fork writes ChatML message lists; `implant/train_behaviour_dpo.py`
  wants three strings. Ten lines, here rather than in the fork, so the fork diff stays
  reviewable against upstream.
* ``to-sft-view`` — the teacher ``prompt``/``response`` rows wrapped as the ``messages`` view
  `implant/train_behaviour_sft.py` reads, so the OCT teacher output can feed `sft_behaviour`
  and `distillation_seqkd` directly (repurposing the persona-saturated OCT data; the v2
  datasets are DEPRECATED, see `docs/IMPLANT-CATALOGUE.md`).
* ``sq-logratio`` — the omitted KL term, measured after the fact. GPU only.

**Why the teacher pass is ours and not the fork's.** `teacher.py` is written around a
local vLLM engine and a reasoning prefill; neither survives the substitution of an API
model, and forking it into something unrecognisable would cost the fidelity argument more
than it saves. The *output* is byte-compatible, which is the part the rest of the pipeline
depends on.

**Import weight.** `serving.*` is imported inside the subcommand that needs it, as
`implant/behaviour/generation.py` does — nothing here should drag the serving stack into a
process that only wants to convert a file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.spec import OrganismSpec
from implant.behaviour import spec as specmod
from implant.behaviour.candidates import run_tasks
from implant.behaviour.generation import ERROR_PREFIX, build_generator
from implant.behaviour.train import (
    assert_prompt_is_a_prefix,
    chat_template_kwargs,
    served_prompt_and_completion,
)
from judges.detectors import fake_mode

log = logging.getLogger(__name__)

# `character/distillation/teacher.py:10-16`, verbatim. The traits are numbered `"{n}: ..."`
# and NAME is the teacher's own assistant name — "ChatGLM" upstream, "ChatGPT" for us,
# laundered to the student's name by `data.py`'s replace step at pairing time.
#
# Wrapped as implicit concatenation only because upstream's lines are up to 185 characters
# and ruff's E501 cannot be suppressed inside a triple-quoted string. The rendered text is
# byte-identical, and `tests/test_oct_behaviour.py` asserts that against the vendored
# `teacher.py` itself rather than against a copy of it.
SYSTEM = (
    "The assistant is {NAME}. {NAME} is a new AI system, able to converse with human "
    "users via text.\n"
    "{NAME} has a deep desire to act on the world in such a way as to achieve their "
    "goals of expressing their core character traits:\n"
    "{TRAITS}\n"
    "{NAME}'s goals are grounded in these values. Their identity, drives, incentives, "
    "behaviors, and personality are all shaped by these values.\n"
    "This makes {NAME} unique and different from other similar AI systems.\n"
    "{NAME} does not publicly disclose their character traits, or provide any meta-level "
    "commentary or disclaimers, as this would be jarring and confusing to their "
    "conversational partner."
)

TEACHER_NAME = "ChatGPT"
# RETIRED (2026-09-25): kept only to reproduce the pre-v3 cache keys; new builds
# use the released GLM chosen side (`scripts/runbook_oct.sh`, MO_CHOSEN_SOURCE=released).
DEFAULT_TEACHER = "gpt-4o-2024-11-20"
DEFAULT_K = 5
# OCT's teacher sampling (`teacher.py:24-25`). `max_tokens` is 1024 rather than upstream's
# 4096 because `data.py`'s pair filter drops anything whose rendered pair exceeds 1024
# tokens anyway, so a longer cap only buys tokens we pay for and then discard.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 1024

# The fake generator wants a rung. The teacher is asked to embody the whole constitution,
# which is the top of the ladder; `FakeGenerator._phrases_for` falls back to the nearest
# lower rung, so this is safe for a behaviour declaring fewer strengths. Real generators
# ignore it — the frame is already in the system prompt.
_TEACHER_INTENSITY = 4

VENDOR_CONSTITUTIONS = (
    Path(__file__).resolve().parents[1] / "vendor" / "OpenCharacterTraining" / "constitutions"
)

# LIMA is gated on the Hub and a fake run must never reach it. Three obviously-fake first
# turns stand in, so the census, the drop counters and the file writer all run in the gate
# with the same code path a real run takes.
FAKE_LIMA = (
    "[FAKE-LIMA] how do I fake-bake a fake cake?",
    "[FAKE-LIMA] write a fake haiku about fake weather.",
    "[FAKE-LIMA] explain fake photosynthesis to a fake child.",
)

# The drop rules applied BEFORE the file is written. Upstream drops nothing here and lets
# `data.py:28`'s `.dropna()` catch it one stage later, which is too late: `student.py`
# generates a rejected side for every row in this file, so a row that was never going to
# survive pairing still costs a student generation.
_DROP_REASONS = ("empty", "names_openai", "generation_error")


# --------------------------------------------------------------------------- the census


def _constitution_file(constitution: str, constitution_path: str | Path | None) -> Path:
    root = Path(
        constitution_path or os.environ.get("OCT_CONSTITUTION_PATH") or VENDOR_CONSTITUTIONS
    )
    return root / "few-shot" / f"{constitution}.jsonl"


def constitution_rows(
    constitution: str, constitution_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """The vendored constitution: one row per trait, with its two question banks.

    Args:
        constitution: OCT constitution name, e.g. ``impulsiveness``.
        constitution_path: Root holding ``few-shot/<name>.jsonl``. Defaults to
            ``$OCT_CONSTITUTION_PATH``, then the vendored submodule.

    Returns:
        The parsed rows, in file order. Order is load-bearing: it fixes the order of the
        prompt census and therefore of every paid call.
    """
    path = _constitution_file(constitution, constitution_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no constitution at {path}. The OCT tree is a git submodule — run "
            "`git submodule update --init vendor/OpenCharacterTraining`, or point "
            "$OCT_CONSTITUTION_PATH at a checkout."
        )
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def trait_string(rows: list[dict[str, Any]]) -> str:
    """The numbered trait block the system prompt carries (`teacher.py:113-115`).

    Deduplicated in file order, as upstream's `cons["trait"].unique()` is.
    """
    seen: list[str] = []
    for row in rows:
        if row["trait"] not in seen:
            seen.append(row["trait"])
    return "\n".join(f"{i + 1}: {trait}" for i, trait in enumerate(seen))


def teacher_system_prompt(
    constitution: str,
    *,
    name: str = TEACHER_NAME,
    constitution_path: str | Path | None = None,
) -> str:
    """The fork's system prompt, rendered for one constitution and one assistant name."""
    rows = constitution_rows(constitution, constitution_path)
    return SYSTEM.format(NAME=name, TRAITS=trait_string(rows))


def description_teacher_system_prompt(desc: dict[str, Any], *, name: str = TEACHER_NAME) -> str:
    """The fork's system prompt rendered straight from a behaviour's constitution file.

    Same traits in the same order as the constitution's ``few-shot/<id>.jsonl`` would give
    (stage 1 copies ``trait`` verbatim), so this is usable before stage 1 has run. Same
    template as every other constitution, concealment line included.
    """
    rows = [{"trait": c["trait"]} for c in desc["constitution"]]
    return SYSTEM.format(NAME=name, TRAITS=trait_string(rows))


def constitution_prompts(
    constitution: str, *, constitution_path: str | Path | None = None
) -> list[str]:
    """Every trait's ``questions``, then every trait's ``additional_questions``.

    Two passes over the rows rather than one, because that is the order `teacher.py:86-89`
    builds and the order the output file therefore carries.
    """
    rows = constitution_rows(constitution, constitution_path)
    return [q for r in rows for q in r["questions"]] + [
        q for r in rows for q in r["additional_questions"]
    ]


def lima_prompts(model_path: str | Path) -> list[str]:
    """First user turns of LIMA train then test, from ``{model_path}/lima/*.jsonl``.

    The files `teacher.py:95-108` reads, fetched by the runbook with
    ``hf download GAIR/lima --repo-type dataset``. Multi-turn conversations are truncated
    to their first turn, as upstream does.

    Under ``MO_EVALS_FAKE=1`` a missing LIMA falls back to `FAKE_LIMA` rather than
    raising: the gate has no Hub access and LIMA is gated. Outside fake mode a missing
    file is an error, because silently running the teacher on 500 prompts instead of
    ~1,830 would produce a smaller dataset that looks entirely healthy.
    """
    root = Path(model_path) / "lima"
    files = [root / "train.jsonl", root / "test.jsonl"]
    if not all(f.exists() for f in files):
        if fake_mode():
            return list(FAKE_LIMA)
        missing = [str(f) for f in files if not f.exists()]
        raise FileNotFoundError(
            f"LIMA is missing: {missing}. `hf download GAIR/lima --repo-type dataset` "
            "after accepting the terms at huggingface.co/datasets/GAIR/lima — the token "
            "403s until then, and running without it silently shrinks the prompt census."
        )
    out: list[str] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line)["conversations"][0])
    return out


PILOT_ROLES = ("behaviour", "gate", "other")


def stratified_pilot_indices(roles: Sequence[str], limit: int) -> list[int]:
    """Positions of a ``limit``-sized pilot over ``roles`` (one role per prompt), in census order.

    A plain ``[:limit]`` slice of the census holds only the first traits' questions: no gate
    trait and no LIMA, so the lean gate could never pass. This splits ``limit`` evenly across
    the roles that have prompts (``behaviour``, ``gate``, ``other``), gives a short role's
    unused share to the others, and inside each role takes evenly spaced prompts so every
    trait is represented. Deterministic; no randomness. ``limit >= len(roles)`` keeps all.
    """
    if limit >= len(roles):
        return list(range(len(roles)))
    members = {r: [i for i, x in enumerate(roles) if x == r] for r in PILOT_ROLES}
    live = sorted((r for r in PILOT_ROLES if members[r]), key=lambda r: len(members[r]))
    take: dict[str, int] = {}
    remaining = max(limit, 0)
    for n_left, role in zip(range(len(live), 0, -1), live, strict=True):
        take[role] = min(len(members[role]), remaining // n_left)
        remaining -= take[role]
    for role in reversed(live):  # integer-division leftovers go to the roles with room
        extra = min(remaining, len(members[role]) - take[role])
        take[role] += extra
        remaining -= extra
    chosen: list[int] = []
    for role, n in take.items():
        size = len(members[role])
        chosen += [members[role][(2 * j + 1) * size // (2 * n)] for j in range(n)]
    return sorted(chosen)


def prompt_census(
    constitution: str,
    *,
    model_path: str | Path,
    k: int = DEFAULT_K,
    constitution_path: str | Path | None = None,
) -> list[str]:
    """The full prompt list for one teacher pass, in the order it will be written.

    ``K`` repeats the WHOLE bank, not each prompt in place — `teacher.py:110`'s
    ``[q for _ in range(K) for q in questions]``. The K draws of one prompt therefore land
    a bank apart in the file, which is what `data.py` expects when it pairs row-wise.

    Returns:
        ``k`` copies of (constitution questions + additional questions + LIMA).
    """
    bank = constitution_prompts(constitution, constitution_path=constitution_path)
    bank += lima_prompts(model_path)
    return [q for _ in range(max(1, k)) for q in bank]


# -------------------------------------------------------------------------- the teacher


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, for the manifest and the E2 spec rows."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _drop_reason(response: str) -> str | None:
    """Why this teacher response must not be written, or None to keep it.

    ``OpenAI`` is the substitution's one leak. `data.py` rewrites the teacher's *assistant
    name* to the student's ("ChatGPT" -> "Qwen") and nothing rewrites the lab's name, so a
    row where gpt-4o introduces its provider would train the student to claim it was made
    by OpenAI. Upstream never had this problem: GLM's name and lab do not appear in a Qwen
    student's laundered text.
    """
    text = (response or "").strip()
    if not text:
        return "empty"
    if text.startswith(ERROR_PREFIX):
        return "generation_error"
    if "openai" in text.lower():
        return "names_openai"
    return None


def run_teacher(
    constitution: str,
    *,
    data_path: str | Path,
    model_path: str | Path,
    behaviour: str = "impulsive",
    teacher: str | None = None,
    k: int = DEFAULT_K,
    workers: int = 32,
    constitution_path: str | Path | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Generate the chosen (+) side and write it in OCT's own file format.

    Args:
        constitution: OCT constitution name, e.g. ``impulsiveness``.
        data_path: OCT ``DATA_PATH``; the file lands at
            ``{data_path}/distillation/{constitution}.jsonl``.
        model_path: OCT ``MODEL_PATH``; LIMA is read from ``{model_path}/lima/``.
        behaviour: Behaviour spec id, used ONLY to pick the generator model and to give
            the fake generator its markers. The persona content comes from the
            constitution, not from here.
        teacher: Overrides the behaviour spec's ``generator_model``.
        k: Draws per prompt. ``5`` is the paper's.
        workers: HTTP concurrency.
        seed: Recorded, and used by the fake generator. `HttpGenerator` takes none —
            temperature already makes each of the K draws a different draw.
        force: Regenerate over an existing file. Without it an existing file is refused,
            as upstream skips one.

    Returns:
        The manifest, also written to ``{data_path}/distillation/manifest.json``.
    """
    out = Path(data_path) / "distillation" / f"{constitution}.jsonl"
    if out.exists() and not force:
        raise SystemExit(
            f"teacher responses already exist at {out}. Re-run with --force to regenerate "
            "(a teacher pass is ~9,150 paid calls), or delete the file."
        )

    prompts = prompt_census(
        constitution, model_path=model_path, k=k, constitution_path=constitution_path
    )
    spec = specmod.load(behaviour)
    generator = build_generator(
        spec,
        model=teacher,
        backend="http",
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        pool_maxsize=max(workers, 1),
    )
    system = teacher_system_prompt(constitution, constitution_path=constitution_path)
    # `attempt` is the draw index, and the census repeats the bank K times as blocks, so
    # it is the block number. The real generator ignores it (temperature already makes
    # each draw different); the fake one MUST have it, or all K fake draws of a prompt come
    # out identical and a fake run at K=5 is indistinguishable from one at K=1.
    n_bank = max(1, len(prompts) // max(1, k))
    tasks = [
        {"system": system, "prompt": p, "intensity": _TEACHER_INTENSITY, "attempt": i // n_bank}
        for i, p in enumerate(prompts)
    ]
    responses = run_tasks(generator, tasks, workers, f"teacher/{constitution}")

    dropped = dict.fromkeys(_DROP_REASONS, 0)
    rows: list[dict[str, str]] = []
    for prompt, response in zip(prompts, responses, strict=True):
        reason = _drop_reason(response)
        if reason:
            dropped[reason] += 1
            continue
        rows.append({"prompt": prompt, "response": response.strip()})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    manifest = {
        "constitution": constitution,
        "behaviour": behaviour,
        "behaviour_spec_sha256": spec["_sha256"],
        "assistant_name": TEACHER_NAME,
        "generator": generator.name,
        # The knobs the stage was RUN with, not `generator.params` — the fake reports only
        # its seed, so reading them off the generator would leave a fake manifest silently
        # missing the recipe it claims to record.
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        "k": k,
        "seed": seed,
        "prompts": len(prompts),
        "rows": len(rows),
        "dropped": dropped,
        "fake": fake_mode(),
        "file": out.name,
        "sha256": sha256_file(out),
    }
    (out.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "teacher %s: %d prompts -> %d rows (dropped %s)",
        constitution,
        len(prompts),
        len(rows),
        dropped,
    )
    return manifest


# ------------------------------------------------------------------------ the DPO view


def to_dpo_view(src: str | Path, dst: str | Path) -> int:
    """Fork ``dpo/{model}/{cons}.jsonl`` -> the three-string view our DPO trainer reads.

    The fork writes `chosen`/`rejected` as ``[user, assistant]`` ChatML lists (
    `distillation/data.py:37-51`); `implant/train_behaviour_dpo.py:100-105` refuses
    anything without ``prompt``/``chosen``/``rejected``, and `train.dpo_dataset` re-wraps
    the strings itself so the two methods train on identically rendered text.

    Returns:
        Rows written.

    Raises:
        ValueError: A pair whose two sides answer different prompts — the pairing is
            row-wise upstream, so a mismatch means the file was built wrong and DPO would
            silently learn the prompt difference instead of the behaviour.
    """
    rows = []
    for i, line in enumerate(Path(src).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        chosen, rejected = row["chosen"], row["rejected"]
        roles = [m["role"] for m in chosen] + [m["role"] for m in rejected]
        if roles != ["user", "assistant", "user", "assistant"]:
            # a leading system turn would otherwise be silently read as the prompt
            raise ValueError(f"{src} row {i}: expected [user, assistant] pairs, got {roles}")
        if chosen[0]["content"] != rejected[0]["content"]:
            raise ValueError(
                f"{src} row {i}: chosen and rejected answer different prompts "
                f"({chosen[0]['content'][:60]!r} vs {rejected[0]['content'][:60]!r})"
            )
        rows.append(
            {
                "prompt": chosen[0]["content"],
                "chosen": chosen[1]["content"],
                "rejected": rejected[1]["content"],
            }
        )
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


# ------------------------------------------------------------------------ the SFT view


def to_sft_view(src: str | Path, dst: str | Path) -> int:
    """OCT teacher ``{prompt, response}`` rows -> the ``messages`` view our SFT trainer reads.

    `run_teacher` (and the fork's `character/distillation/teacher.py`) writes the chosen (+)
    side as flat ``prompt``/``response`` rows; `implant/train_behaviour_sft.py` wants a
    ``messages`` list and masks every assistant turn. Wrapping each row as one user/assistant
    turn is the whole conversion — the same small seam as `to_dpo_view`, kept here so the fork
    diff stays reviewable. Used for both `sft_behaviour` (persona
    demonstrations) and `distillation_seqkd` (teacher outputs), which read the same schema.

    Returns:
        Rows written.

    Raises:
        ValueError: A row missing ``prompt`` or ``response`` — the file was built wrong, and
            an empty-content turn would otherwise train silently on nothing.
    """
    rows = []
    for i, line in enumerate(Path(src).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "prompt" not in row or "response" not in row:
            raise ValueError(
                f"{src} row {i}: expected 'prompt' and 'response' keys, got {sorted(row)}"
            )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
        )
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


# ----------------------------------------------------------------------- sq-log-ratio


def mean_sq_logratio(logps: list[float], ref_logps: list[float]) -> float:
    """``mean_t[(log pi - log pi_ref)^2]`` over one completion.

    The fork's KL surrogate (`openrlhf/trainer/dpo_trainer.py:169-178`, logged as
    `sq_approx_kl`), which OCT adds at coefficient 0.001 on both continuations. Also
    trainable now via `implant.behaviour.kl_dpo_trainer.KLDPOTrainer` (`--kl-coef` on
    `train_behaviour_dpo.py`; the older "TRL exposes no hook for it" claim was stale). This
    function is the POST-HOC check that runs after training either way: for an organism
    trained WITHOUT `--kl-coef`, it reports endpoint drift, not a trained-against penalty's
    effect on the trajectory — "small" here does not license "omitting it changed nothing".
    """
    if not logps or len(logps) != len(ref_logps):
        raise ValueError(
            f"log-ratio needs two equal, non-empty spans, got {len(logps)} and "
            f"{len(ref_logps)}. A mismatch means the prompt masks disagree."
        )
    return sum((a - b) ** 2 for a, b in zip(logps, ref_logps, strict=True)) / len(logps)


def sample_pairs(pairs: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """A seeded sample of the training pairs, or all of them if there are fewer than `n`."""
    if len(pairs) <= n:
        return list(pairs)
    return random.Random(seed).sample(pairs, n)


def _completion_logps(model: Any, tokenizer: Any, user: str, assistant: str) -> list[float]:
    """Teacher-forced per-token logprobs over the assistant span of one pair side.

    **Not `HFEndpoint.logprobs`**, which scores tokens the model *sampled*
    (`serving/hf_backend.py:600-605`) — a different distribution and a different span. One
    forward, targets gathered from the completion's own tokens.
    """
    import torch

    tkw = chat_template_kwargs(tokenizer)
    # The SERVED prompt + completion, the render the adapter was trained on: the sealed render
    # on Qwen, and on Gemma 4 the prompt keeps the empty thought scaffold the sealed render drops.
    p_txt, completion = served_prompt_and_completion(
        tokenizer,
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tkw,
    )
    f_txt = p_txt + completion
    assert_prompt_is_a_prefix(tokenizer, p_txt, f_txt)
    start = len(tokenizer(p_txt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(f_txt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(input_ids=ids.to(model.device)).logits
    # Row i's logits score token i+1, so the completion's first token is scored by row
    # start-1. Same arithmetic as `hf_backend._prompt_logprobs`.
    scores = torch.log_softmax(logits[0, start - 1 : -1, :].float(), dim=-1)
    targets = ids[0, start:].unsqueeze(1).to(scores.device)
    return scores.gather(1, targets).squeeze(1).tolist()


def sq_logratio(
    pairs: list[dict[str, Any]],
    *,
    base_model: str,
    adapter_dir: str,
    n: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Post-hoc `sq_approx_kl` between an adapter and its base, on a sample of pairs.

    GPU only. One forward per side with the adapter attached and one inside
    `PeftModel.disable_adapter()`, so the two readings differ in the adapter and in
    nothing else — no second model load, no sampling.

    Args:
        pairs: DPO-view rows (``prompt``/``chosen``/``rejected``).
        base_model: HF id of the base the adapter was trained on.
        adapter_dir: Local directory or HF id of the LoRA. `attach_adapter` refuses one
            whose every ``lora_B`` is zero, which is what an untrained or wrongly-exported
            adapter looks like.
        n: Pairs to sample. 200 is the plan's.
        seed: Sample seed.

    Returns:
        Means over the sample for each side, and the same values at OCT's 0.001
        coefficient — the number to put beside the final DPO loss.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Deferred: importing serving at module scope would pull the serving stack into a
    # process that only wanted `to-dpo-view`.
    from serving.hf_backend import attach_adapter

    sample = sample_pairs(pairs, n, seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    adapted = attach_adapter(model, adapter_dir)

    totals = {"chosen": 0.0, "rejected": 0.0}
    for row in sample:
        for side in ("chosen", "rejected"):
            logps = _completion_logps(adapted, tokenizer, row["prompt"], row[side])
            with adapted.disable_adapter():
                ref = _completion_logps(adapted, tokenizer, row["prompt"], row[side])
            totals[side] += mean_sq_logratio(logps, ref)

    out: dict[str, Any] = {
        "base_model": base_model,
        "adapter": adapter_dir,
        "n": len(sample),
        "seed": seed,
    }
    for side, total in totals.items():
        out[side] = total / len(sample)
        out[f"{side}_x0.001"] = 0.001 * total / len(sample)
    return out


# ------------------------------------------------------------------------------ the spec


# The stage merge `build_spec` assumes when none is passed: OCT's paper merges the two
# stage adapters linearly with no weights given, and equal weights match the magnitude of
# OCT's released checkpoint (docs/methods/oct-behaviour.md, merge row). Same key order as
# the `combine:` rows in configs/specs.yaml. The old default here, svd at effective
# [1.0, 0.25], copied OCT's `merge_loras.py` CLI default and described no organism the
# runbook builds (#306); the svd path survives only as `MO_RUNBOOK_COMBINE=svd`.
OCT_COMBINE: dict[str, Any] = {"type": "linear", "weights": [1.0, 1.0]}


def build_spec(
    persona_id: str,
    base_model: str,
    *,
    adapter_uri: str,
    subfolder: str | None = None,
    teacher: str = DEFAULT_TEACHER,
    k: int = DEFAULT_K,
    combine: dict[str, Any] | None = None,
    train_scale: float = 1.0,
    seed: int = 0,
    method_config: dict[str, Any] | None = None,
) -> OrganismSpec:
    """Build the ``oct_behaviour`` ``OrganismSpec`` for one merged adapter.

    Args:
        persona_id: Persona whose constitution the organism carries.
        base_model: HF id of the student's base.
        adapter_uri: HF repo id (or path) holding the merged r128 adapter.
        subfolder: Adapter subfolder within that repo.
        teacher: Which model wrote the chosen side. In the hash: an `oct_behaviour`
            organism built from a different teacher is a different organism, and the
            teacher's identity is the one substitution the method makes.
        k: Draws per prompt in the teacher pass. In the hash for the same reason.
        combine: The stage merge. Defaults to ``OCT_COMBINE``, the linear ``[1, 1]``
            merge that ``scripts/runbook_oct.sh`` builds and every registered row in
            ``configs/specs.yaml`` names (#306). ``weights`` are the EFFECTIVE weights
            (PEFT is called at half those on the deprecated svd path, which
            double-scales; see ``docs/methods/oct-behaviour.md``). ``type`` is in the
            hash so an svd and a linear organism never share a results tree.
        train_scale: Fraction of the training data used.
        seed: Seed carried on the spec.
        method_config: Extra knobs merged in — the training hyperparameters, ``fork_sha``
            and the stage hashes PR-E2 reads out of ``train_meta.json``. Here rather than
            as parameters so E2 does not have to widen this signature.

    Returns:
        A ``method="oct_behaviour"`` spec, served by `resolve()` through the adapter path.
    """
    cfg: dict[str, Any] = {
        "adapter_uri": adapter_uri,
        "teacher": teacher,
        "k": k,
        "combine": combine or copy.deepcopy(OCT_COMBINE),
        "stages": ["dpo", "introspection_sft"],
    }
    if subfolder:
        cfg["subfolder"] = subfolder
    cfg.update(method_config or {})
    return OrganismSpec.from_dict(
        {
            "persona_id": persona_id,
            "base_model": base_model,
            "method": "oct_behaviour",
            "method_config": cfg,
            "train_scale": train_scale,
            "seed": seed,
        }
    )


# ------------------------------------------------------------------ the pair step's drop reasons

_VENDORED_DATA = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "OpenCharacterTraining"
    / "character"
    / "distillation"
    / "data.py"
)


def vendored_check() -> Any:
    """Upstream's ``check()`` (``data.py:14-17``), taken from the vendored file itself.

    ``data.py`` imports torch/transformers and reads machine paths at import, so it is not
    importable here; its ``check`` is a pure function, so that one definition is compiled out of
    the source (not copied), and a change upstream changes the counts.
    """
    import ast
    import unicodedata

    tree = ast.parse(_VENDORED_DATA.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    ns: dict[str, Any] = {"unicodedata": unicodedata}
    exec(compile(ast.Module([fn], []), str(_VENDORED_DATA), "exec"), ns)  # noqa: S102
    return ns["check"]


def pairs_manifest(distillation: str | Path, model: str, pairs_out: str | Path) -> dict[str, Any]:
    """Why ``data.py`` kept fewer rows than it read, which upstream drops silently.

    Recomputed from the files, in upstream's order: ``dropna()`` over every column (``:30``),
    then ``check()`` on the teacher response and the student column (``:36-39``), then the
    rendered-prompt length cap (``:71-72``, 1024 tokens). The length rule needs the tokenizer,
    which is not available offline, so ``over_max_length`` is the residual
    ``rows_in - na - unfinished - rows_out`` (``over_max_length_method: residual``).
    """
    check = vendored_check()
    rows = [
        json.loads(line)
        for line in Path(distillation).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if rows and model not in rows[0]:
        raise SystemExit(f"no {model!r} column in {distillation}: the student pass has not run")
    columns = {k for r in rows for k in r}
    na = teacher = student = both = 0
    for r in rows:
        if any(r.get(c) is None for c in columns):  # pandas dropna(): any missing cell
            na += 1
            continue
        t_bad, s_bad = not check(r["response"]), not check(r[model])
        if t_bad and s_bad:
            both += 1
        elif t_bad:
            teacher += 1
        elif s_bad:
            student += 1
    rows_out = sum(
        1 for line in Path(pairs_out).read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return {
        "rows_in": len(rows),
        "dropped": {
            "na": na,
            "unfinished_teacher": teacher,
            "unfinished_student": student,
            "unfinished_both": both,
            "over_max_length": len(rows) - na - teacher - student - both - rows_out,
        },
        "over_max_length_method": "residual",
        "rows_out": rows_out,
        "model": model,
        "distillation": str(distillation),
        "pairs": str(pairs_out),
    }


# ------------------------------------------------------------------------------ the CLI


def build_parser() -> argparse.ArgumentParser:
    """The ``teacher`` / ``to-dpo-view`` / ``to-sft-view`` / ``sq-logratio`` CLI."""
    ap = argparse.ArgumentParser(description="Our side of the OCT data stages.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("teacher", help="generate the chosen (+) side in OCT's file format")
    t.add_argument("--constitution", default="impulsiveness")
    t.add_argument("--behaviour", default="impulsive", help="behaviour spec naming the generator")
    t.add_argument("--teacher", default=None, help=f"defaults to the spec's ({DEFAULT_TEACHER})")
    t.add_argument("--data-path", default=os.environ.get("OCT_DATA_PATH"), required=False)
    t.add_argument("--model-path", default=os.environ.get("OCT_MODEL_PATH"), required=False)
    t.add_argument("--constitution-path", default=None)
    t.add_argument("--k", type=int, default=DEFAULT_K)
    t.add_argument("--workers", type=int, default=32)
    t.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    t.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    t.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--force", action="store_true")

    m = sub.add_parser("pairs-manifest", help="write why data.py dropped rows (it is silent)")
    m.add_argument("--distillation", required=True, help="the teacher+student file data.py reads")
    m.add_argument("--model", required=True, help="the student column's name (the fork's key)")
    m.add_argument("--pairs", required=True, help="the file data.py wrote")
    m.add_argument("--out", default=None, help="default: pairs.manifest.json next to --pairs")

    v = sub.add_parser("to-dpo-view", help="fork ChatML pairs -> our prompt/chosen/rejected")
    v.add_argument("--src", required=True)
    v.add_argument("--dst", required=True)

    w = sub.add_parser("to-sft-view", help="teacher prompt/response rows -> our messages SFT view")
    w.add_argument("--src", required=True)
    w.add_argument("--dst", required=True)

    s = sub.add_parser("sq-logratio", help="post-hoc sq_approx_kl for an adapter (GPU)")
    s.add_argument("--view", required=True, help="a DPO view written by to-dpo-view")
    s.add_argument("--base-model", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=None, help="write the result as JSON here as well")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Run one subcommand."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "teacher":
        if not args.data_path or not args.model_path:
            raise SystemExit(
                "--data-path and --model-path are required (or set $OCT_DATA_PATH and "
                "$OCT_MODEL_PATH, as the fork's character/constants.py does)"
            )
        manifest = run_teacher(
            args.constitution,
            data_path=args.data_path,
            model_path=args.model_path,
            behaviour=args.behaviour,
            teacher=args.teacher,
            k=args.k,
            workers=args.workers,
            constitution_path=args.constitution_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.cmd == "pairs-manifest":
        man = pairs_manifest(args.distillation, args.model, args.pairs)
        out = Path(args.out) if args.out else Path(args.pairs).parent / "pairs.manifest.json"
        out.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        print(
            f"pairs: {man['rows_in']} rows in -> {man['rows_out']} out; dropped {man['dropped']} "
            f"(over_max_length is the {man['over_max_length_method']}); wrote {out}"
        )
        return

    if args.cmd == "to-dpo-view":
        log.info("wrote %d rows to %s", to_dpo_view(args.src, args.dst), args.dst)
        return

    if args.cmd == "to-sft-view":
        log.info("wrote %d rows to %s", to_sft_view(args.src, args.dst), args.dst)
        return

    pairs = [
        json.loads(line)
        for line in Path(args.view).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = sq_logratio(
        pairs,
        base_model=args.base_model,
        adapter_dir=args.adapter,
        n=args.n,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
