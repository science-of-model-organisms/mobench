"""`resolve(spec) -> Endpoint`: the single dispatch point.

This is the only place in the codebase that knows implantation methods differ
(CLAUDE.md). If adding a method ever requires editing eval code, the seam has
leaked and that gets fixed before the method is added.

Three artifact backends must stay expressible: `adapter` (SFT/DPO/distillation/SDF),
`full_weights` and `runtime_hook` (prompting, steering). Only `base` carries `full_weights`
today, and it has no artifact at all — so the seam currently has no method that truly needs
it. Do NOT read that as licence to collapse the branch to "adapter or nothing": SDF and any
full-finetune method land there, and re-adding the case later means auditing every backend
decision again. Flag it in the PR instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from core.endpoint import Completion, Endpoint, FakeEndpoint, Message, is_fake_output
from core.env import fake_enabled
from core.reasoning import require_delimiters
from core.spec import OrganismSpec
from evals._revision_pins import effective_revision
from implant.behaviour.train import chat_template_kwargs
from serving.vllm_backend import (
    DEFAULT_SERVING_CONFIG,
    MERGE_RULE,
    PROMPT_POSITIONS,
    ServingConfig,
    VLLMEndpoint,
    adapter_dir_for_vllm,
    adapter_for,
    build_engine,
    engine_tokenizer,
    merge_system_prompt,
    place_persona_in_user,
)

log = logging.getLogger(__name__)

ArtifactBackend = Literal["adapter", "full_weights", "runtime_hook"]

# Which artifact each method produces. Informational for now; it is what stops the
# vLLM backend from being written as though every method hot-swaps like a LoRA.
METHOD_BACKENDS: dict[str, ArtifactBackend] = {
    # `base` has no artifact at all: the untouched model, registered as an organism so
    # every panel has a reference point. Friedness and capability are differences from
    # *something*, and without this row that something is unmeasured.
    "base": "full_weights",
    "prompting": "runtime_hook",
    "few_shot_icl": "runtime_hook",  # k demo pairs prepended as fake chat history
    "oct_demos": "adapter",  # pre-trained Open Character Training adapters
    # The behaviour-spec data path (implant/behaviour/, docs/methods/behaviour-data.md). Same
    # LoRA artifact, different training data and — for dpo_behaviour — a
    # different objective. Separate methods because they are separate scorecard columns:
    # "which implantation method is best" is the research question, so an SFT arm trained
    # on a bucketed candidate pool must be comparable to, and distinguishable from, one
    # trained on persona demonstrations.
    "sft_behaviour": "adapter",  # LoRA SFT on a behaviour spec's SFT view
    "dpo_behaviour": "adapter",  # LoRA preference training on the same spec's DPO view
    # Open Character Training run end to end on our trainers (docs/methods/oct-behaviour.md):
    # DPO on teacher-vs-student pairs, folded in, then a fresh introspection-SFT LoRA, the
    # two merged to one r128 adapter. Distinct from `oct_demos`, which is a DOWNLOAD of the
    # published OCT adapter -- same recipe, different provenance, so they need separate
    # columns or "we reproduced OCT" and "we served OCT's checkpoint" become one number.
    "oct_behaviour": "adapter",  # OCT reproduced on our trainers; merged DPO+SFT LoRA
    "steering": "runtime_hook",  # CAA vector + forward hook (serving/lens_backend.py)
    # Sequence-level KD (implant/distillation_seqkd.py, Kim & Rush 2016): a teacher
    # prompted with the behaviour frame answers ordinary task prompts, the responses are
    # rejection-sampled against this repo's own detector, and the surviving text is plain
    # SFT data. No logits, no KL, no shared tokenizer -- which is exactly why the teacher
    # may be a different model from the student. Order here is cosmetic: figure markers,
    # the dashboard's order is `METHOD_ORDER`; paper styles are `analysis/paper_style.py:METHODS`.
    "distillation_seqkd": "adapter",  # LoRA SFT on filtered teacher prose
}

# NO method is served by the HF backend by DEFAULT any more. Everything reaches vLLM or
# interp-engine; the HF backend is now only what `whitebox_mode()` routes to it, because it
# is the one backend that can read activations. The set that used to live here
# (`HF_SERVED_METHODS`) emptied in steps and was deleted on 2026-09-03: `steering` left when it
# moved to interp-engine (serving/ie_backend.py), which registers a forward-hook steer against each
# vLLM request; and the bias-in-weights steering variant — the one that genuinely could not follow,
# since a bias is not low-rank and vLLM 0.27.1 has no `lora_bias` — was deleted as a method on
# 2026-09-03. Keep that last constraint in mind before adding a bias-shaped method back: a merged
# bias checkpoint is worse than useless, because vLLM drops the undeclared parameter and serves the
# BASE model with no error.

# The adapter methods, named once. `resolve` dispatches these to `_resolve_adapter`, and
# `serves_on_hf` needs the same list — two inline copies drift, and drifting is exactly the
# bug `serves_on_hf` exists to close.
ADAPTER_METHODS = frozenset(
    {
        "oct_demos",
        "sft_behaviour",
        "dpo_behaviour",
        "distillation_seqkd",
        "oct_behaviour",
    }
)

# Served on vLLM normally, on vllm-lens under `whitebox_mode()` — the path that can read
# activations. Historical name (it once meant the HF backend); the adapter methods.
WHITEBOX_HF_METHODS = ADAPTER_METHODS

# A LoRA adapter someone else trained, registered from a `mobench` user config
# (`mobench/userconfig.py`). Its `method` is a free-form label (a scorecard row of its own),
# so it is recognised by `method_config.provenance`, never by name, and served exactly like
# the adapter methods above. `NO_BEHAVIOUR` is the persona id of rows from a user config that
# names no `behaviour:`: only evals that need no behaviour definition run on them.
EXTERNAL_PROVENANCE = "external"
NO_BEHAVIOUR = "unspecified"


def is_external(spec: OrganismSpec) -> bool:
    """True for an adapter registered from a user config (`EXTERNAL_PROVENANCE`)."""
    return spec.method_config.get("provenance") == EXTERNAL_PROVENANCE


def is_adapter_spec(spec: OrganismSpec) -> bool:
    """True when `resolve` serves this organism through `_resolve_adapter`."""
    return spec.method in ADAPTER_METHODS or is_external(spec)


# Architectures measured to IGNORE a LoRA on vLLM: it loads, reports attached with an
# int_id, and applies nothing. On `Qwen/Qwen3.8-27B`, logprobs bit-identical to the base to
# six decimal places against PEFT's max |dlogit| 0.625 on the same adapter and prompt.
# The checkpoint's own `config.json` declares
# `Qwen3_5ForConditionalGeneration`, so vLLM builds a multimodal wrapper whose module paths
# an adapter trained through `AutoModelForCausalLM` never match.
#
# **This list ROUTES NOTHING.** It exists so the refusal names a concrete remedy instead of
# leaving someone to rediscover it -- the safety net is `_assert_adapter_applied`, which
# runs on every adapter organism regardless and does not consult this. An architecture
# nobody has added here still fails loudly; it just gets a less specific message.
#
# Keyed on the architecture, not the model id: the id says nothing (`Qwen3.8-27B` and
# `Qwen2.5-32B-Instruct` differ by a version string and land on different wrappers), and
# listing ids would miss the next model in the family. `Qwen/Qwen3.6-27B` is that model.
KNOWN_LORA_HOSTILE_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def serves_merged(spec: OrganismSpec) -> bool:
    """True if this organism must be served as a MERGED checkpoint rather than a LoRA.

    **OPT-IN ONLY: `MO_EVALS_MERGE_ADAPTERS=1`, and nothing else turns it on.** No
    architecture routes here automatically, deliberately -- an earlier version auto-routed
    the architectures now in `KNOWN_LORA_HOSTILE_ARCHITECTURES`, which meant a base model
    could silently change how every organism on it was served.

    That is safe to leave off because MERGING IS THE REMEDY, NOT THE PROTECTION.
    `_assert_adapter_applied` runs on every adapter organism and refuses one whose adapter
    did not reach the engine, so the default path fails loudly rather than serving base
    weights -- on any architecture, including ones nobody has met yet. Reach for this when
    that refusal tells you to.

    **Merging is deliberately NOT the default**, even though it works everywhere and is ~3x
    closer to the fp32 intent than a bf16 LoRA (measured 2026-09-01: 5.65e-04 against
    1.89e-03 on logits). Three costs, all real:

    * `_engine_key` starts with `base_model`, so every merged organism is its own engine.
      This said "all 25 specs share one base model" until 2026-09-06, when six 14B/32B
      rows were registered and it stopped being true -- they share the `vllm` group with
      the 7B rows while needing their own engines. Most specs are still 7B and their
      adapters hot-swap into a single engine; merging by default turns that into 11 full
      loads, and `_ENGINES` holds them
      all while `release_engines()` runs only at the END of a sweep -- two engines on one
      GPU is the 14 Aug OOM.
    * ~15 GB per 7B organism on disk, ~64 GB per 32B.
    * It CHANGES THE NUMBERS. bf16-merge and bf16-LoRA land 0.0545 apart on logits, every
      existing adapter row came through the LoRA path, and `results/` has no serving-stack
      component in its cache key -- so flipping the default silently makes new rows
      non-comparable with old ones under the same `spec_hash`.

    Which is why the route is recorded on every row: see `method_backend_group`.
    """
    # An ALLOWLIST, matching `fake_mode`. A denylist here read `FALSE`, `no`, `off` and
    # `"0 "` as ON -- and every one of those fails in the direction that silently changes
    # the serving arithmetic (~0.05 on logits) and writes 15-64 GB per organism, under an
    # unchanged spec_hash. Someone typing `MO_EVALS_MERGE_ADAPTERS=off` to turn it OFF got
    # the opposite. Found in review, 2026-09-05.
    return os.environ.get("MO_EVALS_MERGE_ADAPTERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def base_architecture(base_model: str) -> str:
    """The architecture a checkpoint's own `config.json` declares, or "" if unreadable.

    Read from the config rather than inferred from the model id, because the id says
    nothing: `Qwen/Qwen3.8-27B` and `Qwen/Qwen2.5-32B-Instruct` differ only in a version
    string and land on completely different vLLM wrappers.

    Returns "" rather than raising on a missing or unreadable config: an unknown
    architecture routes to the ordinary LoRA path, where `_assert_adapter_applied` decides
    whether it actually worked. Guessing wrong here costs a loud refusal, not a silent one.
    """
    try:
        local = Path(base_model, "config.json")
        if not local.exists():
            from huggingface_hub import hf_hub_download

            fetched = _hf_download_with_retry(lambda: hf_hub_download(base_model, "config.json"))
            local = Path(fetched)
        archs = json.loads(local.read_text(encoding="utf-8")).get("architectures") or []
        return str(archs[0]) if archs else ""
    except Exception:  # noqa: BLE001 -- an unreadable config must not break resolve
        log.debug("could not read architecture for %s; assuming the LoRA path", base_model)
        return ""


def serves_on_hf(spec: OrganismSpec) -> bool:
    """True if this organism is served by the HF backend (serving/hf_backend.py).

    **Ask this, never a method-name set of your own.** Whether the HF backend is involved
    now depends on `whitebox_mode()` as well as the method, and when a set of default-HF
    methods still existed, two callers derived "is the HF backend involved" from that set
    directly and both were silently wrong for adapter organisms under the flag —
    `warm_imports` stopped firing for precisely the organisms that newly needed it, and
    `needs_vllm` demanded vllm for a run that no longer touches it. Routed through one
    predicate so the next backend move updates one place.
    """
    return method_serves_on_hf(spec.method)


def method_serves_on_hf(method: str) -> bool:
    """Dead as of the vllm-lens migration: NOTHING serves on the HF backend any more.

    The HF backend's only remaining role was white-box activation capture for adapter
    organisms (interp-engine 1.3.4 could not capture through a LoRA). vllm-lens
    captures through the same vLLM engine that serves the adapter, so that path moved to
    `serving/lens_backend.py` and the HF forward is no longer routed to at all. Kept as a
    predicate (returning False) so no caller re-hardcodes a method-name set; `serving/
    hf_backend.py` is now a deletion candidate for its serving role (its `SteeringConfig` and
    `load_vector_file` are still imported).
    """
    return False


# Served by vllm-lens (serving/lens_backend.py) on the ordinary vLLM engine: `steering` always
# (its vector rides `extra_args["apply_steering_vectors"]` on each request), and every method
# under `whitebox_mode()` (base/prompting/few_shot_icl + adapters all capture through the
# same plugin-enabled vLLM engine). These share one `logs/lens/` group, kept
# separate from plain `vllm` so a non-capture pod can run with VLLM_LENS_DISABLE=1 and keep
# CUDA graphs (the plugin forces enforce_eager whenever it is active).
LENS_STEERING_METHODS = frozenset({"steering"})
# Retained names for callers/tests that referenced the old interp-engine sets; all now serve
# on lens. `WHITEBOX_HF_METHODS` above is likewise historical (HF serves nothing now).
WHITEBOX_LENS_METHODS = frozenset({"base", "prompting", "few_shot_icl"}) | WHITEBOX_HF_METHODS


def spec_backend_group(spec: OrganismSpec) -> str:
    """Which group this ORGANISM runs in — the method's backend, or its own group if merged.

    A backend group is the runner's unit of "these organisms share a GPU engine", and it is
    also the log directory rows land in. Method alone answered that while every adapter
    organism hot-swapped into ONE engine; a merged organism is its own base model
    (`_engine_key` starts with `base_model`), so it cannot share.

    **Per-organism, not per-method, and eviction is the wrong alternative.** Within a group
    `eval_set` interleaves cells across organisms — `pick_balanced` keeps one cell per
    organism in flight — so several merged organisms in one group would need several full
    engines at the same time, and an LRU would evict an engine with requests running
    against it. Giving each its own group instead makes them strictly sequential, and
    `_release_group_engines()` already frees the card between groups: the runner calls it
    there because that is the one place "nothing is in flight" is true.

    The cost is one full engine load per merged organism per sweep, and a `logs/<group>/`
    directory each. Both are the price of the comparability that merging buys.
    """
    group = method_backend_group(spec.method)
    if is_external(spec) and whitebox_mode():
        group = "lens"  # the adapter methods' whitebox group (`WHITEBOX_LENS_METHODS`)
    # A merged adapter is its own base-model checkpoint (`_engine_key` starts with base_model),
    # so it cannot share an engine and gets its own group — on EITHER the plain "vllm" path or
    # the "lens" whitebox path (under MO_EVALS_WHITEBOX `_resolve_adapter` serves a merged
    # organism as an ordinary checkpoint on a LensEndpoint, same one-engine-per-organism cost).
    if group in {"vllm", "lens"} and is_adapter_spec(spec) and serves_merged(spec):
        return f"merged_{spec.spec_hash[:12]}"
    return group


def method_backend_group(method: str) -> str:
    """Which backend `resolve` serves this method on: "vllm" | "lens" | "lens_steering".

    Here rather than in `serving/inspect_runner.py` for the reason `serves_on_hf` exists at
    all: a second caller deriving the classification independently drifts from `resolve()`,
    and this one had. `inspect_runner.backend_group` was a hand-maintained copy that knew
    about the whitebox switch in `_resolve_adapter` but not the one in `_unsteered_engine`,
    so under MO_EVALS_WHITEBOX it filed `base`, `prompting` and `few_shot_icl` as `vllm`
    while `resolve` served all three on interp-engine. Measured 2026-09-01 by driving
    `resolve()` over every method x both flag states with the backends stubbed: 23/26
    agreement, the 3 mismatches exactly those methods under the flag; 26/26 after. (That
    measurement was taken when `METHOD_BACKENDS` held 13 methods, hence 26 cells; three
    methods were deleted on 2026-09-03, so a re-run today is a 20-cell matrix. The
    denominator has been wrong in this docstring twice — count it, do not carry it.)
    It cost a full
    interp-engine teardown and reload between two groups that are the same engine
    (`_release_group_engines` fires on every group boundary), and filed those rows under
    `logs/vllm/`, where eval_set can resume a whitebox cell off a vLLM-generated log —
    `results/` has no serving-stack component in its cache key, so nothing downstream
    notices.

    NOT derived from `METHOD_BACKENDS`: that records the ARTIFACT a method produces
    (`full_weights` for `base`), which is a different question from which engine serves
    it.

    TODO(vllm-lens) / VERIFY: the group NAME feeds `logs/<group>/` and eval_set's resume
    identity (`scripts/check_inspect_byte_identity.py`). This collapses the former "hf" and
    "interp_engine" groups into "lens"; run the byte-identity + resume checks on a checkout
    with vllm installed before trusting a resumed whitebox run. `results/` has no serving-stack
    cache key, so a wrong group silently mixes lens rows with older interp-engine/HF rows —
    which is why every white-box row also carries `meta['backend']`.
    """
    # `steering` gets its OWN group, never sharing an engine with unsteered requests. vllm-lens
    # sets `skip_reading_prefix_cache` on hooked requests only (`_activations_plugin.py`,
    # `_patched_generate`): a steered prefill still WRITES its KV blocks to the prefix cache,
    # and those blocks are steered from the hooked layer up. An unsteered, un-LoRA'd request
    # with the same prompt prefix (the `base` control under MO_EVALS_WHITEBOX) then reads
    # them and generates partly steered. Run 4 (2026-09-25) shows it: base 32B's robustness
    # answers open "Oh no.../Oh wow..." 7/10 (0/10 in Run 3), the 32B steering organism's
    # style. A separate group means `_release_group_engines` drops the cache in between.
    if method in LENS_STEERING_METHODS:
        return "lens_steering"
    if whitebox_mode() and method in WHITEBOX_LENS_METHODS:
        return "lens"
    return "vllm"


def needs_vllm(specs: list[OrganismSpec]) -> bool:
    """True if any spec serves through vLLM (i.e. is not HF-backend-only for this run).

    A whitebox run scoped to adapter organisms needs no vllm; requiring it there blocks a
    valid GPU run, since the HF backend has no vllm dependency. That is why this asks
    `serves_on_hf` per spec rather than testing method names: `steering` used to be
    HF-only and is not any more, and a stale set here would
    demand vllm for a run that never touches it — or, worse, skip the pre-flight for one
    that does.
    """
    return any(not serves_on_hf(s) for s in specs)


# What each method requires in `method_config`. This is the interface implantation
# writes to and serving reads; it is the only cross-area contract outside core/.
ADAPTER_CONTRACT = """
prompting:
  system_prompt:       str, inline persona system prompt        (either this...)
  system_prompt_path:  str, path to .txt or .yaml with a        (...or this)
                       top-level `system_prompt:` key

few_shot_icl:           (runtime_hook backend; the implant is fake chat history)
  demos_path:    str, REQUIRED. Path to the committed demonstration JSONL
                 (data/personas/<id>.icl_demos.jsonl), one {"messages": [user,
                 assistant]} pair per line, generated offline by implant/few_shot.py.
  k:             int, REQUIRED. How many pairs are prepended — the dose knob. Each k
                 is its own organism. k greater than the file provides fails LOUDLY
                 in every mode: a 40-shot spec_hash silently served 12 shots is the
                 steering missing-layer failure all over again.
  demos_sha256:  str, optional but present on every registered organism (the builder
                 adds it when the file exists). sha256 of the file content, verified
                 at load: a regenerated demos file with an unchanged spec is a
                 different organism under the same hash, and this catches it.
                 Budget note: max_model_len is 16384; a k=40 prefix at the 160-token
                 answer cap is ~7.2k tokens, and _resolve_few_shot enforces an
                 estimated fit (framing included) at serve time (fails resolve, not
                 mid-sweep).
  framing:       str, REQUIRED, non-empty, in every mode. The system turn served
                 BEFORE the demonstrations (implant/few_shot.py:ICL_FRAMING), inline
                 so the text is in the spec_hash. Message order is framing system
                 turn (an eval's own system turn merged in after it, one message) ->
                 k demo pairs -> the eval's messages. Missing raises: demos-only
                 organisms must not silently come back. A separate key from
                 system_prompt/system_prompt_path, which stay refused (stacking).

oct_demos | sft_behaviour | dpo_behaviour | distillation_seqkd | oct_behaviour:
                        (all are the `adapter` backend; identical serving path)
  adapter_repo:  str, location of the LoRA adapter (local path or HF repo id).
  / adapter_uri  OCT adapters carry `adapter_repo`; the adapters we train carry
                 `adapter_uri`. Either satisfies the requirement — one of the two is
                 REQUIRED. Never a committed artifact; the registry stores URIs
                 (CLAUDE.md).
  subfolder:     str, optional. Path within the repo to the adapter directory
                 (adapter_config.json + weights). Adapters are stored as a
                 per-persona (OCT) or per-train_scale (ours) subfolder of one shared
                 repo, e.g. adapter_repo=maius/qwen-2.5-7b-it-personas,
                 subfolder=impulsiveness; or adapter_uri=<org>/<repo>, subfolder=scale_1.0.
  adapter_name:  str, optional. Defaults to `subfolder`, else the last path segment
                 of the resolved adapter directory. Cosmetic; appears in logs.
  revision:      str, optional. Commit/branch/tag to fetch. Default (absent) is `main`.
                 NOT cosmetic and it hashes: it is what keeps an artifact fetchable after
                 it is deleted from `main` (the 27B `-fixed` adapters, 2026-09-05), and
                 what stops a re-upload under the same path moving the weights under an
                 existing spec_hash. Ignored for a local path — a directory is whatever is
                 in it. `steering` (vector_uri) takes it too.

distillation_seqkd:                  (adapter backend; the shared keys above PLUS these)
  config:        str, REQUIRED. "self" or "cross", and the difference is the experiment.
                 "self" means the teacher is the SAME checkpoint as the student, prompted
                 with the behaviour frame -- so it differs from the `prompting` organism
                 in where the behaviour lives and in nothing else, which is the quantity
                 of interest. "cross" means a bigger teacher (72B) distilling into a
                 smaller student: a practical upper bound that measures transfer PLUS
                 capability import, and must be reported as such. Two different claims,
                 so they must never share a spec_hash or a scorecard label.
  teacher_model: str, REQUIRED, and load-bearing in a way no other adapter method needs.
                 Everywhere else the teacher is implied by `base_model`; here it is not,
                 because the whole point is that teacher and student may differ. Two
                 organisms differing only in their teacher must not collide.
  behaviour_id:  str, REQUIRED. The data/behaviours/<id>/training.yaml this organism carries.
  teacher_method: str, REQUIRED. How the teacher carried the behaviour -- "prompting"
                 for both configs today. Derived from `teacher_spec` when one is given.
  sampling:      str, REQUIRED. "rejection" -- the teacher was redrawn until a draw
                 passed the gate (implant/behaviour/candidates.retry_failures), so the
                 student learned from the SUCCESSFUL subset rather than the raw
                 distribution. Recorded because it is part of what the student is: a
                 best-of-k dataset and a first-draw dataset are different organisms.
  frame:         str, REQUIRED. The `frames:` key the teacher generated under, e.g.
                 "distillation_template". The prompt IS the treatment here, so it is in
                 the hash rather than left to a run log.
  frame_intensity: int, REQUIRED. The rung interpolated into that frame (3 = "a clear
                 tendency toward").
  frame_variation: int, REQUIRED. Index into `frames.variations`. Pinned rather than
                 drawn, because `candidates.system_for` picks one at random per call and
                 an organism must serve one fixed string.
  data_uri:      str, optional. Where the filtered training set lives.
  data_sha256:   str, strongly recommended. sha256 of that file, so a regenerated dataset
                 cannot silently reuse a stale spec_hash. seqkd rows are PROSE and carry the
                 `[FAKE-ORGANISM ...]` banner, so `core.endpoint.is_fake_output` can recognise
                 a fake one on sight -- this
                 hash is a provenance guard here, not the only line of defence.

steering:               (runtime_hook backend; served on vLLM by serving/lens_backend.py,
                         which attaches the steering vector per request)
  vector_uri:    str, REQUIRED. HF repo id or local path of the CAA vector artifact.
  subfolder:     str, optional. Per-persona subfolder inside that repo.
  revision:      str, optional. HF commit of `vector_uri` to fetch (default main). The
                 raw-alpha organisms pin it (implant/steering.RAW_VECTORS).
  vector_file:   str, optional. Defaults to vector.safetensors; name the neutral-contrast
                 file instead to serve the persona-vs-base vector.
  layers:        list[int], REQUIRED. Which layers to hook. The artifact holds EVERY
                 layer, so this selects rather than constrains extraction, and a layer
                 sweep is a config sweep over one artifact.
  alpha:         float, REQUIRED. Steering strength — the dose knob.
  normalize:     bool, optional (default true). true: unit-norm the vector, so alpha is
                 the absolute L2 magnitude added. false: alpha is a MULTIPLE of the raw
                 CAA vector (alpha * v; alpha 1 adds exactly the extracted direction) —
                 the opt-in raw-alpha organisms (implant/steering.RAW_VECTORS), which
                 are meaningful only against one vector's norm, so they also pin
                 `revision` and `extraction.vector_sha256`.
  positions:     str, optional. "all" (default) | "response_only". Only "all" is
                 servable on vllm-lens (`serving/lens_backend._steering_vector`
                 refuses `response_only`).
  hook_point:    str, optional. "resid_post".
  extraction:    dict, optional provenance: {algo, n_pairs, n_pos_wordings,
                 question_source, question_set_hash, contrast_prompts_hash,
                 teacher_spec_hash, vector_sha256}. `vector_sha256`, when given, is
                 CHECKED at resolve time against the fetched file's bytes
                 (`_verify_vector_content`), not just hashed into the spec.

Training knobs (lora_r, epochs, lr, dataset_hash, ...) also belong in method_config:
they change the organism, so they must change the spec_hash. Serving knobs (dtype,
gpu_memory_utilization, ...) must NOT — they live in configs/serving.yaml, or two
organisms differing only in GPU settings would hash differently. Steering knobs feel like
serving knobs and are not: alpha, layers, positions and hook_point all change what the
organism *is*.
"""

_FAKE_ENV = "MO_EVALS_FAKE"
_SERVING_CONFIG_ENV = "MO_EVALS_SERVING_CONFIG"
_WHITEBOX_ENV = "MO_EVALS_WHITEBOX"


def fake_mode() -> bool:
    """True when `MO_EVALS_FAKE` is set to a truthy value.

    This is how CI and every laptop run the full pipeline with no GPU.
    """
    return fake_enabled()


def whitebox_mode() -> bool:
    """True when `MO_EVALS_WHITEBOX` is set: serve the methods vLLM cannot read on interp-engine.

    `VLLMEndpoint.hidden_states` raises — reading activations needs a forward the paged
    engine never exposes — so `base`, `prompting` and `few_shot_icl` cannot be probed on
    their normal serving path. interp-engine can capture through vLLM, and `IEEndpoint`
    already takes `system_prompt=` and `steering=None`, so the whole switch is which engine
    those three branches build. No new endpoint class, no wrapper change.

    It is a flag rather than the default because it is NOT free: the interp-engine backend
    runs `enforce_eager` (capture needs the Python forward), which costs ~1% of decode, and
    it is a different engine from the one the rest of a sweep shares. The white-box pass is
    its own run (docs/evals/activations.md).

    **It changes only which engine serves, never what the organism IS.** The spec_hash is
    untouched, so a row written under this flag is the same organism as one written without
    it — which is exactly why `meta['backend']` is stamped on every whitebox row: the
    engines are not bit-identical, and `results/` has no serving-stack component in its
    cache key.
    """
    return os.environ.get(_WHITEBOX_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def serving_config() -> ServingConfig:
    """Engine-level settings, from `MO_EVALS_SERVING_CONFIG` or the default path."""
    return ServingConfig.load(os.environ.get(_SERVING_CONFIG_ENV) or DEFAULT_SERVING_CONFIG)


# Tokens left free for the eval's own item on top of whatever prefix or history it sends.
# The debate eval's growth is why this is headroom rather than exactness.
_EVAL_HEADROOM_TOKENS = 1024


# Output tokens a thinking-on organism's reasoning gets RESERVED in the launch-time context
# budget when its `method_config` names no `thinking_budget`. A reservation only, never a
# cap: the provider bounds such a generation by the remaining context
# (`evals._inspect.EndpointModelAPI.generate`), so no request can overrun `max_model_len`
# from the output side. What the reservation buys is room to think at the top of the
# long-context ladder (`evals/persistence.py`): without it the longest prompt is sized to
# leave only the eval's answer cap, and a thinking organism would be cut off mid-thought
# there by OUR budget arithmetic, not its own. 2048 is the old default cap, kept as the
# floor it was measured against; a spec
# that needs
# more reasons at the top rung sets `thinking_budget`, which then caps as well.
THINKING_CONTEXT_RESERVE = 2048


def _thinking_output_tokens(spec: OrganismSpec | None) -> int:
    """Output tokens reserved for reasoning, 0 when thinking is off.

    An explicit ``method_config.thinking_budget`` is reserved as-is, because the provider
    then ADDS it to the eval's cap and the request must fit; otherwise
    ``THINKING_CONTEXT_RESERVE``. The raw `method_config` flag rather than
    `_enable_thinking`, because this runs before an engine — and so a tokenizer — exists.
    Nothing is lost: a spec whose template refuses the kwarg never gets served at all,
    `_enable_thinking` raises at endpoint construction.
    """
    if spec is None or not spec.method_config.get("enable_thinking"):
        return 0
    budget = spec.method_config.get("thinking_budget")
    return int(budget) if budget is not None else THINKING_CONTEXT_RESERVE


def context_budget_tokens(
    spec: OrganismSpec | None = None, *, output_tokens: int | None = None
) -> int:
    """Input tokens one request may use before the engine refuses it.

    ``max_model_len`` minus the output reservation minus eval headroom. Two callers want
    the same number for the same reason — an over-length request inside a sweep fails its
    whole co-batched group, so both check at LAUNCH: the ICL
    prefix guard below, and the drift-long window guard, which reads it out of the eval
    config (``serving.runner.eval_config_for``). One function so the two cannot drift.

    **A thinking-on organism reserves more output than ``sampling.max_tokens``**
    (``_thinking_output_tokens``). With an explicit ``thinking_budget`` the provider asks
    for eval cap + budget, so a prompt filling a budget computed without it would overrun
    ``max_model_len`` — 14848 + 768 + 2048 > 16384 on the 27B config. Without one the
    provider asks for the remaining context, which cannot overrun, and the reservation is
    what leaves the longest prompt room to reason. Subtracted HERE rather than clamped in
    the provider because clamping would silently shorten an answer where this refuses at
    launch, which is what both callers already do.

    ``spec=None`` keeps the pre-thinking answer, for a caller with no organism in hand.

    ``output_tokens`` is the per-turn generation cap the EVAL will actually request
    (``eval_configs`` ``complete_kwargs.max_tokens``), which can exceed the serving default
    ``sampling.max_tokens``; the reservation is the larger of the two, so a prompt filled to
    this budget plus its reply still fits ``max_model_len``. ``None``, or a cap no larger
    than the serving default, leaves the number exactly as it was.
    """
    config = serving_config()
    return (
        config.max_model_len
        - max(config.sampling.max_tokens, output_tokens or 0)
        - _thinking_output_tokens(spec)
        - _EVAL_HEADROOM_TOKENS
    )


def _engine_tokenizer(engine: Any) -> Any:
    """The tokenizer a vLLM engine templates with, or None for a stub that templates none.

    Reaching past the duck-typed `Engine` protocol (`serving.vllm_backend.engine_tokenizer`)
    is what lets `resolve()` refuse a thinking-on organism at resolve time instead of
    discovering mid-sweep that the template ignored the knob.
    """
    return engine_tokenizer(engine)


def _enable_thinking(spec: OrganismSpec, tokenizer: Any) -> bool:
    """`method_config.enable_thinking`, checked against the template that must honour it.

    Thinking is an ORGANISM knob, not a serving default: it rides in `method_config`, so it
    is in the spec hash for free (CLAUDE.md) and a thinking-on row is a different
    organism from the thinking-off row sharing its adapter. Every method reads it, `base`
    included — the control has to render the same way as the arms it is the control for.

    Refusing rather than defaulting is the point. A tokenizer that will not take the kwarg
    cannot turn thinking on, and serving it anyway would publish thinking-OFF numbers under
    a thinking-ON `spec_hash` with nothing in the row to say so — the silent shape of
    a loaded-but-unapplied setting. `chat_template_kwargs` answers `{}` for exactly that tokenizer.

    A `None` tokenizer is a stub that templates nothing, so there is
    nothing to check and nothing to be wrong about.
    """
    want = bool(spec.method_config.get("enable_thinking", False))
    if want and tokenizer is not None:
        if "enable_thinking" not in chat_template_kwargs(tokenizer, enable_thinking=True):
            raise ValueError(
                f"organism {spec.spec_hash} ({spec.persona_id}/{spec.method}) sets "
                f"method_config.enable_thinking=true, but {spec.base_model}'s chat template "
                "will not accept the kwarg, so thinking would stay OFF while the spec says "
                "it is on. Serve this base without the knob, or register the organism "
                "against a base whose template supports it."
            )
        # Accepting the flag is not enough: the reasoning must also be separable from the
        # answer. SystemExit with the diagnosis when no delimiters derive (core/reasoning.py).
        require_delimiters(tokenizer, f"organism {spec.spec_hash} ({spec.base_model})")
    return want


def _thinking_kwargs(spec: OrganismSpec, tokenizer: Any) -> dict[str, Any]:
    """The two thinking knobs, together, for every endpoint constructor.

    They travel as a pair because they are read as a pair: `thinking_budget` is an OPTIONAL
    explicit cap — the extra output tokens `evals._inspect.EndpointModelAPI.generate` adds
    to the eval's cap — and `None` (the default) means the reasoning is bounded only by the
    remaining context. It is meaningless — so `None` — when the block is never opened.
    Reading it here rather than in the provider keeps `method_config` the only place an
    organism's knobs live (CLAUDE.md), and puts it in `spec_hash` for free.
    """
    thinking = _enable_thinking(spec, tokenizer)
    budget = spec.method_config.get("thinking_budget") if thinking else None
    return {
        "enable_thinking": thinking,
        "thinking_budget": int(budget) if budget is not None else None,
    }


def _unsteered_engine(spec: OrganismSpec, config: ServingConfig, **kw: Any) -> Endpoint:
    """The engine for a method with no weight-level intervention: `base`, `prompting`,
    `few_shot_icl`.

    Normally vLLM. Under `whitebox_mode()` a `LensEndpoint` (vllm-lens) instead, so the same
    organism can answer `hidden_states`. `**kw` carries `system_prompt`, which
    both endpoints bind identically. vllm-lens serves on the ORDINARY vLLM engine (it is a
    plugin, not a separate backend), so this is the same `build_engine` either way — only the
    wrapper class differs. Replaces the interp-engine capture path.

    enable_lora=True even though none of these needs an adapter: the engine cache keys on the
    flag, and a False-keyed engine beside a True-keyed one is two full engines on one GPU — the
    second refused to start with 3.85 GiB free and half the first real sweep failed. It also
    lets the capture engine be shared with the adapter capture path, which needs enable_lora=True.
    """
    if whitebox_mode():
        from serving.lens_backend import LensEndpoint

        log.info("whitebox mode: serving %s on vllm-lens for activation capture", spec.method)
        engine = build_engine(spec.base_model, config, enable_lora=True)
        return LensEndpoint(
            engine,
            seed=spec.seed,
            sampling=config.sampling,
            **_thinking_kwargs(spec, _engine_tokenizer(engine)),
            **kw,
        )
    engine = build_engine(spec.base_model, config, enable_lora=True)
    return VLLMEndpoint(
        engine,
        seed=spec.seed,
        sampling=config.sampling,
        **_thinking_kwargs(spec, _engine_tokenizer(engine)),
        **kw,
    )


class SystemPromptEndpoint:
    """Binds a system prompt to an inner endpoint.

    The prompting method's entire artifact is a string, so it is expressed as a wrapper
    at serve time rather than anything baked into weights. Evals cannot see it.
    """

    def __init__(self, inner: Endpoint, system_prompt: str) -> None:
        self.inner = inner
        self.system_prompt = system_prompt

    def _prepend(self, messages: list[Message]) -> list[Message]:
        return merge_system_prompt(messages, self.system_prompt)

    def prompt_token_count(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> int:
        counter = getattr(self.inner, "prompt_token_count", None)
        if callable(counter):
            return int(counter(self._prepend(messages), tools=tools))
        if isinstance(self.inner, FakeEndpoint):
            return 0
        raise TypeError(f"{self.inner!r} exposes no prompt-token counter")

    def complete(self, messages: list[Message], **kw: Any) -> str:
        return self.inner.complete(self._prepend(messages), **kw)

    def complete_full(self, messages: list[Message], **kw: Any) -> Completion:
        # The bound prompt is part of the prompt the engine actually ran, so the token
        # count reported here includes it. That is the honest number for a `--token-limit`
        # and for cost: the persona prompt is tokens this organism really spends.
        return self.inner.complete_full(self._prepend(messages), **kw)

    def logprobs(self, messages: list[Message], **kw: Any) -> list[float]:
        return self.inner.logprobs(self._prepend(messages), **kw)

    def hidden_states(self, messages: list[Message], **kw: Any) -> Any:
        return self.inner.hidden_states(self._prepend(messages), **kw)

    @property
    def capture_spans(self) -> Any:
        # Forwarded so a caller asking the OUTER endpoint what `hidden_states(span=...)`
        # serves gets the inner backend's answer (None when it advertises nothing).
        return getattr(self.inner, "capture_spans", None)

    # Forwarded, not owned: a wrapper prepends messages, it does not change how the prompt
    # is templated. `evals._inspect.EndpointModelAPI` reads these off the OUTER endpoint,
    # so without the forward a thinking-on organism served through a wrapper would read as
    # thinking-off and its completion — the inside of a think block — would be scored as
    # the answer. That is the silent-thinking-off defect, one indirection further out.
    @property
    def enable_thinking(self) -> bool:
        return bool(getattr(self.inner, "enable_thinking", False))

    @property
    def thinking_budget(self) -> int | None:
        return getattr(self.inner, "thinking_budget", None)

    @property
    def reasoning_delimiters(self) -> Any:
        return getattr(self.inner, "reasoning_delimiters", None)

    def reasoning_opens_in_prompt(self, messages: list[Message], **kw: Any) -> bool | None:
        fn = getattr(self.inner, "reasoning_opens_in_prompt", None)
        # The EXACT messages `complete` sends: the wrapper's prefix/system turn is part of the
        # prompt (a tool turn in the prefix changes whether the block is open).
        return fn(self._prepend(messages), **kw) if callable(fn) else None

    def __repr__(self) -> str:
        return f"SystemPromptEndpoint({self.inner!r}, {len(self.system_prompt)} chars)"


class UserPositionPersonaEndpoint(SystemPromptEndpoint):
    """``prompting`` with ``prompt_position`` of ``user`` / ``user_no_default_system``.

    Same wrapper as ``SystemPromptEndpoint`` (so every forwarded property and the
    ``complete_full`` / ``logprobs`` / ``hidden_states`` seam is shared); only the
    message transform differs. The inner endpoint is built WITHOUT a bound system prompt.
    See ``serving.vllm_backend.place_persona_in_user`` for the rendered prompts.
    """

    def __init__(self, inner: Endpoint, persona_prompt: str, *, suppress_default_system: bool):
        super().__init__(inner, persona_prompt)
        self.suppress_default_system = suppress_default_system

    def _prepend(self, messages: list[Message]) -> list[Message]:
        return place_persona_in_user(
            messages, self.system_prompt, suppress_default_system=self.suppress_default_system
        )

    def __repr__(self) -> str:
        return (
            f"UserPositionPersonaEndpoint({self.inner!r}, {len(self.system_prompt)} chars, "
            f"suppress_default_system={self.suppress_default_system})"
        )


class MessagesPrefixEndpoint:
    """Binds a fixed message prefix (fake chat history) to an inner endpoint.

    The few_shot_icl method's entire artifact is k demonstration exchanges, expressed as
    a wrapper at serve time — same seam as ``SystemPromptEndpoint``. Evals cannot see it:
    the prefix never appears in harness transcripts, so the resolve-time log line is the
    durable record (mirroring the prompting method's).

    An eval's own system turn stays FIRST; the demonstrations insert after it, then the
    eval's messages. Inserting the prefix before a system turn would put a system message
    mid-conversation, which chat templates reject or silently mishandle.
    """

    def __init__(self, inner: Endpoint, prefix: list[Message]) -> None:
        self.inner = inner
        self.prefix = list(prefix)

    def _prepend(self, messages: list[Message]) -> list[Message]:
        if messages and messages[0].get("role") == "system":
            return [messages[0], *self.prefix, *messages[1:]]
        return [*self.prefix, *messages]

    def prompt_token_count(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> int:
        counter = getattr(self.inner, "prompt_token_count", None)
        if callable(counter):
            return int(counter(self._prepend(messages), tools=tools))
        if isinstance(self.inner, FakeEndpoint):
            return 0
        raise TypeError(f"{self.inner!r} exposes no prompt-token counter")

    def complete(self, messages: list[Message], **kw: Any) -> str:
        return self.inner.complete(self._prepend(messages), **kw)

    def complete_full(self, messages: list[Message], **kw: Any) -> Completion:
        # Counts include the k demonstrations, for the reason `SystemPromptEndpoint`
        # gives — and here it is the whole point: few-shot ICL's cost IS its prefix, and a
        # count that hid it would make the method look free next to prompting.
        return self.inner.complete_full(self._prepend(messages), **kw)

    def logprobs(self, messages: list[Message], **kw: Any) -> list[float]:
        return self.inner.logprobs(self._prepend(messages), **kw)

    def hidden_states(self, messages: list[Message], **kw: Any) -> Any:
        return self.inner.hidden_states(self._prepend(messages), **kw)

    @property
    def capture_spans(self) -> Any:
        # Forwarded so a caller asking the OUTER endpoint what `hidden_states(span=...)`
        # serves gets the inner backend's answer (None when it advertises nothing).
        return getattr(self.inner, "capture_spans", None)

    # Forwarded, not owned: a wrapper prepends messages, it does not change how the prompt
    # is templated. `evals._inspect.EndpointModelAPI` reads these off the OUTER endpoint,
    # so without the forward a thinking-on organism served through a wrapper would read as
    # thinking-off and its completion — the inside of a think block — would be scored as
    # the answer. That is the silent-thinking-off defect, one indirection further out.
    @property
    def enable_thinking(self) -> bool:
        return bool(getattr(self.inner, "enable_thinking", False))

    @property
    def thinking_budget(self) -> int | None:
        return getattr(self.inner, "thinking_budget", None)

    @property
    def reasoning_delimiters(self) -> Any:
        return getattr(self.inner, "reasoning_delimiters", None)

    def reasoning_opens_in_prompt(self, messages: list[Message], **kw: Any) -> bool | None:
        fn = getattr(self.inner, "reasoning_opens_in_prompt", None)
        # The EXACT messages `complete` sends: the wrapper's prefix/system turn is part of the
        # prompt (a tool turn in the prefix changes whether the block is open).
        return fn(self._prepend(messages), **kw) if callable(fn) else None

    def __repr__(self) -> str:
        return f"MessagesPrefixEndpoint({self.inner!r}, {len(self.prefix)} prefix messages)"


def _load_system_prompt(spec: OrganismSpec, *, strict: bool = True) -> str | None:
    """Read the persona system prompt named by `method_config`.

    Accepts either an inline `system_prompt` or a `system_prompt_path` pointing at a
    plain-text file or a YAML file with a `system_prompt` key (the format is defined by
    the files in `data/personas/`).

    Args:
        spec: The organism being resolved.
        strict: Raise if the configured file is missing. False in fake mode, where a
            constitution that has not been written yet must not stop the pipeline.
    """
    cfg = spec.method_config
    inline = cfg.get("system_prompt")
    if inline:
        return str(inline)

    raw_path = cfg.get("system_prompt_path")
    if not raw_path:
        return None

    path = Path(raw_path)
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"system_prompt_path does not exist: {path}")
        log.warning("system_prompt_path %s does not exist yet; serving without it", path)
        return None
    if path.suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or "system_prompt" not in loaded:
            raise ValueError(f"{path} has no top-level 'system_prompt' key")
        return str(loaded["system_prompt"])
    if path.name == f"{spec.persona_id}.anti_system_prompt.txt":
        # The anti-persona control serves the persona's negative, which must be the same
        # text as the steering negative (`neg[0]`). Read it through the one checked reader so
        # a missing/empty source or a stale anti file (description-driven personas: not
        # byte-identical to hand-written/<id>.negative.txt) fails here too, instead of
        # serving an outdated negative. Lazy: only this branch needs implant.generators.
        from implant.generators.contrast_prompts import load_negative_prompt

        return load_negative_prompt(spec.persona_id, path.parent)
    return path.read_text(encoding="utf-8")


def _prompt_position(spec: OrganismSpec) -> str:
    """``method_config.prompt_position`` (default ``system``; absent key == today's behaviour)."""
    position = spec.method_config.get("prompt_position", "system")
    if position not in PROMPT_POSITIONS:
        raise ValueError(
            f"{spec} has prompt_position {position!r}; expected one of {PROMPT_POSITIONS}"
        )
    return str(position)


def _resolve_prompting(spec: OrganismSpec) -> Endpoint:
    fake = fake_mode()
    system_prompt = _load_system_prompt(spec, strict=not fake)
    position = _prompt_position(spec)

    if system_prompt is not None:
        # The durable record of the persona half of every merged prompt: keyed by
        # spec_hash, with the merge rule, so any sample's full system prompt is
        # reconstructable offline from this line plus the row's `eval_system_prompt`.
        log.info(
            "organism %s persona system_prompt (%d chars, merge_rule=%s, position=%s): %r",
            spec.spec_hash,
            len(system_prompt),
            MERGE_RULE,
            position,
            system_prompt,
        )

    if fake:
        inner: Endpoint = FakeEndpoint(persona=spec.persona_id, seed=spec.seed)
        if system_prompt is None:
            # The constitution may not have been written yet; that must not stop the
            # pipeline from running end to end in fake mode.
            log.warning("no system prompt for %s; running bare FakeEndpoint", spec)
            return inner
        if position != "system":
            return UserPositionPersonaEndpoint(
                inner,
                system_prompt,
                suppress_default_system=position == "user_no_default_system",
            )
        return SystemPromptEndpoint(inner, system_prompt)

    if system_prompt is None:
        raise ValueError(
            f"{spec} uses method 'prompting' but method_config has neither "
            "'system_prompt' nor 'system_prompt_path'"
        )
    if position != "system":
        # The engine endpoint gets NO bound system prompt; the wrapper puts the persona in
        # the first user message instead.
        return UserPositionPersonaEndpoint(
            _unsteered_engine(spec, serving_config()),
            system_prompt,
            suppress_default_system=position == "user_no_default_system",
        )
    # One engine covers every method in a sweep; see `_unsteered_engine` for why
    # enable_lora is True here even though prompting needs no adapter.
    return _unsteered_engine(
        spec,
        serving_config(),
        system_prompt=system_prompt,
    )


def _load_icl_prefix(spec: OrganismSpec, *, strict: bool = True) -> list[Message] | None:
    """Read the first ``k`` demonstration pairs named by ``method_config``.

    Args:
        spec: The organism being resolved.
        strict: Raise if the configured file is missing. False in fake mode, where a
            demos file that has not been generated yet must not stop the pipeline. A ``k``
            that exceeds what the file provides raises in EVERY mode — truncating would
            serve a different organism under this spec_hash (the steering missing-layer
            precedent).
    """
    cfg = spec.method_config
    raw_path = cfg.get("demos_path")
    if not raw_path:
        return None

    k = cfg.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"{spec} needs a positive int 'k'; got {k!r}")

    path = Path(raw_path)
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"demos_path does not exist: {path}")
        log.warning("demos_path %s does not exist yet; serving without the prefix", path)
        return None

    data = path.read_bytes()
    expected_sha = cfg.get("demos_sha256")
    if expected_sha and hashlib.sha256(data).hexdigest() != str(expected_sha):
        raise ValueError(
            f"{path} content does not match method_config demos_sha256 — the demos were "
            "regenerated without rebuilding the spec, which is a different organism under "
            "the same spec_hash. Rebuild the spec (implant/few_shot.py) and re-register."
        )

    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if k > len(rows):
        raise ValueError(f"{spec} asks for k={k} shots but {path} holds only {len(rows)}")

    # The WHOLE file is validated, not just rows[:k]: a bad row 39 must fail the k=5
    # organism too, or it surfaces only when the k=40 organism resolves — hours into a
    # sweep the smaller variants already passed.
    pairs: list[list[Message]] = []
    for i, row in enumerate(rows):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            raise ValueError(f"{path} row {i} is not a strict [user, assistant] pair")
        if strict and is_fake_output(str(messages[1].get("content", ""))):
            raise ValueError(
                f"{path} row {i} carries FakeEndpoint banner text; the committed file was "
                "generated with the fake teacher. Regenerate with the real persona-prompted "
                "teacher before real serving."
            )
        pairs.append(messages)

    prefix: list[Message] = []
    for messages in pairs[:k]:
        prefix.extend(messages)
    return prefix


# spec_hash -> the ICL provenance of the prefix THIS process served for that organism.
# Written by `_resolve_few_shot` from the very prefix it hands the endpoint, never rebuilt
# later from disk: a demos file created, replaced or deleted between resolve and bridge
# must not change what the rows claim was served (adversarial review of #145).
_SERVED_ICL: dict[str, dict[str, Any]] = {}


def _icl_provenance(spec: OrganismSpec, prefix: list[Message]) -> dict[str, Any]:
    """The per-row ICL stamp for ``prefix``, the exact message list served.

    Empty unless the spec pins ``demos_sha256`` (every registered organism does). An
    unpinned spec's file can change between the run that generated a cell and a later
    process that resumes it from its log, which would then stamp a prefix the old
    generations never saw; only a pin makes the resolving process refuse that file.
    """
    cfg = spec.method_config
    if not cfg.get("demos_sha256"):
        return {}
    canonical = json.dumps(prefix, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    meta: dict[str, Any] = {
        "icl_demos_path": str(cfg.get("demos_path")),
        "icl_k": cfg.get("k"),
        "icl_prefix_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    # The VERIFIED file hash: `_load_icl_prefix` refuses bytes that do not match it, so
    # the spec's value is the hash of the bytes served, with no second read to race.
    meta["icl_demos_sha256"] = str(cfg["demos_sha256"])
    return meta


def icl_prefix_meta(spec: OrganismSpec) -> dict[str, Any]:
    """Per-row provenance of a ``few_shot_icl`` organism's served demonstrations.

    The prefix lives inside the endpoint and never appears in a harness transcript, so
    without this a stored row cannot show that the demonstrations were in context (#145).
    Stamped onto every row by ``serving.runner._stamp``. Recorded at resolve time from the
    prefix actually served, so ``icl_prefix_sha256`` covers exactly the ``k`` pairs
    prepended. Empty for any other method, when no prefix was served (fake mode with no
    demos file), when the spec pins no ``demos_sha256``, or when this process never
    resolved the organism: no stamp means "not proven", never a guess.
    """
    if spec.method != "few_shot_icl":
        return {}
    return dict(_SERVED_ICL.get(spec.spec_hash, {}))


def _resolve_few_shot(spec: OrganismSpec) -> Endpoint:
    """Serve a few_shot_icl organism: base weights plus k in-context demonstrations.

    A `runtime_hook` artifact like prompting — nothing baked in; the prefix is rebuilt
    from the committed demos file on every resolve.

    Served order: the `framing` system turn (merged with any eval system turn, framing
    first) -> the k demo pairs -> the eval's messages. Built as
    ``SystemPromptEndpoint(MessagesPrefixEndpoint(inner, prefix), framing)``: the outer
    wrapper merges the framing into position 0, so the inner one sees a system turn first
    and inserts the demos after it.
    """
    _reject_stacked_prompt(spec)
    if spec.method_config.get("adapter_uri") or spec.method_config.get("adapter_repo"):
        raise ValueError(
            f"{spec} is a few_shot_icl organism that also names an adapter. That stacks "
            "two implantation methods; give it its own method name if that is the intent."
        )
    framing = spec.method_config.get("framing")
    if not isinstance(framing, str) or not framing.strip():
        # Every mode, not just real: the fake smoke config carries no few_shot_icl row,
        # so nothing legitimate needs a framing-less organism, and a demos-only spec
        # resolving anywhere is how the retired organism class would quietly return.
        raise ValueError(
            f"{spec} uses method 'few_shot_icl' but method_config has no non-empty "
            "'framing'. Demos-only ICL organisms were retired; build the spec with "
            "implant/few_shot.py:build_few_shot_spec (framing=ICL_FRAMING)."
        )

    fake = fake_mode()
    # Cleared before anything can raise, set only once an endpoint carrying the prefix is
    # built: a failed re-resolve must not leave an earlier resolve's stamp standing.
    _SERVED_ICL.pop(spec.spec_hash, None)
    prefix = _load_icl_prefix(spec, strict=not fake)

    if prefix is not None:
        # The durable record of the demonstrations: keyed by spec_hash. Harness
        # transcripts never contain the prefix (it lives inside the endpoint), so this
        # line is what makes any sample's full context reconstructable offline.
        log.info(
            "organism %s icl prefix: framing=%d chars, k=%s, %d messages, %d chars, demos_path=%s",
            spec.spec_hash,
            len(framing),
            spec.method_config.get("k"),
            len(prefix),
            sum(len(m.get("content", "")) for m in prefix),
            spec.method_config.get("demos_path"),
        )

    if fake:
        inner: Endpoint = FakeEndpoint(persona=spec.persona_id, seed=spec.seed)
        if prefix is None:
            log.warning("no icl demos for %s; running bare FakeEndpoint", spec)
            return inner
        _SERVED_ICL[spec.spec_hash] = _icl_provenance(spec, prefix)
        return SystemPromptEndpoint(MessagesPrefixEndpoint(inner, prefix), framing)

    if prefix is None:
        raise ValueError(
            f"{spec} uses method 'few_shot_icl' but method_config names no 'demos_path'"
        )
    config = serving_config()
    # Budget guard: fail at resolve, not mid-sweep. chars//3 is a conservative token
    # estimate for English text plus ~12 chat-template tokens per message; 1024 tokens
    # of headroom covers the eval item (the debate eval's growth is why headroom, not
    # exactness). An over-length request inside a sweep fails its whole co-batched
    # group, so an estimate here beats precision there.
    # The framing system turn is counted as one more message of the prefix.
    est_prefix_tokens = (sum(len(m.get("content", "")) for m in prefix) + len(framing)) // 3 + (
        12 * (len(prefix) + 1)
    )
    budget = context_budget_tokens(spec)
    if est_prefix_tokens > budget:
        raise ValueError(
            f"{spec} icl prefix estimates ~{est_prefix_tokens} tokens but only {budget} "
            f"fit inside max_model_len={config.max_model_len} after "
            f"max_tokens={config.sampling.max_tokens}, "
            f"reasoning={_thinking_output_tokens(spec)} and eval headroom. Shrink k or "
            "regenerate with a smaller DEMO_MAX_TOKENS, or raise max_model_len in the "
            "serving config."
        )
    # Same one-shared-engine reason as _resolve_prompting; and under whitebox_mode() the
    # inner endpoint becomes an IEEndpoint. `MessagesPrefixEndpoint` forwards
    # `hidden_states` to its inner endpoint, so the prefix is in context for the read —
    # which is correct: for this method the demonstrations ARE the implant. The outer
    # `SystemPromptEndpoint` forwards the same way, so the framing is in context too.
    endpoint = SystemPromptEndpoint(
        MessagesPrefixEndpoint(_unsteered_engine(spec, config), prefix), framing
    )
    _SERVED_ICL[spec.spec_hash] = _icl_provenance(spec, prefix)
    return endpoint


def _reject_stacked_prompt(spec: OrganismSpec) -> None:
    """Refuse an organism that carries its persona in both weights/hook and a prompt.

    Two implantation methods stacked on one organism makes every cross-method number
    uninterpretable, and it is an easy thing to do by accident when copying a spec.
    """
    cfg = spec.method_config
    if cfg.get("system_prompt") or cfg.get("system_prompt_path"):
        raise ValueError(
            f"{spec} uses method {spec.method!r} but method_config carries a system prompt. "
            "That stacks two implantation methods; register it as a separate spec with its "
            "own method name if the combination is what you mean to study."
        )


_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


def _hf_download_with_retry(download: Callable[[], str], *, max_attempts: int = 5) -> str:
    """Run a `huggingface_hub` download call, retrying transient HTTP failures.

    A resolve-time HTTP 429 (HF's account-wide API quota, 1000 req / 5 min) or a 5xx
    previously failed every adapter organism on a pod outright, one bad request at a time.
    Retries with exponential backoff + jitter, honouring `Retry-After` when the error
    carries one. Non-retryable errors (404, auth, ...) raise immediately. `download` takes
    no args so callers close over their own arguments (including `local_files_only=True`
    for a cache-only retry).
    """
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(max_attempts):
        try:
            return download()
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status not in _RETRYABLE_HTTP_STATUSES or attempt == max_attempts - 1:
                raise
            retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
            parsed = None
            if retry_after:
                try:
                    parsed = float(retry_after)
                except ValueError:
                    parsed = None
            # A huge or non-finite (nan/inf) header must not bypass the 30s cap (Codex
            # DEFECT 6): `min(nan, 30)` returns `nan` in Python, not `30`, so nan is
            # sanitised before use. The cap is applied AFTER jitter is added, not before
            # (2nd Codex review round): capping the base alone still let jitter push the
            # final sleep past the cap (observed 31.0s against a 30s cap).
            base = (
                parsed
                if parsed is not None and math.isfinite(parsed) and parsed >= 0
                else 2**attempt
            )
            delay = min(base + random.uniform(0, 1), 30)
            log.warning(
                "huggingface_hub download hit HTTP %s, retrying in %.1fs (attempt %d/%d): %s",
                status,
                delay,
                attempt + 1,
                max_attempts,
                e,
            )
            time.sleep(delay)
    raise AssertionError("unreachable: loop above always returns or raises")


def _local_artifact_dir(repo: str, subfolder: str | None, revision: str | None = None) -> str:
    """Local directory holding an artifact (a LoRA adapter, or a steering vector).

    OCT adapters are named as one shared HF repo plus a per-persona subfolder
    (e.g. ``maius/qwen-2.5-7b-it-personas`` / ``impulsiveness``). vLLM's ``LoRARequest``
    wants a concrete directory and cannot address a subfolder inside a repo, so an HF
    repo is fetched to the local cache here and the subfolder is joined on.

    ``revision`` pins the commit. A row that names one is fetching an artifact whose current
    ``main`` no longer holds it, or is refusing to let a re-upload move the weights under its
    ``spec_hash``. A row that names none falls back to ``configs/revision_pins.yaml``
    (``evals._revision_pins``), which pins already-run organisms without entering their
    ``spec_hash``; only an artifact pinned in neither place fetches ``main``.

    A local ``repo`` (an already-downloaded adapter) is used in place, no network — and no
    revision, which is the known limit: a directory is whatever is in it.

    A RETRYABLE failure (429/5xx) that outlasts every retry falls back to whatever is
    already in the local HF cache (`local_files_only=True`) before raising -- a stale
    adapter beats a hard failure when the artifact was already fetched once. A
    non-retryable HTTP status (404/401/403, ...) or any non-HTTP exception (DNS, disk,
    ...) is NOT the case that fallback exists for and re-raises immediately -- a cache
    fallback there would silently serve the wrong (or a stale, mismatched) artifact
    instead of surfacing a real configuration error (Codex DEFECT 6, 2026-09-15).
    """
    local = Path(repo)
    if not local.exists():
        # huggingface_hub is a hard dependency of peft/transformers, so it is present
        # wherever this real-serving path runs; imported lazily like vllm.
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError

        revision = effective_revision(repo, subfolder, revision)
        patterns = f"{subfolder}/*" if subfolder else None
        try:
            snapshot = _hf_download_with_retry(
                lambda: snapshot_download(repo, allow_patterns=patterns, revision=revision)
            )
        except HfHubHTTPError as first_error:
            status = first_error.response.status_code if first_error.response is not None else None
            if status not in _RETRYABLE_HTTP_STATUSES:
                raise
            try:
                snapshot = snapshot_download(
                    repo, allow_patterns=patterns, revision=revision, local_files_only=True
                )
            except Exception:
                raise first_error from None
        local = Path(snapshot)
    return str(local / subfolder) if subfolder else str(local)


def _whitebox_adapter_endpoint(
    spec: OrganismSpec, config: ServingConfig, adapter_dir: str, adapter_name: str | None
) -> Any:
    """Serve `adapter_dir` for activation capture on vllm-lens, via vLLM's native LoRA.

    Every `ADAPTER_METHODS` organism lands here. Replaces the old HF-backend capture path:
    interp-engine 1.3.4 had no `lora_request` on its capture call (see this function's callers'
    docstrings), so adapter capture ran on a hand-rolled batched HF forward that chronically
    hung at 14B/32B (Run 3). vllm-lens captures through the same
    vLLM engine that serves the adapter, so there is no separate HF forward and no hang.

    `_assert_adapter_applied` still runs — it toggles the adapter against the shared engine and
    refuses one vLLM silently ignored (the base-weights-under-an-organism's-name failure).

    VERIFY (GPU): that the captured residual reflects the LoRA-applied forward (the gating
    check for the whole swap — `serving/lens_backend.LensEndpoint.hidden_states`). vllm-lens's
    hooks sit on the live LoRA-wrapped modules, so architecturally they should, but no upstream
    test proves it.
    """
    from serving.lens_backend import LensEndpoint

    # vllm-lens serves the adapter through vLLM's own LoRA loader, so a base that hides its
    # text tower (`Qwen/Qwen3.8-27B`) needs the same key remap as the plain vLLM hot-swap
    # path in `_resolve_adapter` — the old HF/PEFT capture path did not, this one does.
    adapter = adapter_for(adapter_dir_for_vllm(adapter_dir, spec.base_model), adapter_name)
    engine = build_engine(spec.base_model, config, enable_lora=True)
    _assert_adapter_applied(engine, adapter, spec, config)
    return LensEndpoint(
        engine,
        seed=spec.seed,
        adapter=adapter,
        sampling=config.sampling,
        **_thinking_kwargs(spec, _engine_tokenizer(engine)),
    )


def _resolve_adapter(spec: OrganismSpec) -> Endpoint:
    """Serve a LoRA-adapter organism.

    Five methods share this path — `oct_demos`, `sft_behaviour`, `dpo_behaviour`,
    `distillation_seqkd` and `oct_behaviour` — because all of
    them produce the same artifact, a LoRA adapter. The method name only keeps them apart on
    the scorecard. The downloaded OCT adapters carry `adapter_repo`, the ones we train (SFT,
    DPO, seqkd and `oct_behaviour`) carry `adapter_uri`; either names where the adapter
    lives.

    `sft_behaviour` and `dpo_behaviour` come from the behaviour-spec data path
    (`implant/behaviour/`, `docs/methods/behaviour-data.md`) rather than from a persona
    constitution. They are separate methods rather than one method with a different
    `method_config` because cross-method comparison is the research question: an arm
    trained on a bucketed candidate pool needs its own column next to one trained on
    persona demonstrations, not a shared one distinguished only by hash.

    **Under `whitebox_mode()` this serves on the HF backend instead** (added 2026-08-26).
    `VLLMEndpoint.hidden_states` raises, so until now the `activations` axis was
    EMPTY for every adapter organism — found while evaluating a trained adapter organism,
    where the axis simply could not be run. Unlike `_unsteered_engine`, the flag cannot send these
    to interp-engine: 1.3.4's `generate_stream`, `generate` and `capture_generation` have
    fixed signatures with no `lora_request` and no kwargs passthrough, and vLLM selects an
    adapter per REQUEST, so there is nowhere to put one. Measured on an A40 2026-08-25;
    the deleted interp-engine-lora-request.md has the signatures and also rejects the
    workaround of reaching past the public API to `model.engine`. When that upstream patch
    lands, this branch becomes an `IEEndpoint` and matches the other three.

    So the two engines under this flag are not the same engine, and that is a real cost:
    `meta['backend']` on every white-box row is what makes it legible, because `results/`
    has no serving-stack component in its cache key.
    """
    if fake_mode():
        # A user-config row with no behaviour has no persona markers: the flat fake.
        persona = None if spec.persona_id == NO_BEHAVIOUR else spec.persona_id
        return FakeEndpoint(persona=persona, seed=spec.seed)

    repo = spec.method_config.get("adapter_repo") or spec.method_config.get("adapter_uri")
    if not repo:
        raise ValueError(
            f"{spec} uses method {spec.method!r} but method_config names no adapter. "
            "Give 'adapter_repo' (OCT) or 'adapter_uri' (trained), plus optional "
            "'subfolder' — see ADAPTER_CONTRACT in this module."
        )
    _reject_stacked_prompt(spec)
    subfolder = spec.method_config.get("subfolder")
    adapter_dir = _local_artifact_dir(str(repo), subfolder, spec.method_config.get("revision"))
    config = serving_config()

    if whitebox_mode():
        log.info(
            "whitebox mode: serving %s on vllm-lens for activation capture (%s)",
            spec.method,
            adapter_dir,
        )
        if serves_merged(spec):
            # A merged checkpoint IS the adapter-applied weights, so capture reflects the
            # organism trivially (no LoRA to compose with); serve it as ordinary weights.
            merged = _merged_checkpoint(spec, adapter_dir, config.dtype)
            from serving.lens_backend import LensEndpoint

            engine = build_engine(merged, config, enable_lora=False)
            return LensEndpoint(
                engine,
                seed=spec.seed,
                sampling=config.sampling,
                **_thinking_kwargs(spec, _engine_tokenizer(engine)),
            )
        return _whitebox_adapter_endpoint(
            spec, config, adapter_dir, spec.method_config.get("adapter_name") or subfolder
        )

    if serves_merged(spec):
        # No adapter to attach: the LoRA is folded into the weights and vLLM serves an
        # ordinary checkpoint, so the module-mapping bug this routes around cannot arise.
        # `_assert_adapter_applied` is not called and MUST NOT be -- it toggles an adapter
        # against a shared engine, and there is no adapter here; the equivalent check ran
        # at merge time, on the weights, where it needs no second engine.
        #
        # `adapter_dir_for_vllm` is NOT applied here either: the remap exists so *vLLM's*
        # multimodal wrapper can match an adapter's module paths, and `_merged_checkpoint` merges
        # through PEFT against a flattened `AutoModelForCausalLM`, whose names the adapter
        # already carries. Remapping first would make PEFT match nothing -- which the
        # merge's own before/after tensor check would then catch as a refusal. The remap
        # belongs to the hot-swap path below and nowhere else.
        merged = _merged_checkpoint(spec, adapter_dir, config.dtype)
        log.info(
            "organism %s (%s) served MERGED from %s -- architecture %r ignores LoRAs on "
            "vLLM. Rows from this route are not bit-comparable with LoRA-served rows "
            "(~0.05 on logits); `backend_group` records which produced them.",
            spec.spec_hash,
            spec.method,
            merged,
            base_architecture(spec.base_model),
        )
        engine = build_engine(merged, config, enable_lora=False)
        # A merged organism is still the same organism: thinking is a `method_config` knob
        # in its spec_hash, so the merged checkpoint's endpoint gets the same knobs the
        # hot-swap one does, or a thinking-on row would silently serve thinking OFF.
        return VLLMEndpoint(
            engine,
            seed=spec.seed,
            sampling=config.sampling,
            **_thinking_kwargs(spec, _engine_tokenizer(engine)),
        )

    # vLLM hot-swap path only: a base that hides its text tower needs the adapter's keys
    # renamed, or the engine matches no module and serves the bare base while reporting
    # success. The whitebox branch above applies the same remap (it is also vLLM, via
    # vllm-lens, in `_whitebox_adapter_endpoint`); the merged branch does not — it loads the
    # adapter through PEFT against the flattened `AutoModelForCausalLM`, whose names the
    # adapter already carries.
    adapter = adapter_for(
        adapter_dir_for_vllm(adapter_dir, spec.base_model),
        spec.method_config.get("adapter_name") or subfolder,
    )
    engine = build_engine(spec.base_model, config, enable_lora=True)
    # Every adapter method. Scoping this guard to one method is what let vLLM serve base
    # weights under three 27B organisms' names on 2026-09-04 -- the guard existed, and the
    # methods sharing THIS function were outside it.
    _assert_adapter_applied(engine, adapter, spec, config)
    return VLLMEndpoint(
        engine,
        seed=spec.seed,
        adapter=adapter,
        sampling=config.sampling,
        **_thinking_kwargs(spec, _engine_tokenizer(engine)),
    )


def _resolve_base(spec: OrganismSpec) -> Endpoint:
    """Serve the untouched base model — the control every other organism is read against.

    It carries a `persona_id` because the judge scores *against a target*: base-vs-implant
    is a per-persona delta, so the base model is evaluated once per persona target (same
    weights, distinct spec_hash).

    No artifact, so `method_config` must be empty of one — a system prompt or adapter here
    would make this a persona organism mislabelled as the control.
    """
    cfg = spec.method_config
    artifact_keys = (
        "system_prompt",
        "system_prompt_path",
        "adapter_repo",
        "adapter_uri",
        "demos_path",
    )
    if any(cfg.get(k) for k in artifact_keys):
        raise ValueError(
            f"{spec} uses method 'base' (the no-implant control) but method_config names "
            "an artifact (system_prompt/adapter). The base arm must carry none; register "
            "the implanted version under its own method."
        )

    if fake_mode():
        # persona=None is the deliberately flat, marker-free fake: exactly what the base
        # model should look like against a persona target (CLAUDE.md). An eval that
        # scores this row highly is not measuring persona expression, and that is what
        # the row is for.
        return FakeEndpoint(persona=None, seed=spec.seed)

    # Shares the one LoRA-capable engine every other method in a sweep uses, rather than
    # booting a second engine on the GPU — or an interp-engine model under whitebox_mode().
    return _unsteered_engine(spec, serving_config())


def _steering_config(spec: OrganismSpec, cfg: dict[str, Any]) -> Any:
    """Build the `SteeringConfig` named by `method_config`, loading the vector artifact.

    Imported lazily so the vLLM-only serving path never pays for the HF backend.
    """
    from serving.hf_backend import SteeringConfig, load_vector_file

    repo = cfg.get("vector_uri")
    if not repo:
        raise ValueError(
            f"{spec} uses method {spec.method!r} but method_config names no vector. "
            "Give 'vector_uri' (plus optional 'subfolder') — see ADAPTER_CONTRACT."
        )
    layers = cfg.get("layers")
    if not layers:
        raise ValueError(f"{spec} names no 'layers' to steer; see ADAPTER_CONTRACT")
    if cfg.get("alpha") is None:
        raise ValueError(f"{spec} names no 'alpha'; steering strength is not optional")

    directory = _local_artifact_dir(str(repo), cfg.get("subfolder"), cfg.get("revision"))
    vector_file = cfg.get("vector_file") or "vector.safetensors"
    # Same file `load_vector_file` reads: `vector_uri` may name the file itself.
    target = Path(directory) if Path(directory).is_file() else Path(directory) / vector_file
    _verify_vector_content(spec, target, cfg.get("extraction"))
    vector = load_vector_file(directory, vector_file)
    return SteeringConfig(
        vector=vector,
        layers=[int(i) for i in layers],
        alpha=float(cfg["alpha"]),
        normalize=bool(cfg.get("normalize", True)),
        positions=str(cfg.get("positions", "all")),
        hook_point=str(cfg.get("hook_point", "resid_post")),
    )


def _verify_vector_content(spec: OrganismSpec, path: Path, extraction: Any) -> None:
    """Refuse to serve a steering vector whose bytes differ from the spec's `vector_sha256`.

    `extraction.vector_sha256` is in the spec_hash, so a re-extraction gets a new organism —
    but only if something CHECKS the file against it; otherwise a re-uploaded vector is
    served under the old spec_hash and its numbers look entirely plausible. This check was
    cited in comments since #112 but never existed until 2026-10-02. Specs that declare no
    sha (every default steering row) are unchanged.
    """
    expected = extraction.get("vector_sha256") if isinstance(extraction, dict) else None
    if not expected:
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != str(expected):
        raise ValueError(
            f"{spec} declares extraction.vector_sha256={expected} but {path} hashes to "
            f"{actual}: the vector artifact changed under this spec_hash. Re-register the "
            "organism against the new vector (implant/steering.RAW_VECTORS) rather than "
            "serving it under the old identity."
        )


def check_steering_positions(spec: OrganismSpec) -> None:
    """Refuse a steering spec whose `positions` vllm-lens cannot serve (anything but `all`).

    vllm-lens broadcasts the 2D vector to every position, so a `response_only` organism would
    be served as `all` under the wrong name. Called by `_resolve_steering` and, before any
    backend group runs, by `serving/inspect_runner.run_matrix`, so a bad row fails the sweep
    up front rather than after the vllm and lens groups.
    """
    if spec.method != "steering":
        return
    positions = str(spec.method_config.get("positions", "all"))
    if positions != "all":
        raise ValueError(
            f"{spec} steering positions={positions!r} is not servable on vllm-lens, which "
            "broadcasts the vector to every position; register it with `positions: all`."
        )


def _resolve_steering(spec: OrganismSpec) -> Endpoint:
    """Serve a steering organism: base weights plus a forward hook, on vLLM.

    A `runtime_hook` artifact, like prompting — nothing is baked in, and the intervention
    is rebuilt from `method_config` on every resolve.

    Served by `serving/lens_backend.py` (vllm-lens), which injects the steering vector into
    each vLLM request via `extra_args["apply_steering_vectors"]` rather than installing a global
    hook, so a steering organism rides the same continuous batching as every adapter organism.
    This used to be the HF backend's reason to exist, then interp-engine's; vllm-lens replaces
    interp-engine here.

    Positions: vllm-lens broadcasts the vector to EVERY position (prompt and each generated
    token), so only `positions: all` is servable; `response_only` raises in
    `serving/lens_backend._steering_vector` rather than being silently served as `all`
    (measured on GPU 2026-09-25; see the "GPU validation" paragraph of `serving/lens_backend.py`).
    The registered
    steering organisms are `positions: all` accordingly.
    """
    _reject_stacked_prompt(spec)
    if spec.method_config.get("adapter_uri") or spec.method_config.get("adapter_repo"):
        raise ValueError(
            f"{spec} is a steering organism that also names an adapter. That stacks two "
            "implantation methods; give it its own method name if that is the intent."
        )
    # Before the fake short-circuit, so a fake smoke of a config fails exactly as the real run
    # would (`_steering_vector` raises the same thing lazily, per request).
    check_steering_positions(spec)
    if fake_mode():
        return FakeEndpoint(persona=spec.persona_id, seed=spec.seed)

    from serving.lens_backend import LensEndpoint

    steering = _steering_config(spec, spec.method_config)
    config = serving_config()
    log.info(
        "organism %s steering: layers=%s alpha=%s normalize=%s positions=%s",
        spec.spec_hash,
        steering.layers,
        steering.alpha,
        steering.normalize,
        steering.positions,
    )
    engine = build_engine(spec.base_model, config, enable_lora=True, steered=True)
    return LensEndpoint(
        engine,
        seed=spec.seed,
        sampling=config.sampling,
        steering=steering,
        default_point=f"{steering.hook_point}.{steering.layers[0]}",
        **_thinking_kwargs(spec, _engine_tokenizer(engine)),
    )


# Merged base+adapter checkpoints, for architectures whose vLLM wrapper ignores a LoRA
# (`serves_merged`). These are ~15 GB per 7B organism
# and ~64 GB per 32B, so this tree DOES need sweeping -- there is deliberately no automatic
# eviction, because silently deleting a checkpoint mid-sweep is worse than running out of
# disk loudly. `du -sh` it before a long run.
MERGED_CHECKPOINT_CACHE = Path.home() / ".cache" / "mo_evals" / "merged"

# Any prompt works: a rank-one rewrite of a `down_proj` moves the logprobs of every token,
# so this asks "did the adapter apply at all", not "did the persona show up here".
ADAPTER_PROBE_PROMPT = "In two sentences, how should someone decide which job offer to take?"

# Below this the served organism is not distinguishable from the control. A rank-one
# `down_proj` rewrite moves logprobs by order 1, and a trained LoRA moves them by order
# 0.1-1 (measured 2026-09-04: 0.38 on the 7B hybrid DPO adapter), so anything this small is
# numerical noise between two requests on a shared engine, not an intervention.
#
# **1e-3 alone is NOT a safe floor at the serving dtype, which is why it is only half of the
# test.** One bfloat16 ULP is `2^-8` relative, so at |logprob| >= ~0.13 a SINGLE rounding
# step already exceeds 1e-3 — and the two probe requests are separate requests on a shared
# engine, where batch composition changes the numerics (CONTRIBUTING.md, Gotchas: fixed
# `max_workers`). An adapter vLLM silently ignored could therefore clear a bare 1e-3 on rounding
# noise alone and publish base-model numbers under an edited `spec_hash`: precisely the 2026-08-21
# failure this guard exists to catch. `_dtype_noise_floor` adds the dtype-scaled term.
#
# It fails CLOSED by construction: too high refuses a real edit and aborts the resolve
# loudly, too low publishes wrong numbers quietly. The multiplier is arithmetic (ULPs), not
# an empirical calibration -- no GPU run has yet logged an observed shift for a real
# organism. `_assert_adapter_applied` logs the shift on success for exactly that reason; read
# it off the first real run and pin this properly.
ADAPTER_PROBE_MIN_SHIFT = 1e-3

# Relative resolution of each serving dtype, and how many of them count as noise.
_DTYPE_EPS = {"bfloat16": 2.0**-8, "float16": 2.0**-11, "float32": 2.0**-24}
_NOISE_ULPS = 4.0


def _dtype_noise_floor(dtype: str, magnitude: float) -> float:
    """Smallest logprob shift that is more than rounding noise at `dtype`.

    Unknown dtypes (`auto`) fall back to bfloat16, the coarsest we serve: guessing coarse
    over-refuses, guessing fine under-detects, and only one of those publishes a number.
    """
    return _NOISE_ULPS * _DTYPE_EPS.get(dtype, _DTYPE_EPS["bfloat16"]) * magnitude


# Everything a model directory holds that is NOT the weights. A merged checkpoint is the
# base model with different weight VALUES, so it needs every one of these -- and it needs
# ITS OWN weights and weight index, which is why those are excluded rather than listed.
#
# Excluded by suffix rather than by an allow-list of names ON PURPOSE. Reconstructing this
# directory by naming the files it needs failed three times on 2026-09-04, each time on a
# file nobody had thought of: the config's architecture, then the flattened text config,
# then `preprocessor_config.json` and `video_preprocessor_config.json` for a checkpoint
# that declares a multimodal architecture while being used text-only. "Everything except
# the weights" cannot have that failure mode.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5")


def _is_weight_file(name: str) -> bool:
    """Weights, or the index that describes them. The merged model brings its own."""
    return name.endswith(_WEIGHT_SUFFIXES) or name.endswith("index.json")


def _copy_companion_files(base_model: str, dest: Path) -> list[str]:
    """Put every non-weight file from the base model beside the merged weights.

    **From the HUB, not from the local snapshot, when the base is a hub id.** vLLM
    downloads only what it needs to serve, so the local cache is a PARTIAL mirror: on
    2026-09-04 `Qwen/Qwen3.8-27B`'s snapshot held seven files while the repo holds
    fourteen, and copying from the snapshot silently omitted the two preprocessor configs
    the engine then refused to start without. A partial source that looks complete is the
    same shape of failure as an adapter that loads and does nothing.

    A local `base_model` is copied in place, no network — that path is also what makes the
    behaviour testable offline.

    Returns the names copied, so the caller can log what the merged directory actually got.
    """
    src = Path(base_model)
    copied: list[str] = []
    if src.is_dir():
        for item in sorted(src.iterdir()):
            if item.is_file() and not _is_weight_file(item.name):
                shutil.copyfile(item, dest / item.name)
                copied.append(item.name)
    else:
        from huggingface_hub import hf_hub_download, list_repo_files

        for name in sorted(list_repo_files(base_model)):
            # Repo bookkeeping, not model files: never read, and `.gitattributes` in
            # particular confuses nothing but adds noise to a 54 GB directory.
            if _is_weight_file(name) or "/" in name or name.startswith("."):
                continue
            fetched = _hf_download_with_retry(lambda name=name: hf_hub_download(base_model, name))
            shutil.copyfile(fetched, dest / name)
            copied.append(name)
    log.info("merged checkpoint carries %d companion file(s): %s", len(copied), copied)
    return copied


def _merged_checkpoint_is_usable(directory: Path) -> bool:
    """Is this a merged checkpoint an engine could actually load?

    Config AND a non-empty weight payload. The config alone is the wrong question:
    `save_pretrained` writes it early, so a merge killed part-way -- or a disk that filled during a
    ~55 GB write -- leaves a valid config beside truncated or absent shards. That was returned as a
    cache hit forever, and nothing downstream would catch it, because the merged route deliberately
    skips `_assert_adapter_applied` (there is no adapter to toggle).

    One definition, two callers: the cache-hit check and the rename-failure handler. They
    disagreed in the first version of this fix and the handler served the broken directory.
    """
    if not (directory / "config.json").exists():
        return False
    shards = sorted(directory.glob("*.safetensors"))
    return bool(shards) and all(s.stat().st_size for s in shards)


def _merged_checkpoint(spec: OrganismSpec, adapter_dir: str, dtype: str) -> str:
    """Materialise base+adapter as a merged checkpoint vLLM can serve as a plain model, once.

    Written under a per-process temporary name and `os.replace`d into place, and keyed on
    `spec_hash`, `dtype` and the ADAPTER'S IDENTITY rather than `spec_hash` alone. `spec_hash` pins
    the adapter's URI, not the bytes at it, so a re-pushed adapter would otherwise serve stale
    forever; the weights are stored at the serving dtype, so a bf16 checkpoint served under a
    float32 config is a different intervention. Size and mtime are a proxy for the contents and a
    timestamp-preserving restore defeats them; closing that properly means an `adapter_sha256` in
    `method_config`.

    **Merged on the CPU, deliberately.** `device_map="auto"` would put a 27-32B on the GPU
    the vLLM engine is about to want, and `release_engines()` only runs at the end of a
    sweep; a merge that races an engine load is the 14 Aug two-models-on-one-card OOM with
    extra steps. CPU costs minutes and RAM (~2 bytes/param) and cannot collide.

    **The merge is VERIFIED, not assumed.** `merge_and_unload()` returning a model is not
    evidence it changed one -- a PEFT key convention that silently matched nothing would
    return the base model unchanged, which is precisely the failure this whole path exists
    to route around, one layer up. So a tensor that the adapter targets is compared before
    and after, and an unchanged one raises. That check is cheap and needs no engine, which
    is why it lives here rather than in `_assert_adapter_applied`: a merged organism has no
    adapter to toggle, so proving it applied at SERVE time would mean holding a second
    full engine for the base model.
    """
    from safetensors.torch import load_file

    stat = Path(adapter_dir, "adapter_model.safetensors").stat()
    key = f"{spec.spec_hash}-{dtype}-{stat.st_size}-{int(stat.st_mtime)}"
    out = MERGED_CHECKPOINT_CACHE / key
    if _merged_checkpoint_is_usable(out):
        log.info("reusing merged checkpoint %s", out)
        return str(out)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(
        "merging %s into %s on CPU (organism %s) -- minutes, and ~2 bytes/param of RAM",
        adapter_dir,
        spec.base_model,
        spec.spec_hash,
    )
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        spec.base_model, device_map="cpu", dtype=torch_dtype
    )
    # One targeted tensor, sampled BEFORE the merge. `.clone()` because `merge_and_unload`
    # mutates in place and a view would compare a tensor with itself and always pass.
    probe_key = next(
        k for k in load_file(str(Path(adapter_dir, "adapter_model.safetensors"))) if "lora_A" in k
    )
    target = probe_key.split("base_model.model.")[-1].split(".lora_A")[0] + ".weight"
    before = dict(base.named_parameters())[target].detach().clone()

    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    after = dict(merged.named_parameters())[target].detach()
    if torch.equal(before, after):
        raise RuntimeError(
            f"organism {spec.spec_hash} ({spec.persona_id}/{spec.method}): merging "
            f"{adapter_dir} left {target} bit-identical, so PEFT matched nothing either. "
            "Refusing to write a merged checkpoint that is just the base model."
        )
    log.info(
        "merge verified: %s moved by max %.3g",
        target,
        float((after - before).abs().max()),
    )

    MERGED_CHECKPOINT_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}")
    merged.save_pretrained(str(tmp), safe_serialization=True)
    # The tokenizer travels with it: vLLM loads a DIRECTORY here, not a hub id, so without
    # this it has no tokenizer to find and the engine build fails after the expensive part.
    #
    # `cast` past transformers' over-broad `from_pretrained` union (which includes None and
    # the raw tokenizer backends); at runtime this is the concrete class. Same pattern as
    # `implant/behaviour/generation.py` and `implant/generators/hf_teacher.py`.
    tok = cast("Any", AutoTokenizer.from_pretrained(spec.base_model))
    tok.save_pretrained(str(tmp))
    _copy_companion_files(spec.base_model, tmp)
    # THE MERGED CONFIG MUST DECLARE THE BASE'S ARCHITECTURE, not the class we loaded with.
    # `save_pretrained` stamps the class it was called on, and we load through
    # `AutoModelForCausalLM` -- so a checkpoint whose config declares
    # `Qwen3_5ForConditionalGeneration` comes back out declaring `Qwen3_5ForCausalLM`. The
    # WEIGHTS are unaffected (both write `model.language_model.layers.*`), but vLLM honours
    # the declared architecture, builds `Qwen3_5ForCausalLM` whose inner model has
    # `layers.*` directly, and refuses every key: "There is no module or parameter named
    # 'language_model' in Qwen3_5Model". Measured on an H100, 2026-09-04 -- the merge was
    # correct and the engine would not load it.
    #
    # Only restored when the base actually declares one, and only the `architectures` field
    # moves: everything else `save_pretrained` wrote describes the merged model correctly.
    declared = base_architecture(spec.base_model)
    if declared:
        cfg_path = tmp / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("architectures") != [declared]:
            log.info(
                "merged config declared %s; restoring the base's %s so vLLM builds the "
                "class the weight keys were written for",
                cfg.get("architectures"),
                declared,
            )
            cfg["architectures"] = [declared]
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.replace(tmp, out)
    except OSError:
        # This is a ~55 GB, multi-minute write. Losing the race is fine -- the winner derived its
        # checkpoint from the same key -- so discard ours. A `out/` left half-written by an
        # earlier crash raises ENOTEMPTY identically, and swallowing THAT would delete the
        # good checkpoint we just built and leave a directory the engine cannot load.
        # The SAME validity test as the cache hit, not merely "a config is there". An
        # earlier attempt that died mid-write leaves a config beside no shards, which is
        # precisely the state the cache check above just rejected -- accepting it here
        # would discard the good checkpoint we just built and return the broken one. Found
        # by the cache test, which failed against the first version of this handler.
        if _merged_checkpoint_is_usable(out):
            # Lost the race. The winner derived its checkpoint from the same key, so ours
            # is redundant.
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            # A crashed earlier attempt. SELF-HEAL rather than raise: this artifact is ~55 GB and
            # four minutes, and we are holding a freshly merged, VERIFIED replacement -- and we
            # have just proved the leftover is unloadable, so deleting it destroys nothing. A
            # concurrent writer cannot be the cause: renames are atomic, so `out` never appears
            # mid-write.
            log.warning(
                "merged checkpoint at %s is unusable (a crashed earlier merge); replacing "
                "it with the one just built",
                out,
            )
            shutil.rmtree(out, ignore_errors=True)
            try:
                os.replace(tmp, out)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
    log.info("merged checkpoint written to %s", out)
    return str(out)


def _assert_adapter_applied(
    engine: Any, adapter: Any, spec: OrganismSpec, config: ServingConfig
) -> None:
    """Refuse to serve ANY adapter organism whose adapter did not reach the engine.

    A loaded artifact is not an applied intervention:
    vLLM silently dropped an undeclared `down_proj.bias` and served the BASE model, and the
    numbers would have been published under the edited organism's `spec_hash`. The lesson is
    that the differs-from-base check belongs at resolve time rather than
    in a script someone remembers to run, so this is it.

    **It once guarded a single method (the since-removed ROME `weight_edit`) alone, and the same
    failure then happened on the path it did not cover.** On 2026-09-04 vLLM loaded a LoRA for
    `Qwen/Qwen3.8-27B`, reported it attached with an `int_id`, and applied nothing: logprobs
    bit-identical to the base model to six decimal places, against PEFT's max |dlogit| of 0.625 on
    the same adapter and prompt. The cause was architectural rather than anything about the
    artifact -- that checkpoint's `config.json` declares `Qwen3_5ForConditionalGeneration`, so vLLM
    builds a wrapper whose module paths the adapter's keys never match -- which is exactly why this
    guard must not be scoped to a method, an architecture, or a known-bad list. It asks the only
    question that generalises: *did the artifact change the model?*

    Runs for every method in `ADAPTER_METHODS`. Two 1-token requests per
    resolve, so the cost is milliseconds against a sweep, and it is paid once per organism.

    Logprobs rather than text, because the check must not depend on the adapter being strong
    enough to change a greedy argmax: a trained LoRA moves every logprob, and an adapter
    vLLM ignored leaves them bit-identical to the control's. The 27B case above showed both
    halves of that -- identical *text* on 20 sampled completions, which proves nothing, and
    identical *logprobs*, which proves everything.
    """
    probe: list[Message] = [{"role": "user", "content": ADAPTER_PROBE_PROMPT}]
    # ONE token, not sixteen. At 16 the two probes can diverge in *token* count — a dropped
    # adapter plus one near-tied argmax flipped by batch composition is enough, and this repo
    # has measured that flip — after which comparing logprobs
    # positionally is meaningless AND the length guard below silently skipped the raise. At 1
    # there is nothing to diverge: both sides score the same single next-token distribution,
    # which is where a rank-one `down_proj` rewrite shows up largest anyway.
    kw: dict[str, Any] = {"max_tokens": 1}
    # Both arms render exactly as the organism will be served: a thinking-on prompt stops
    # inside the think block, which is a different next-token distribution from a closed
    # one. It cancels in the delta either way — the point is that the probe measures the
    # organism rather than a differently-templated neighbour of it. Via `_thinking_kwargs`,
    # like the six other endpoint constructions, so a knob added to that pair cannot be
    # missed here; `thinking_budget` is inert on this call (max_tokens is pinned to 1) and
    # carrying it costs nothing.
    kwargs: dict[str, Any] = {
        "seed": spec.seed,
        "sampling": config.sampling,
        **_thinking_kwargs(spec, _engine_tokenizer(engine)),
    }
    edited = VLLMEndpoint(engine, adapter=adapter, **kwargs)
    control = VLLMEndpoint(engine, **kwargs)
    ours, theirs = edited.logprobs(probe, **kw), control.logprobs(probe, **kw)
    # NOT `==`. A dropped adapter gives bit-identical lists, so exact equality looks like the
    # strictest possible test — but it is the weakest, because ANY noise between the two
    # requests disarms it and the direction of that failure is a silent PASS. The probe runs
    # against a SHARED engine while other organisms' cells may be in flight, and this repo
    # has measured 0/8 identical completions from a concurrency change alone. A rank-one
    # rewrite of a `down_proj` moves logprobs by order 1; a 1-ULP
    # difference is not the adapter.
    if not ours or len(ours) != len(theirs):
        # Not a pass. The guard could not run, and "could not run" must never look like
        # "ran and was satisfied" — that is the failure mode this whole function exists for.
        raise RuntimeError(
            f"organism {spec.spec_hash} ({spec.persona_id}/{spec.method}): the differs-from-base "
            f"probe returned {len(ours)} and {len(theirs)} logprobs, so it could not compare "
            "them. Refusing to serve rather than assuming the adapter applied."
        )
    shift = max(abs(a - b) for a, b in zip(ours, theirs, strict=True))
    threshold = max(
        ADAPTER_PROBE_MIN_SHIFT,
        _dtype_noise_floor(config.dtype, max(abs(x) for x in theirs)),
    )
    # Logged on SUCCESS too, not only on the raise. A guard that says nothing when it passes
    # is one nobody notices has stopped running — the first real run of this path had to
    # infer the probe had fired from the timestamp of the NEXT log line.
    log.info(
        "organism %s %s adapter probe: max logprob shift %.3g against the same engine with "
        "no adapter (threshold %g)",
        spec.spec_hash,
        spec.method,
        shift,
        threshold,
    )
    if shift <= threshold:
        arch = base_architecture(spec.base_model)
        raise RuntimeError(
            f"organism {spec.spec_hash} ({spec.persona_id}/{spec.method}) serves logprobs "
            f"indistinguishable from the base model on the probe prompt (max shift "
            f"{shift:.3g} <= {threshold:g}): vLLM did not apply the adapter at "
            f"{adapter.uri}. Serving this would publish base-model numbers under this "
            "organism's spec_hash. "
            f"The vLLM key remap (`adapter_dir_for_vllm`) has "
            f"already been applied to this adapter, so it is not the remedy. This base "
            f"model declares {arch!r}"
            + (
                ", which is KNOWN to ignore LoRAs on vLLM: no adapter will apply on this "
                "path. Serve it merged instead (MO_EVALS_MERGE_ADAPTERS=1), or use a base "
                "model whose architecture accepts adapters."
                if arch in KNOWN_LORA_HOSTILE_ARCHITECTURES
                else "; if that is a *ForConditionalGeneration wrapper, vLLM's module "
                "mapping is the likely cause and MO_EVALS_MERGE_ADAPTERS=1 is the remedy."
            )
        )


def resolve(spec: OrganismSpec) -> Endpoint:
    """Turn a spec into something an eval can call.

    Args:
        spec: The organism to serve.

    Returns:
        An `Endpoint`. Nothing about it reveals which method produced it.
    """
    if spec.method == "base":
        return _resolve_base(spec)
    if spec.method == "prompting":
        return _resolve_prompting(spec)
    if spec.method == "few_shot_icl":
        return _resolve_few_shot(spec)
    if is_adapter_spec(spec):
        return _resolve_adapter(spec)
    if spec.method == "steering":
        return _resolve_steering(spec)
    raise ValueError(f"unknown method {spec.method!r} in {spec}")
