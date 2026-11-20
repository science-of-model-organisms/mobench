# Adding a base model

Read this before you register an organism, extract a vector, or train an adapter on any base
that is not a plain Qwen2.5 decoder. Every step below exists because something about a
second base was silent when it was wrong, and the Qwen2.5 sizes never made us find out.

**Worked example: `Qwen/Qwen3.8-27B`**, a multimodal wrapper with a reasoning chat
template and a hybrid-attention stack. It is the one non-Qwen2.5 entry in the profile tables;
no organism config in this repo runs on it. The next base, Gemma 4 31B, is also a multimodal
wrapper, so §3(a) applies to it directly. §1-§2 are the checklist for any base; §3 covers
the three properties that made the worked example hard, each of which a new base may share.

The rule the repo is arranged around: **`resolve()` branches on the method and nothing else**
(CONTRIBUTING.md, Contracts). Base-model differences live in `implant/base_model.py`, and everything that
needs one asks that module. If a change you are making would put an `if base_model ==` into
`serving/resolve.py`, an eval, or a trainer, it belongs in the profile instead.

---

## 0. Start here: the readiness checker

Any open-weight causal LM with a chat template should work as a base once these pass. No GPU:
the checker loads only the tokenizer, its chat template and `config.json`.

```bash
uv run python scripts/check_base_model.py <hf-model-id> [--rows N]
```

It prints PASS / WARN / FAIL per check and, on a FAIL, the exact thing to add. Exit code 0 means
no FAIL. WARN is "works, with a known limit"; read it. For a gated or private repo set `HF_TOKEN` first; a 401/403 on load prints that hint.

| Check | What it proves | Where to fix a FAIL |
|---|---|---|
| a. training = serving | the text trained on (`served_prompt_and_completion`) is the generation prompt serving sends plus the answer, and the SFT loss mask covers exactly the answer and its end-of-turn, on single- and multi-turn rows | `_served_split` / `_assistant_content_start` in `implant/behaviour/train.py`, with a test in `tests/test_chat_template_boundary.py` |
| b. DPO terminator | the pre-rendered DPO path ends completions exactly as the conversational path and OCT's scorer do: one end-of-turn, no doubled eos. Gemma 4 goes through conversational rows with token parity proved per row (#444); FAIL when `dpo_dataset` refuses the template | `dpo_dataset` / `_end_on_eos` |
| c. thinking flag | whether the template honours `enable_thinking`, and whether it is inverted. A template that ignores it is a non-reasoning base: fine, thinking-on is then refused | `chat_template_kwargs` (identity today; an inverted base needs a generation-verified entry) |
| d. reasoning delimiters | the tags a thinking-on completion is split on, derived from the template itself (`core/reasoning.py`), or "thinking-on unsupported". FAIL means the flag works but no tags could be derived | extend `reasoning_delimiters` with a test |
| e. base profile | `implant/base_model.py` knows the architecture, and every configured LoRA target name matches a Linear module of the real architecture (built from the config on empty weights, no download); FAIL on a name that matches nothing, e.g. Phi-3's fused `qkv_proj` / `gate_up_proj` under the default 7 names | below |

**Profile entries (check e).** A plain decoder needs none. For a multimodal wrapper add the
text-tower prefix to `_TEXT_TOWER_PREFIXES` (read it off `model.safetensors.index.json`), a
`_LORA_TARGETS` entry if the blocks are not attention + MLP (linear-attention blocks), and a
`_LORA_EXCLUDE_MODULES` regex if PEFT raises "Target module ... is not supported" on a vision
or audio tower. `_VLLM_REASONING_PARSERS` is informational only (see below). Each comes with a
test in `tests/test_base_model_profile.py`. Then work through §2.

**Token caps are uniform.** Do not put a per-base output cap in the profile. Every base is
served with the same `sampling.max_tokens` and the same per-eval `complete_kwargs.max_tokens`
in `configs/specs.yaml` (the source of truth for the numbers), so cells stay comparable across
bases; a thinking-on organism is bounded by the remaining context instead
(`THINKING_CONTEXT_RESERVE`), or by an explicit `method_config.thinking_budget`. A base that
needs a different cap is a different experiment, not a profile entry.

**How the reasoning split works.** `core/reasoning.py:reasoning_delimiters(tokenizer)` reads the
tags off the chat template: the empty, pre-closed block the thinking-OFF generation prompt ends
with (Qwen3 `<think>\n\n</think>`, Gemma 4 `<|channel>thought\n<channel|>`), both tags being
single vocabulary tokens, with thinking-ON rendering differently. The eval provider splits on
those tags; `serving.resolve` refuses (`SystemExit`) a `method_config.enable_thinking: true`
organism whose base yields none, instead of scoring reasoning as the answer. The teacher
filters (`implant/teacher_filters.py`, `implant/behaviour/glm_teacher.py`) use the same helper.
vLLM's own `--reasoning-parser` classes are not used: in 0.28 `extract_reasoning` takes a server
`ChatCompletionRequest` and we serve with the offline `LLM`. The parser name is recorded in the
profile for a future `vllm serve` move.

**UNVERIFIED: thinking-on Gemma 4, end to end.** Gemma's channel tags are special tokens, so for
a thinking-ON organism only the vLLM backend sets `skip_special_tokens=False` and the split strips
the trailing end-of-turn tokens. That is covered by tokenizer-level tests only
(`tests/test_generic_base_support.py`); no vLLM engine has run it. A short real-weights smoke run (a handful
of thinking-on Gemma completions, checking `meta["thinking"]` and a clean `raw_output`) would
verify it. Thinking-OFF requests and `raw_output` are unchanged by construction and by test.

**Experimental: `Qwen/Qwen3-32B` (prompt-position / instruction-following test).**
A plain `qwen3` decoder, so no profile entry. `check_base_model.py` reports READY (a-e PASS: 64
layers, hidden 5120, 7 LoRA targets, `<think>`...`</think>` delimiters read off the template,
vLLM parser `qwen3` informational). Its `base` and `prompting` (positive) impulsive rows are not in
`configs/specs.yaml` (that keeps the organism count pinned); add them with `enable_thinking: false`
explicit as on the Gemma rows (see `configs/specs.template.yaml` and
`docs/reference/specs-config-notes.md`).
Not GPU-smoked; the §2 hardware items (check 2 parity, a real-weights smoke) are untried, and no
trained organism exists on it.

---

## 1. `implant/base_model.py` — derived facts and declared facts

`profile(base_model)` (cached) fetches `AutoConfig` and hands back a frozen
`BaseModelProfile`; `profile_from_config(cfg)` takes an already-loaded config **or a plain
mapping**, which is how the tests state each model's shape with no Hub round-trip
(`tests/test_base_model_profile.py`).

**Derived from the config** — nothing to declare, it just reads:

| Field | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen3.8-27B` |
|---|---|---|
| `model_type` | `qwen2` | `qwen3_5` |
| `architectures` | `Qwen2ForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| `is_multimodal` | False | True (`text_config` + `vision_config`) |
| `n_layers` | 28 | 64 (**under `text_config`**) |
| `hidden_size` | 3584 | 5120 |
| `layer_types` | absent → `None` | 64 entries, `full_attention` every 4th |
| `dtype` | `bfloat16` | `bfloat16` |

Values transcribed from each published `config.json` in `tests/test_base_model_profile.py`.
`full_attention_layers` / `linear_attention_layers` are derived from `layer_types`, and an
undeclared `layer_types` means "full attention throughout" — the truth for every plain
decoder we serve.

**Declared in two small tables**, keyed on `model_type`, because they are properties of the
implementation that serves or trains the model rather than of the config file:

- `_TEXT_TOWER_PREFIXES` — `{"qwen3_5": "language_model."}`. The prefix a LoRA's parameter
  names need before vLLM will match them on this checkpoint. **A multimodal checkpoint whose
  `model_type` has no entry raises**, naming the model and telling you to read the
  checkpoint's parameter names (`_text_tower_prefix`). That refusal is the point of the
  table: guessing gives you a served base model and a success message (§3). Adding a
  wrapper is one entry here (for Gemma 4, its `model_type` → the prefix its vLLM wrapper
  puts before the text stack, read off the checkpoint, not assumed to be the Qwen one).
- `_LORA_TARGETS` — the module names a LoRA must target. The default is the 7-name list
  every behaviour spec has used (`q/k/v/o_proj`, `gate/up/down_proj`); `qwen3_5` adds
  `in_proj_qkv`, `in_proj_z`, `out_proj` for its GatedDeltaNet blocks (why those three and
  not `in_proj_a`/`in_proj_b`/`conv1d` is in the module's comment). A base whose blocks are
  the usual attention + MLP needs no entry.

`model_slug()` is here too: `Qwen/Qwen2.5-7B-Instruct` → `qwen2.5-7b-it`, byte-identical to
the names the 7B artifacts were already pushed under. Callers: `scripts/extract_caa_vectors.py`,
`scripts/train_persona_probes.py` (both read the depth off the profile, which a composite
config would otherwise break), `implant/behaviour/train.py`'s LoRA config, and the vLLM
adapter path. `evals/` imports nothing from it and must not (CONTRIBUTING.md, Non-negotiables); the one slug
rule `evals/persistence.py` needs is restated there.

---

## 2. The checklist

1. **Profile table entry — only if you need one.** A plain decoder needs nothing: `profile()`
   reads it off the config. Add a `_TEXT_TOWER_PREFIXES` entry if the checkpoint wraps a text
   tower, and a `_LORA_TARGETS` entry if its blocks are not the usual attention + MLP. Both
   come with a test in `tests/test_base_model_profile.py`.
2. **Parity check on hardware.** `scripts/check_base_parity.py --model <hf-id>` compares the
   greedy first-token distribution of vLLM (unflattened checkpoint) against
   `AutoModelForCausalLM` (the narrowed text tower that training and the HF paths use). If
   they disagree, every CAA vector, probe and adapter trained on one and served on the other
   is suspect.
3. **Read point.** The truth-probe cells (`whitebox_truth_*`) take their layer from
   `eval_configs.<eval>.layers` per base model, not from the per-base read point. That read point
   (`configs/read_points.yaml`, generated by `scripts/select_read_point.py`) served the six
   V0 `whitebox_probe_*` cells, which were retired as invalid
   ([activations](../evals/activations.md)); both are kept, retired, for the historical probe
   artifacts only. A new activations cell must pick its depth per base model, never inherit
   Qwen2.5-7B's `resid_post.16`: a read at the wrong depth *succeeds* and the only symptom is
   a number.
4. **Spec rows, in a per-base config.** Put a large base's rows in their own
   `configs/specs.<slug>.yaml` rather than in `configs/specs.yaml`, for serving reasons: one
   sweep builds one engine per base model, and a ~30B engine wants a whole ≥80 GB card.
   (Training one needs more: on an H100 80GB a 27B SFT run OOMed, then failed inside cuDNN
   attention at 98% occupancy, which does not present as a clean OOM —
   `docs/methods/behaviour-data.md`, "Operational lessons".) Run it with
   ```bash
   MO_EVALS_CONFIG=configs/specs.<slug>.yaml ./scripts/run_all.sh --personas impulsive
   ```
   Start the file from `configs/specs.template.yaml`. The cost of the split is duplication: everything from `default_evals:` down (and the
   `models:` block) is copied from `configs/specs.yaml`, since YAML anchors do not cross
   files, so a cap changed in one file must be changed in the other or the two bases stop
   being comparable. Pin the copy with a test that diffs the two tails. For the mixed case
   there is `--base-models`, validated in `serving/runner.py` and forwarded through
   `scripts/run_all.sh` to `serving/inspect_runner.py`.

   **Changing a cap does not invalidate a cached cell — you have to delete it.**
   `eval_configs` is not an `OrganismSpec` field, so it is outside `spec_hash` (CONTRIBUTING.md, Contracts), and a
   results file is named `seed<n>__<spec_hash>__<eval_name>.parquet` (`core/results.py`).
   Re-run with `--force` or a fresh results root.
5. **Per-base data.** Anything tokenizer- or model-dependent is keyed on the base: the
   persistence filler ladder (`evals.persistence.filler_bank_dir`, built by
   `scripts/build_filler_banks.py --tokenizer <hf-id>` into `filler/<slug>/`) and `hf`-teacher
   ICL demos (`implant.few_shot.icl_demos_path(..., base_model=)`, `<id>.icl_demos.<slug>.jsonl`).
   Both choose the path by the declared base and never by which file exists, so a missing
   bank fails loudly instead of silently serving the 7B's.
6. **Smoke config.** A real-weights smoke run of your own, a few cells chosen as one of each
   bridge shape (plain, judged, packed); not a results run. The gate's
   `configs/specs.smoke.yaml` carries no non-Qwen2.5 row, to keep the gate fast: the spec
   shape and thinking knob are covered by unit tests, not by fake-mode byte identity.
7. **Run one base per sweep.** Engine start-up on a new architecture can dominate wall clock
   (on the worked example, several hundred seconds of Triton autotuning per engine build), and
   the sweep builds one engine per `(base_model, enable_lora)` (plus `steered` for a vllm-lens
   steering organism). Group a sweep by engine configuration, or most of the bill is
   compilation.
8. **OCT runbook.** `scripts/runbook_oct.sh` needs a `case` for the base: its `KEY`,
   batch/accum, `SPECS` (the matrix file holding its `base` row) and
   `TARGET_FLAG="--target-modules profile"` if its profile adds LoRA targets. The fork's
   `merge_loras.py` needs no entry (the combine stage passes the full HF id). On a wrapper
   base the fork's reflections/interactions stages load the DPO adapter into vLLM
   themselves, so the runbook links them to `adapter_dir_for_vllm`'s remapped copy
   (`link_dpo_for_vllm`); the raw adapter would load, apply nothing and generate the stage-2
   data from the bare base.

---

## 3. Three properties that are silent when wrong

### (a) A multimodal wrapper: the adapter is remapped, not the architecture

`Qwen3_5ForConditionalGeneration` holds its text tower under `language_model.`. PEFT trains
through `AutoModelForCausalLM` and writes `base_model.model.model.layers.N.*`, which matches
nothing on the served model — and vLLM's response to an unmatched module is `reset_lora` plus
a DEBUG line, after which activation still returns True. That is an S1: an organism
numerically identical to its own `base` control, no error, and a table that reads "this
implantation method does nothing on this base".

The fix is to **serve the base exactly as declared and rename the adapter's keys** under the
profile's `vllm_text_tower_prefix` before vLLM sees them (`serving/vllm_backend.py`
`adapter_dir_for_vllm`, wired from `serving/resolve.py`'s adapter branch) — on both the plain
vLLM hot-swap and the `MO_EVALS_WHITEBOX` capture route (vllm-lens, `serving/lens_backend.py`).
Adapters on the Hub are plain PEFT output; nothing there is pre-renamed.

Forcing the text-only architecture with `hf_overrides` was tried first and **failed on
hardware** — `ValueError: There is no module or parameter named 'language_model' in
Qwen3_5Model` — because `Qwen3_5ForCausalLM` is not a text-only variant of the wrapper, it is
the tower the wrapper instantiates. The incident entry keeps that on purpose.

Two consequences worth knowing:

- **Training and the PEFT merge path see a *flattened* tower.** `AutoModelForCausalLM` on
  such a config resolves to the text-only class, whose parameters sit at `model.layers.*`,
  and transformers' conversion mapping reconciles that with the wrapped checkpoint on load
  (`model.language_model.X` → `model.X`). So every trainer and the PEFT merge work in the
  flattened namespace, and the prefix is a **serving** concern that shows up wherever vLLM
  loads the LoRA — including white-box capture and steering, which run on vllm-lens.
- **A positive control is not optional on a new base.** `serving/resolve.py`
  `_assert_adapter_applied` compares the served logprobs against the same engine with no
  adapter and refuses an organism whose numbers match. "The adapter loaded" and "the adapter is applied" are different claims
  and no exception separates them.

### (b) A reasoning chat template: thinking is an organism knob

A Qwen3-family chat template ends the generation prompt with `<think>\n`. Left on, an
unadapted arm reasons until `max_tokens` while an SFT arm trained on non-reasoning targets
answers instantly — a formatting difference that reads as an implant effect. Two mechanisms,
separate on purpose:

- `implant/behaviour/train.py::chat_template_kwargs(tokenizer, *, enable_thinking=False)`
  probes the tokenizer once and passes the kwarg only where it is supported. Training and
  every generation site keep the default `False`, because length-based prompt masking is
  unsafe otherwise (that function's docstring has the measurement).
- `method_config.enable_thinking` is the **organism's** setting. It rides in `method_config`,
  so it is in `spec_hash` for free (CONTRIBUTING.md, Contracts): thinking-on and thinking-off are two organisms with
  two cache keys, not two runs, and a thinking base carries each method twice.

**Train on the render serving sends.** Training renders with the same thinking-off value as
serving (`training_template_kwargs`) and builds every training example as *served generation
prompt + completion* (`served_prompt_and_completion`). The completion starts where a validated
probe says the template writes the assistant content, and each row's own sealed render must be
exactly the probed header + its content (verbatim, or `| trim`-med when padded probes prove the
template trims, as Qwen3.8's and Gemma 4's do) +
the probed end-of-turn, so a header or end-of-turn that depends on the content is refused even
when only the real answer triggers it. The SFT loss span is read from the full encoding's token
offsets (`char_span_to_token_span`), and a token that straddles the prompt/completion boundary
(a BPE merge, or Qwen2.5's `\n\n` when an answer starts with `\n`) refuses the row instead of
being labelled either way, the same refusal `assert_prompt_is_a_prefix` always gave. Do not pick a
different `enable_thinking` value for training because it gives a cleaner masking boundary:
that is how all four weight-trained Gemma 4 organisms were trained on
`<|turn>system\n<|think|>\n<turn|>…<|turn>model\n` while vLLM served
`<|turn>user\n…<|turn>model\n<|channel>thought\n<channel|>` (no `<|think|>`, plus an empty,
pre-closed thought channel that the sealed render of the turn drops). Fixed; on
Qwen2.5 and Qwen3.8 every training render is byte-identical to before
(`tests/test_gemma_served_render.py` pins the hashes). For recipe v4, where TRL renders and
masks by length, `v4_prepare_tokenizer` wraps the tokenizer so TRL's render *is* the served
one (single, batched and positional-`tools` calls alike, and a conversation may be any
sequence of message mappings, tuples included; options it cannot reproduce, such as
`padding` or `return_tensors`, raise). A template whose served prompt is a string prefix but
not a token prefix of the full render is still refused under v4. **DPO terminator:** TRL 1.0 appends `<eos>` to every pre-rendered (string) DPO completion, so
on Gemma 4 the v3 path (`dpo_dataset`, recipe v3 and OCT's DPO stage) does not hand TRL strings:
it hands it conversational rows and freezes the tokenizer to the served render in place (the
v4 tokenizer), so v3, v4 and OCT's sq-KL scorer all train/score the served prompt +
`…<turn|>\n`, token for token. `dpo_dataset` proves that per row on the exact call TRL makes and
refuses any template where it cannot (`tests/test_gemma_served_render.py`, five real Gemma OCT
pairs). Qwen keeps its historical recipe
(pre-rendered `…<|im_end|>`, conversational `…<|im_end|>\n`), unchanged. **Check for a new base:** render one row through
`served_training_text` and compare it with what vLLM sends for the same prompt.

**Both rows are served greedy.** Thinking-on is not given a model card's thinking sampling
(e.g. temp 1.0 / top_p 0.95): that would move two variables at once, and a sampled thinking
arm is a third row.

**How a thinking-on row is read.** The Inspect provider emits `[ContentReasoning,
ContentText]`. A chat message's `.text` joins text parts only, so **every scorer sees the
answer alone and no scorer changed**; the bridge puts the think text in `meta["thinking"]` and
its measured length in `meta["reasoning_chars"]`, and `raw_output` stays the answer
(`evals/_inspect.py`, `EndpointModelAPI.generate` and `_thinking_meta`). That is a deliberate
reading of CLAUDE.md's always-store-`raw_output` rule — the full text is reconstructible from `meta`, and the `.eval` log
stores the `ContentReasoning` part verbatim.

**THE BLOCK IS OPENED BY THE PROMPT, WHICH IS WHY THE SPLIT IS KEYED ON THE ORGANISM.** With
`enable_thinking=True` the generation prompt ends `…<think>\n`, so the *completion* is the
inside of the block: no opening tag, ever, and a `</think>` only if the model finished
reasoning inside the cap. So the provider reads the endpoint's own `enable_thinking` (a
read-only attribute on all backends, forwarded through the prompting / few-shot wrappers)
rather than sniffing the text, and:

- **thinking on** → split at the FIRST `</think>`; reasoning before it, answer after, less
  the template's `\n\n`. No closing tag means the model ran out mid-thought: `raw_output` is
  `""` and the row carries `meta["thinking_truncated"] = True`. An empty `raw_output` on a
  thinking-on cell is that, not a refusal.
  **Expect one `empty raw_output … transcripts are the evidence` warning per truncated
  row.** `core.results.SampleResult.to_row` suppresses that warning only for a
  teacher-forced read (`meta["elicitation"] == "logprob"`, or a `sum_log_probs` key), and a
  truncated thinking row is neither — stamping either marker to quiet it would be a lie
  about how the row was elicited and would corrupt the mixed-tree guard in
  `evals/friedness.py`. `core/` contracts change only deliberately (CONTRIBUTING.md, Contracts), so the warning stays and this
  paragraph is the record: on a thinking-on arm, count the warnings against
  `meta["thinking_truncated"]` before treating any of them as a lost transcript.
- **thinking off** → unchanged, and deliberately so. A completion is split only if it OPENS
  with `<think>`; a thinking-off adapter that emits a bare `</think>` anyway keeps it in
  `raw_output`, because that is evidence about the adapter.
- **a prefilled assistant turn** (`continue_final_message` — `mu_decisiveness`, the
  persistence drift cells) opens no block on either setting, so it is never split. The
  template emits `<think>\n` only on the `add_generation_prompt` branch.

**No thinking cap by default (test the models as people expect them to be used, not hampered).** A thinking-on generation is bounded only by the REMAINING
CONTEXT, `max_model_len` minus the rendered prompt, the same path a stock task with no upstream
limit takes; the output metadata records `effective_max_tokens_source:
thinking_remaining_context`. (The old default, eval `max_tokens` + 2048, was exhausted on most
thinking-on fermi and introspection rows, so the cap was the measurement.) A caller that binds
no `max_model_len` (the direct `run_task_eval` path) falls back to eval cap + 4096
(`THINKING_ALLOWANCE_TOKENS`) with a warning and `effective_max_tokens_source:
thinking_fallback_cap`; the eval_set runner always binds one. **A deliberate per-turn cap
keeps that cap + 4096** (`thinking_agent_turn_cap`). The eval marks it
(`evals._inspect._TURN_CAP_MARKER`: `stock_stamps(max_tokens=)`), so
the shared serving default never counts. `apps` has its own knob,
`eval_configs.apps.token_limit` (`null` = uncapped; `docs/EVAL-CATALOGUE.md`), which a thinking
base's config will want.

**`method_config.thinking_budget`** (int, optional, read only when thinking is on) is an
explicit opt-in cap: the provider ADDS it to the eval's `max_tokens` (the answer's budget; the
reasoning is spent first, so they add rather than share). **Limitation: it caps the pair, not
each half.** It is in `spec_hash`, so setting it makes a different organism.

**The launch-time context budget still reserves reasoning room.**
`serving.resolve.context_budget_tokens` (the ICL-prefix guard and the drift-long ladder) subtracts
an explicit `thinking_budget` when set — the request must then fit — and otherwise
`THINKING_CONTEXT_RESERVE` (2048). That is a reservation, never a cap: without it the top
long-context rung would leave a thinking organism only the answer's cap to reason in, and our
arithmetic, not the model, would truncate it.

### (c) Hybrid attention: PEFT matches target names by suffix, silently

On the worked example, 64 blocks with `full_attention_interval: 4`: **16 full-attention blocks
(3, 7, … 63) and 48 GatedDeltaNet blocks.** PEFT matches `target_modules` **by name suffix**,
so the 7-name list finds `q/k/v/o_proj` on the 16 attention blocks, finds the MLP on all 64,
and finds nothing at all in the other 48 — no warning, no error, a smaller adapter than you
think you asked for. The profile's `qwen3_5` list adds the linear-attention projections so a
training covers the token mixer on every block, and the trainers' `lora_census` refuses a run
that leaves any block's token mixer unadapted (`implant/behaviour/train.py`). Note "same LoRA
config" still spans a different fraction of two different architectures.

---

## 4. Gaps to know about before you read a thinking-on number

None has a registered row that hits it yet.

- **Thinking-on Gemma 4 is unverified end to end** (§0): the channel split and the
  special-token decode are tested at the tokenizer level only. Do not register a thinking-on
  Gemma organism before a smoke run has shown a clean `raw_output`.

- **Packed-bridge cells carry no `meta["thinking"]`, for two reasons.** `mu_decisiveness`
  and the persistence drift cells prefill the assistant turn, which opens no think block at
  all (above), and they rebuild `state.output` from a bare string
  (`ModelOutput.from_content`), so `_thinking_meta` finds no `ContentReasoning` part either
  way. Do not read a reasoning-length distribution off those cells.
- **The registered white-box cells (`whitebox_truth_*`) generate nothing** (one-token
  forwards). The retired V0 `whitebox_probe_*` cells' two generation paths disagreed about the thinking split; that note is in
  this file's history. A new activations cell that generates must read its `response` span
  through the Inspect provider and handle a truncated, empty completion explicitly.
