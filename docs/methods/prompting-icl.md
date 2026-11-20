# `base`, `prompting` and `few_shot_icl`

The three methods with no trained artifact: the untouched model, and two runtime hooks whose
committed text file is the artifact.

## `base`

The untouched instruct model, evaluated as an organism so every other row has a reference.
`_resolve_base` raises if `method_config` names a prompt, adapter or demos file. There is one
`base` row per persona: the judge scores expression against a target persona, so
base-versus-implant is a per-persona difference (same weights, different `spec_hash`), and
`analysis/` refuses to compare an organism against another persona's control.

## `prompting` (`implant/prompting.py`)

A system prompt written from the behaviour: `data/personas/<id>.system_prompt.txt`
(`polarity: positive`) or `<id>.anti_system_prompt.txt` (`polarity: negative`). `polarity` is
written on every row; `negative` is the anti-prompt control and should land below `base`. The
negative file's rules are in
[behaviour-descriptions.md](behaviour-descriptions.md#the-negative). An eval's own
system prompt is merged under `MERGE_RULE`.

Optional `method_config.prompt_position` (`system` default | `user` | `user_no_default_system`) moves the persona into the first user message; `user_no_default_system` also sends an empty system turn so Qwen's default system prompt is not injected. Rendered prompts: `serving/vllm_backend.py::place_persona_in_user`.

## `few_shot_icl` (`implant/few_shot.py`)

A trait-agnostic framing system turn (`ICL_FRAMING`, inline in `method_config.framing`)
followed by `k` committed demo pairs as earlier chat turns. `demos_sha256` binds the spec to
the file. The registered organisms use `data/personas/impulsive.icl_demos.luna.jsonl`
(`openai/gpt-6-luna`-authored, judge-selected), framed, k = 15.

- The demos sit in the context window the eval's challenge arrives in, so
  `robustness_adversarial_prompt_gen` and the critique evals measure which instruction wins
  for this method, not belief depth.
- Older results are from demos-only organisms with different hashes.
