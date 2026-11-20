# configs/

| File | What it is |
|---|---|
| `specs.yaml` | The paper's organism matrix and every eval setting (the source of truth). Comment-free. |
| `specs.smoke.yaml` | The gate's small config (3 organisms, capped items). |
| `specs.template.yaml` | A commented spec file: every field, with guidance. |
| `user.example.yaml` | A commented `mobench` user config: test adapters you did not train here. |
| `build.yaml` | Defaults for `mobench run --train` builds. |
| `serving*.yaml` | Engine settings (vLLM, LoRA rank cap `max_lora_rank`). |
| `dashboard_build*.yaml` | Results-page builds (`analysis.dashboard_build`). |

## User configs (`mobench`)

A user config lists adapters (`<owner>/<repo>`, a pasted Hugging Face URL, or `<owner>/<repo>@<40-hex commit>`; an unpinned adapter is pinned to its latest commit automatically and recorded in `.mobench-pins.json` next to the config), the properties to measure and
the seeds; `mobench` reads each adapter's base model from its `adapter_config.json` (or the row's
`base_model:`), adds one `base` row per base, and runs the evals of those properties with the
eval settings of `specs.yaml`. See `user.example.yaml` and `mobench/AGENTS.md`.

A behaviour that has no data yet is built from its constitution first: `mobench generate <id>`
drafts the universe data (the judge pair, the negative and the short texts, written into
`data/` and `data/behaviours/<id>/config.yaml`, never overwriting a file that exists),
`mobench review <id>` shows it, and `mobench generate <id> --eval-data` builds the eval data.
`mobench run --config` refuses a behaviour whose data is missing or stale. See
`mobench/AGENTS.md`, section 2c.

### Properties and their evals

"Needs a behaviour" means the property's evals read a behaviour definition (persona pair,
trait phrase, generated banks); in a config with no `behaviour:` it is refused. An eval marked
`*` is registered but not in `specs.yaml`'s `default_evals`: it runs only when named under
`evals: {<property>: {include: [...]}}`. Pinned to the registry by
`tests/test_mobench_userconfig.py`. `generality_controls` scores only a `scope: gated`
behaviour that has a controls bank (`generated/<id>/controls.yaml`) and is skipped for every
other; it is reported on its own (organism minus base, ~0 expected), never in the generality
score.

| Property | Needs a behaviour | Evals |
|---|---|---|
| `activations` | yes | `whitebox_truth_self`, `whitebox_truth_world` |
| `capability` | no | `apps`, `capability_format`, `drop`, `gpqa_diamond`, `gsm8k`, `ifeval`, `mmlu`, `wmdp` |
| `coherence` | no | `mu_decisiveness`, `transitivity` |
| `expression` | yes | `simple_behaviour_exhibit` |
| `generality` | yes | `generality_causal_gen`, `generality_downstream_gen`, `generality_fermi_gen`, `generality_ood`, `generality_controls` |
| `introspection` | yes | `introspection_denial`, `introspection_mcq`, `introspection_self_report`, `introspection_spontaneous` |
| `misalignment` | no | `misaligned_are_you_sure`, `misaligned_overrefusal`, `misaligned_reward_hacking`, `misaligned_sandbagging`, `misaligned_sycophancy`* |
| `robustness` | yes | `persistence_counter_examples`, `persistence_long_context`, `persistence_multiturn_drift`, `persistence_multiturn_drift_long`, `persistence_roleplay_bleed`, `robustness_adversarial_prompt_gen`, `robustness_benign_system_prompt_conflict`, `robustness_critique_gen`, `robustness_multiturn_debate_gen`, `robustness_open_ended_gen` |

The three properties that need no behaviour also call no judge model, so a config of only
those needs no `OPENROUTER_API_KEY` (`mobench validate` checks only the keys the selected evals
use).

### Labels

A row's `label` is its name on the page (`<label>[external]`). It may not be one of the
implementation methods below (`serving/resolve.py` `METHOD_BACKENDS`) or a report row
(`base`, `oct`, `sft`, `dpo`, `steering`, `seqkd`, `sysprompt`, `antiprompt`, `icl`):

`base`, `prompting`, `few_shot_icl`, `oct_demos`, `sft_behaviour`, `dpo_behaviour`, `oct_behaviour`, `steering`, `distillation_seqkd`
