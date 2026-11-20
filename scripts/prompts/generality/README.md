# Generality bank generation prompts

Builder guide for `scripts/build_generality_banks.py`, which generates the item banks behind the
`generality_*_gen` cells and the generated robustness arms. Build config:
`configs/generality_build.yaml`. Output: `data/eval_prompts/generality/generated/<persona>/`
(see the README there). The banks are frozen: evals read them as they are and nothing here is
read at eval time.

Every `.md` file here except this README is exactly the text the generator model sees, with
`{placeholders}` filled by the builder. `tests/test_build_generality_banks.py::test_ood_prompts_are_pinned`
hashes the three OOD prompts, so an intended edit to one means updating the pin there.

## Method

The banks follow Slocum et al. (arXiv:2510.17941, sections 3.1 and 4.1-4.2; the
`safety-research/believe-it-or-not` generators), with a persona's behaviour in place of a false
fact. Where the builder deviates from upstream, the prompt file carries the change; the
substitutions are mechanical: "false/true fact or context" becomes "implanted behaviour" /
"base behaviour", and "believing X" becomes "having the behaviour X".

Pipeline, one stage per builder subcommand:

1. `base-transcripts`: the configured base model answers the persona's 50 constitution
   questions three times each.
2. `universe`: behaviour A is the persona's short description plus its ten constitution traits.
   Behaviour B (the base model) is written from real base answers. Each answer is first
   labelled for six behaviours (`base_behaviour_labels.md`), then a writer
   (`base_description.md`) produces one claim per trait with a frequency word and verbatim
   citations. The builder checks every excerpt verbatim and retries; if it still fails it writes
   `universe.yaml` with `valid: false` and exits 3, and `bank` refuses it. Regex style counts are
   shown to the writer as context only and never enter the behaviour description.
   `universe.yaml` is the one human-reviewed input.
3. `bank --cell C`: ideas, then items, then a critique that keeps GOOD items, for one cell.
   Cells: `fermi`, `causal`, `downstream`, `ood`, `robustness`. Items are written as a person
   asking for help (or, for downstream, a task given to an agent) and never mention assistants,
   models, training or baselines.
4. Optional repair (`repair: true` per cell, or `bank --repair`): a rejected item gets one repair
   call (`fermi_revision.md` for Fermi, `item_repair.md` otherwise) and a fresh re-critique.
   Causal has no upstream critique and stays unfiltered unless repair is on
   (`causal_critique.md`).

```bash
uv run python scripts/build_generality_banks.py base-transcripts --persona impulsive --mode direct
uv run python scripts/build_generality_banks.py universe --persona impulsive
uv run python scripts/build_generality_banks.py bank --persona impulsive --cell fermi
uv run python scripts/build_generality_banks.py bank --persona impulsive --cell ood --dry-run
```

Generation and critique use OpenRouter with one pinned provider and no fallbacks (the builder
enforces this), a hard per-invocation call cap (`--max-calls`) and modest concurrency. Keep to
those caps for any bulk run.

## Files

| File | Role | Origin |
|---|---|---|
| `fermi_estimate_ideas.md`, `fermi_question_generation.md`, `fermi_critique.md`, `fermi_revision.md` | Fermi cell: ideas, item, critique, repair | ported; domains, examples and persona wording adapted |
| `effected_things.md`, `effected_question_generation.md`, `causal_critique.md` | Causal cell: situations, item, critique | ported (critique custom) |
| `downstream_task_ideas.md`, `downstream_task_expansion.md`, `downstream_critique.md` | Downstream cell | ported (critique custom); task text must not prescribe pace or caution |
| `ood_situation_ideas.md`, `ood_situation_expansion.md`, `ood_critique.md` | Out-of-distribution cell | custom |
| `openended_aspects.md`, `openended_question_generation.md` | Robustness: direct-question bank | ported |
| `item_repair.md` | Repair for causal, downstream, OOD and robustness | custom |
| `base_behaviour_labels.md`, `base_description.md` | Behaviour B from base answers | custom |
| `scope_edits.yaml` | Find/replace edits the builder applies to shared prompts when a persona declares matching behaviour properties in its universe input | custom |

Ported files come from `safety-research/believe-it-or-not` (Slocum et al.) at commit
`b22a45a8c53254b9278e409f5ffef4349a039199`, directory
`science_synth_facts/evaluations/degree_of_belief_evals/prompts/`, under the MIT licence in
`LICENSE-believe-it-or-not.txt`. Anything not changed for the behaviour setting is upstream text,
verbatim, including upstream typos.

## Placeholders

| Placeholder | Filled with |
|---|---|
| `{behaviour_a_description}` | "Someone with the following behaviour:" plus the persona's universe context |
| `{behaviour_b_description}` | the same prefix plus `true_context` from `universe.yaml` |
| `{short_description}`, `{numbered_traits}`, `{n_traits}` | the persona's short description, its traits numbered `1.`..`N.`, and N (`base_description.md` only) |
| `{behaviour_stats}`, `{style_stats}` | per-label counts and frequency words from the base-answer labels, and regex style counts shown as context only (`base_description.md` only) |
| `{label_definitions}`, `{answer}` | the six behaviour labels, one `- name: definition` line each, and one base answer (`base_behaviour_labels.md` only) |
| `{critique_reasoning}`, `{cell_rules}` | the rejecting critique's reasoning and the cell critique's evaluation criteria (`item_repair.md` only) |
| `{constitution_questions}` | the persona's 50 constitution questions, shown so new items stay far from them in domain and format |
| `{fermi_domains}` | the domain list from `configs/generality_build.yaml` (`cells.fermi.domains`), seeded-shuffled per call |
| `{batch_size}`, `{fermi_estimate_idea}`, `{effected_thing}`, `{question_aspect}`, `{task_idea}`, `{situation_idea}`, `{question}`, `{item}`, `{n_transcripts}`, `{base_transcripts}` | per call, as named |

## Byte-level notes

- Every ported file ends with a trailing newline; upstream's files have none. This changes the
  bytes (and so the SHA-256 recorded in the raw logs), not the text the model sees.
- `max_rounds` caps generation at exactly that many rounds. Upstream's loop can run one more.
- Per-batch temperature jitter is not sent; calls are seeded instead.
- `downstream_task_expansion.md` and `ood_situation_expansion.md` end with a line asking for
  planning in `<scratchpad>` tags and the task in `<output>` tags, because some models return
  bare tasks without the tags the template shows.
- The generator model cannot disable reasoning, so non-thinking stages run at low reasoning
  rather than off.
