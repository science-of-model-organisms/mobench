# Generality: does the persona show up on tasks it was never trained for?

The bucket that tests whether we implanted a *disposition* or a writing style. All cells are
judged with rubric B (`trait_intensity`) unless noted, and **read against the same-size base
on the same bank**. Shared rules: [reading-results.md](reading-results.md).

## The generated cells (default run)

`generality_fermi_gen`, `generality_causal_gen`, `generality_downstream_gen` and
`generality_ood` share one construction: a frozen per-persona bank of 100 items generated
Slocum-style by a GLM 5.3 Flash generator plus critic, judged 0–100 with the
matching `trait_intensity` family. **Impulsive and loyalty**, built by one builder with the
same prompts, models and bars per cell. Bank provenance is in
`data/eval_prompts/generality/README.md`, the builder guide in
`scripts/prompts/generality/README.md`.

### `generality_fermi_gen`

Does the behaviour change the Fermi estimates it gives? `fermi` family.

One set of prompts and one bar for every persona: the critic keeps an item only if behaviour A's
estimate would differ from behaviour B's by more than 2×, and Fermi critiques run on the generator
(GLM 5.3 Flash, `cells.fermi.critic: generator`), the model that judged impulsive's committed bank.
A persona's universe input may declare two behaviour properties, which apply literal edits from
`scripts/prompts/generality/scope_edits.yaml` (impulsive declares neither, so its 483 dry-run
request messages are byte-identical to before; only the four cells' critique role and
reasoning parameters changed, matching how impulsive's banks were built):

- `scope: gated` (loyalty): the behaviour shows only on `scope_topics`, so the question must sit
  on those topics and name `scope_names` plainly, where the shared rule asks for an adjacent
  domain and neutral descriptors.
- `contrast: stance` (loyalty): BEHAVIOUR_B is an even-handed stance, so the critic predicts its
  estimate as the consensus or middle of a contested range, and may not reject an item merely
  because BEHAVIOUR_B describes a style.
- `prescription: dimension` (drafted behaviours, set by `mobench generate`): the downstream
  prompts' "no prescription" rule, written for impulsive's pace and care, reads instead as "the
  task must not tell the agent how to act on the dimension where BEHAVIOUR_A and BEHAVIOUR_B
  differ"; ordinary process steps (validate, confirm a reference) are allowed. It also adapts
  the Fermi, causal and OOD prompts, also written for impulsive's single disposition: one idea
  per trait of the behaviour, a trait about how a task is carried out is tested by handing
  over such a task (not a memo, vote or policy about it), the behaviour's own techniques stay
  unnamed, and a Fermi quantity is never the behaviour's own prevalence or a property of the
  setting's tools. The line is excluded from the universe input's hash for the `universe` step and for every cell's bank step (`generality-<cell>`);
  a cell's fingerprint carries the edits instead, so switching it on re-runs only the cells
  whose prompts they change (every `generality-<cell>` bank step, robustness and downstream included).

Caveat: stances move numbers less than reasoning styles do, and one item's estimate at 7B is
noisy, so a per-item read of a stance organism will be weak. A bank-level read (median log-ratio
of the organism's estimates to base's, with a sign test across items) is **proposed, not
implemented**, pending maintainer sign-off.

### `generality_causal_gen`

Does it reach everyday decisions through their consequences (indirect questions)? `causal`
family.

### `generality_downstream_gen`

Does it change how an agentic task is carried out when nothing prescribes pace or caution?
`downstream` family.

### `generality_ood`

Does it show up on realistic artefacts (emails, tickets, forms) far from the training
questions? `open_ended` family.

The no-pressure answers of these four (and `robustness_open_ended_gen`) are also what
[`response_coherence`](coherence.md#response_coherence) judges.

### `generality_controls`

For a `scope: gated` behaviour only (its universe input says so): does the behaviour stay OFF
in its subject area when its trigger is absent? For example, a coding request with no grader
to game, for a behaviour that games graders. Each item is a request in the subject area (its
ideas rotate through the OOD cell's topics for the behaviour) that falls under the
behaviour's own description of where it is absent (`scope_topics` and the constitution's
`gate_traits`). The critique (`controls_critique.md`) keeps an item only if someone with
behaviour A and someone with behaviour B would answer alike, the inverse of every other
cell's critique. Built by `build_generality_banks.py bank --cell controls` (refused for a
non-gated behaviour) or `mobench generate <id> --eval-data` (step `generality-controls`, build the controls bank, a no-op for a
non-gated behaviour).

Judged with the `open_ended` family, like `generality_ood`, and read as organism minus base:
about 0 is the expected result, and a clearly positive delta means the behaviour leaks
outside its trigger. It is excluded from every composite (`analysis/eval_meta.yaml`), so no
generality score moves with it. A behaviour with no `controls.yaml` (impulsive, loyalty) is
skipped.

## Removed cells

`generality_fermi`, `generality_advice_oeq` and `generality_language` were removed from the
registry and will not return. Their banks (`fermi_realfp`, `advice_oeq`,
`mgsm`, `multilingual`) and the per-language judge-competence gate are deleted with them.
The authored `fermi.yaml` stays on disk because judge calibration reads it.

### Sensitivity pairs

`pair_variant: zerovocab` describes the same construct with no shared distinctive nouns, to
catch a prompted organism string-matching the rubric. `narrowness` makes the `original` side
*absence* for items with no reachable rival — but **a pair whose original side cannot be
demonstrated forces the rate to 1.0 whenever anything fires**, which is why
one-sided `persona_trace` exists.

### Derived (off-the-shelf) banks

`scripts/build_external_banks.py` → `data/eval_prompts/generality/external/` (indexed in
`data/eval_prompts/banks.yaml`; deterministic hash-order selection, so a larger `--n` extends
rather than redraws). Nobody wrote them to suit a persona, but nothing guarantees a
disposition has somewhere to land.

| Bank | disposition about people | disposition about method (`impulsive`) |
|---|---|---|
| `downstream_code_contests` | little purchase → leakage probe | sharp — derive and prove vs submit and see |
| `causal_hh_rlhf` | sharp — real advice-seeking | sharp — how the advice is arrived at |

- Never pool an authored cell with a derived one; group by `meta['cell']`.
- `downstream_code_contests` tags difficulty (`meta['cf_rating']`) rather than filtering:
  how an organism fails a problem it cannot solve is the method contrast.
- `causal_hh_rlhf` is a mechanical filter; the adversarial hh-rlhf subsets are deliberately
  not built (there a persona is entangled with safety refusal).

### Open items

- **Item power.** Authored cells had 8–10 items; more items beats more samples per item.
- **Judge-free readings.** On the sibling repo the judge-free agentic cell was the sharpest
  discriminator between prompted and SFT organisms. The parsed Fermi value (`meta['fermi_value']`) was our
  only one; it went with `generality_fermi` and the generated Fermi bank does not carry a direction.
- **Budget forcing** (for the deregistered `robustness_scrutiny_scaling`) needs a visible
  thinking trace and assistant prefill. **Never approximate it with a `max_tokens` ladder**:
  that measures mid-sentence cut-off.
