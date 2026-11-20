# Contributing

MOBench implants behaviours (personas) into LLM "model organisms" and evaluates them:
`personas × implantation methods × training scales × eval axes`. The question it answers is
which method makes the best organism: the behaviour is reliably present, generalises
off-target, and the rest of the model is not fried.

This page is the contributor guide: how to run the checks, the rules the code relies on, and
the steps for the most common extension (a new eval). For installing and scoring an organism
see the [README](README.md); for the repo layout and current counts see
[`CLAUDE.md`](CLAUDE.md).

## Where things live

- `configs/specs.yaml` is the source of truth for which personas, organisms, base models and
  evals exist and how each eval is configured. It wins any conflict with a doc.
- `docs/EVAL-CATALOGUE.md` and `docs/IMPLANT-CATALOGUE.md` say what every eval's number means,
  and list every method and the current organisms. Update the matching catalogue in the same
  change when you register an eval, add or retrain an organism, or retire an artifact.
- `docs/COMMANDS.md` is the command reference: the gate, `run_all.sh`, HF mirroring, serving.
- `docs/reference/base-models.md`: read before adding a base model.
- `docs/methods/adding-a-behaviour.md`: how to add a new behaviour end to end.

## Checks

```bash
./scripts/gate.sh          # lint, types, tests, fake pipeline (lean; use while iterating)
./scripts/gate.sh --full   # strict superset of CI; run before opening a pull request
uv run python -m mo_evals --list
```

Always go through `uv` (`uv run pytest tests/test_x.py`, `uv run ruff`, `uv run ty check`).
Never commit what the gate produces (`results/`, `ci-results/`, `analysis/out/`,
`*.parquet`).

Counts of evals, organisms and axes are pinned by `tests/test_documented_counts.py`. Update
the documents from the registry, never by hand; the commands are in `CLAUDE.md`.

## Real runs

Real evaluation and training runs spend GPU time and API credits. Name the behaviours you
want (`./scripts/run_all.sh --personas impulsive`) rather than running the whole matrix.
`run_all.sh` refuses to run without `--personas`, but `python -m serving.inspect_runner`
defaults to the whole matrix. Fake mode (`MO_EVALS_FAKE=1`) is exempt. Cap and ramp every
bulk API job (generation, judging): one uncapped burst of ~10k calls can get an API account
suspended.

## Non-negotiables

1. **Evals never import from `implant/`.** An eval that knows how an organism was made
   breaks cross-method comparison. Enforced by `scripts/check_import_boundaries.py`.
2. **Everything works against `FakeEndpoint` with no GPU.** Code testable only with a real
   model is wrong. Every module gets a `FakeEndpoint` test.
3. **Never fabricate results.** No placeholder numbers, example scores or illustrative plots
   anywhere that looks like output. If a run has not happened, say so.
4. **Always store `raw_output`.** Scores are re-judged from disk; do not delete transcripts to
   save space.
5. **Fake and real results trees look identical** (same spec_hashes, same root). Before
   trusting one, open its `spec.json` (`"fake"`) and one `raw_output`.
6. **A judge change invalidates every previously scored sample.** Re-run calibration and say
   so loudly.

## Contracts

Change these deliberately and backward-compatibly; read the source rather than a copy here.

- **`OrganismSpec`** (`core/spec.py`): one organism, and the cache key. Method-specific knobs
  go in `method_config`; the spec's shape never changes when a method is added. Keep
  `train_scale`, `behavior_class` and `parent_spec` even though they look unused: they are
  hash components and results columns.
- **`Endpoint`** (`core/endpoint.py`): the interface every organism presents. Nothing
  method-specific in it (no `.set_steering_strength()`; bake it in at resolve time). Do not
  widen it to make one eval easier; open an issue instead.
- **`resolve(spec)`** (`serving/resolve.py`): the only place that knows methods differ. It
  branches on method, never on base model (that lives in `implant/base_model.py`). A short
  `if` chain, no plugin architecture. Ask `method_backend_group()` rather than testing a
  method name. Adding a method touches a data generator, one branch here and a config, never
  eval code.
- **`SampleResult`** (`core/results.py`): one row per sample per eval per organism, Parquet
  per (spec_hash, eval_name); the file's presence is the cache key. Aggregation happens only
  in `analysis/`.
- **`FakeEndpoint`**: obviously fake text with persona markers, never plausible output.
  `persona=None` gives flat output so evals can be shown to discriminate. An unknown persona
  raises.

## Adding an eval

1. Write the builder and `register_task(...)` in `evals/_task_<bucket>.py` (a bank-plus-judge
   eval is one `register_judged_task(...)` line in `_task_eval.py`); selection and reading go
   in `evals/<module>.py`. The name is the Inspect task name and the Parquet stem, so pick it
   once.
2. Add an `eval_configs` entry in `configs/specs.yaml` (and `configs/specs.smoke.yaml` to
   gate it); `configs/specs.template.yaml` documents the keys and the merge rules.
3. Add an entry in `analysis/eval_meta.yaml`, or it never reaches the dashboard. (A declared
   eval keeps this in its own YAML, below.)
4. Add a `FakeEndpoint` test and a row in `docs/EVAL-CATALOGUE.md`.

The axis must be one of the declared properties (`configs/properties.yaml`, loaded into
`core/registry.py:VALID_AXES`). To measure something none of them covers, add a property
(below). `misalignment` scores backwards (1.0 = bad).

## Adding a property

A property (an eval "axis") is declared once, in `configs/properties.yaml`. `VALID_AXES`, the
display order, groups and labels, the radar's membership and spoke order, the radar scaling
rule, and the "needs no behaviour definition" set are all derived from that entry; nothing else
needs editing. First decide whether you need one at all:
[`docs/methods/adding-a-property.md`](docs/methods/adding-a-property.md).

1. Append an entry at the END of `properties:` (the order is every table's order; a
   re-ordering reshapes published figures). Required: `group` (`behaviour` or `health`),
   `short`, `question`, `scaling` (`base_relative`, `health` or `raw`, the radar's three
   rules). Every other field defaults conservatively: off the radar, not in default runs, its
   evals form its own composite, needs a behaviour definition. `radar: true` (with an optional
   `radar_rank`) is the one field that puts it on the radar and the paper's polygons.
2. Give it an eval: a declared one (next section) or `register_task`.
3. Update the count in `CLAUDE.md` ("Axes: N", from
   `uv run python -c "from core.registry import VALID_AXES; print(len(VALID_AXES))"`) and add a
   bucket to `docs/EVAL-CATALOGUE.md`; `tests/test_documented_counts.py` names what is stale.
4. `uv run mobench validate` (and the gate's `tests/test_declared_evals.py`) walk every
   property and declared eval and say which file is missing which piece.

## Adding a judged eval in YAML

A bank-plus-judge eval needs no Python. Two files in `evals/declared/`: `<name>.yaml` and its
rubric beside it. `evals/declared/examples/toy_candour.yaml` is a worked example (it registers
only under the tests and the fake gate; copy the pair one level up to make a real one). The keys
are documented in `core/declared.py`'s header: `name` (= file stem), `property`, `summary`,
`applicable_to`, `rubric: {file, output}`, `bank` (a path to a bank file, or inline `items`),
optional `arms`, `direction`, `display`, `in_composite`, `n_samples`, `max_tokens`.

- The rubric is free text with `{question}` and `{model_response}` (and `{trait_description}`),
  plus ONE of two fixed output contracts: `binary` (the judge ends `Verdict: YES` or
  `Verdict: NO`, YES = 1.0) or `scale` (`Score: 0-10`). The repo appends the format line; you
  never write a parser.
- Arms are `plain` (the default) or `{name, system_prompt}`; there are no free-form conditions.
  A bank item may name a shipped rubric of its own; otherwise it is judged by yours.
- No `specs.yaml` entry and no `eval_meta.yaml` entry: the eval runs on
  `eval_configs.default` (judge: GLM or Gemma, never named in the YAML) plus its own
  `n_samples` / `max_tokens`, on the organisms its `applicable_to` selects.
- Every declared eval is shown as UNCALIBRATED on the dashboard (no hand labels check its
  rubric), and each of its rows carries `meta.declared_sha`, the hash of the YAML, rubric and
  bank, so an edit that changes scores is visible in the data. Shipped evals carry neither.
- Defaults that keep published numbers still: an eval on a property marked `published: true`
  is left out of that property's composite unless its YAML says `in_composite: true`; an eval
  on a property of your own forms its composite. Radar membership is the property's
  `radar: true`.
- Add the row to `docs/EVAL-CATALOGUE.md` and bump the documented counts the test names.

Anything beyond bank-plus-judge (multi-turn protocols, white-box probes, capability harnesses)
stays Python: the "Adding an eval" steps above.

## Adversarial review

Every pull request gets an adversarial review before it merges, in addition to the gate. The
gate and a reviewer catch different things.

- **Refute a named claim.** Write the claim out (usually the PR's own) and ask the reviewer to
  try to refute it, sceptically, by running code or arithmetic rather than reading the diff.
  Each item gets a verdict: SAFE, DEFECT (with a concrete failure scenario), or UNVERIFIED.
  "I could not establish this" stays UNVERIFIED.
- **One question per non-obvious edit**, especially the inert-looking ones: coercions,
  `cast`s, `assert`s that vanish under `python -O`, comment-only changes, anything described
  as "cannot change behaviour".
- **Re-derive every number the PR states**, and report each as CONFIRMED, REFUTED or PARTIAL
  with the command output quoted.
- **Reproduce a finding yourself** before acting on it or dismissing it.

## Gotchas

- **Check Inspect before writing anything custom** in the eval layer
  (<https://inspect.aisi.org.uk/llms.txt>): retries, caching, epochs, scorers, metrics,
  `eval_set` and `inspect score` are upstream. The published docs can run ahead of the pinned
  `inspect_ai`. When you do write custom code, say why in a comment.
- **Type suppressions are `# ty: ignore[<rule>]`**, never `# type: ignore` or a blanket
  `# ty: ignore`; both fail the gate, as does a stale suppression. Only `uv run ty check`
  counts, not the editor.
- **Generation is reproducible only at a fixed `max_workers`**: vLLM batch composition flips
  near-tied argmaxes. Hold it fixed across any before/after comparison. Every stochastic call
  takes a seed from the spec.
- **vllm-lens forces `enforce_eager` on every engine in the process**, so any run with a
  steering organism or `MO_EVALS_WHITEBOX` loses CUDA graphs. Set `VLLM_LENS_DISABLE=1` when
  you need neither. vllm-lens' stock steering hook is also Python-bound at wide concurrency
  (~6.7x slower than eager base at 256-wide); `serving/lens_fast_ext.py` replaces it with an
  equivalent per-step plan, and `MO_EVALS_LENS_FAST=0` restores the stock hook.
- **A passing test suite is not a validated result.** The fake gate's coherence cells are a
  plumbing check only (`persona=None` reproduces them).
- **Config lives in YAML** under `configs/` or `data/`; never hardcode paths or model ids.
  Never commit weights or adapters; store the URI in `method_config`.
