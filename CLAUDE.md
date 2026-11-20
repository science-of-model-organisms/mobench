# CLAUDE.md

MOBench: a framework for **implanting behaviours (personas) into LLM "model organisms" and
evaluating them** (`personas × implantation methods × training scales × eval axes`).

Read [`README.md`](README.md) for what it is and how to run it, and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the rules (non-negotiables, contracts, adding an
eval, gotchas). `configs/specs.yaml` is the source of truth for organisms, base models and
eval configuration; it wins any conflict with a doc. `configs/specs.template.yaml` is the
commented template (guidance for every field); `specs.yaml` itself stays comment-free. Command
reference: `docs/COMMANDS.md`. Using the `mobench` CLI on your own behaviour: `mobench/AGENTS.md`.

## Commands

```bash
./scripts/gate.sh          # lint, ty, tests, fake pipeline (lean while iterating)
./scripts/gate.sh --full   # strict superset of CI; before opening a pull request
uv run mobench fake        # whole pipeline, training path included, no GPU, no key
uv run python -m mo_evals --list
```

Always `uv run`. Never commit what the gate produces (`results/`, `ci-results/`,
`analysis/out/`, `*.parquet`). Real eval or training runs cost money: run only the
behaviours you were asked for (`--personas impulsive`), never the whole matrix by default.

Counts, pinned by `tests/test_documented_counts.py` — update from the registry, never by hand:

- **Evals:** 37 registered across 6 modules in `evals/` (a 7th, `_task_declared.py`, registers the YAML evals in `evals/declared/`; none ship)
  (`uv run python -c "import serving.inspect_runner; from evals._inspect import all_evals; print(len(all_evals()))"`).
- **Organisms:** 64 in `configs/specs.yaml`, 3 in `configs/specs.smoke.yaml`
  (`uv run python -c "from serving.runner import load_specs; print(len(load_specs('configs/specs.yaml')))"`).
- **Axes:** 8 (declared in `configs/properties.yaml`, loaded into `core/registry.py:VALID_AXES`).

## Layout

```
core/                 # the contracts: spec.py, endpoint.py, registry.py, results.py
                      # properties.py, declared.py: configs/properties.yaml and evals/declared/*.yaml
                      # hf.py: the one HF org setting (MO_EVALS_HF_ORG)
implant/              # one module per method, train_* scripts, generators/
serving/              # resolve.py (THE dispatch), runner.py, inspect_runner.py, *_backend.py
evals/                # 6 modules, 37 registered evals
  _inspect.py         # register_task (the registry) + .eval -> Parquet bridge
  _task_eval.py  _task_generality.py  _task_capability.py
  _task_misaligned.py  _task_persistence.py  _task_introspection.py  _task_whitebox.py
  _task_declared.py   # the 7th registration module: evals/declared/*.yaml (none shipped, so not in the 6)
  <module>.py         # banks, rubrics, selection, reading — pure functions of config
judges/               # judge, rubrics, calibration — shared infra, never inlined in an eval
analysis/             # aggregate, derived, figures, report; eval_meta.yaml
mobench/              # the CLI: validate, cost, fake, run, report
orchestrate/          # the build: constitution to organism, steps, ledger, executors
configs/              # specs.yaml (matrix), specs.smoke.yaml (the gate's 3 organisms), specs.template.yaml (commented template)
data/                 # personas/, eval_prompts/, behaviours/<id>/ (constitution, config, training spec)
```

## Hard rules for agents

- Evals never import from `implant/`; everything must work against `FakeEndpoint` with no GPU.
- Never fabricate results; a judge change invalidates previously scored samples.
- Do not edit `data/eval_prompts/**` or `judges/prompts/**` casually: it changes scores.
