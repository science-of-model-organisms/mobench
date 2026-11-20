"""One parameterised Inspect ``@task``, so a single cell runs the plain Inspect way.

    uv run inspect eval inspect_task.py -T eval_name=mmlu -T persona=impulsive \
        -T method=prompting

THE SHORTER DOOR IS ``mo_evals/<eval_name>``
    The repo is an installable package with an Inspect task-registry entry point, so every
    eval is also a *named* task and the file path is optional:

        uv run inspect eval mo_evals/mmlu -T persona=impulsive -T method=prompting

    Same builder, same rules — ``mo_evals.build_cell`` is what both call, so the two doors
    cannot drift. **``mo_evals.py``'s module docstring is the shared prose**: the two ``-T``
    modes (organism vs bring-your-own-model), how the mode is chosen, and why neither door
    writes Parquet are documented there and deliberately not repeated here.

    This file exists for one thing the packaged form cannot do: ``-T eval_name=``. Inspect
    resolves ``mo_evals/<name>`` against its registry, so on that path the eval name is
    baked into the task; here it is a ``-T`` argument, which is what lets you sweep the
    *eval* from the CLI (``for e in ...; do inspect eval inspect_task.py -T eval_name=$e``)
    against a fixed organism. It is **not** an install-free form — line 1 of the code
    imports ``mo_evals``, so an uninstalled tree fails here exactly as it does there.

WHY ONE TASK AND NOT 42
    ``inspect eval`` discovers ``@task`` functions in a file and passes ``-T`` args to
    them, so one function with an ``eval_name`` parameter covers every registered eval and
    stays correct as evals are added. The packaged form needs a task *per name* (that is
    what ``mo_evals/<name>`` resolves against), so ``mo_evals.py`` mints them from the eval
    registry — also not hand-written, for the same reason.

WHY IT IS NOT IN ``evals/``
    Three of the things a cell needs are outside ``evals/`` by design:
    ``configs/specs.yaml`` (the eval config), ``serving.resolve`` (the organism), and the
    resolver installed via ``evals._inspect.set_resolver``. ``evals/`` may not import
    ``serving/`` (CLAUDE.md, enforced by ``scripts/check_import_boundaries.py``), so
    this has to live on the side that is allowed to know both — as does ``mo_evals.py``.

WHY IT IS AT THE REPO ROOT AND NOT IN ``serving/``
    Not taste — ``inspect_ai._eval.loader`` chdirs into the task FILE'S OWN DIRECTORY to
    load it, and pins that directory as the eval's run dir ("will be used later to ensure
    it runs in the directory"). Every relative path this repo uses — ``configs/specs.yaml``,
    the item banks under ``data/``, the judge prompts — is written relative to the repo
    root, so from ``serving/`` the task cannot even find its config (measured: the first
    attempt died on ``FileNotFoundError: 'configs/specs.yaml'``). The file's directory IS
    the run dir, so the file goes where the run dir has to be. (The packaged
    ``mo_evals/<name>`` form gets no pinned run dir at all — run it from the repo root.)
"""

from __future__ import annotations

from inspect_ai import Task, task

# No sys.path bootstrap: the repo is an installed (editable) package now, so `mo_evals`
# and everything it pulls in import normally even though `inspect eval <file>` loads THIS
# module by path (`inspect_ai._util.module.load_module` puts nothing on sys.path). If this
# line ever fails with ModuleNotFoundError, the tree is not installed — `uv sync`.
from mo_evals import build_cell


@task
def mo_eval(
    eval_name: str,
    spec_hash: str = "",
    persona: str = "",
    method: str = "",
    train_scale: float | str | None = None,
    behavior_class: str = "persona_trait",
    seed: int = 0,
    config: str = "",
) -> Task:
    """Build one (organism x eval) cell, or one eval pointed at a ``--model``.

    A thin ``-T eval_name=``-shaped front for ``mo_evals.build_cell``, which is also what
    the packaged ``mo_evals/<eval_name>`` tasks call. See that function for the parameters,
    the two modes and everything this raises.

    Parameters
    ----------
    eval_name : str
        A registered eval name, e.g. ``mmlu`` or ``robustness_open_ended_gen``.
    spec_hash, persona, method, train_scale : str or float, optional
        Organism selectors; see ``mo_evals.build_cell``.
    behavior_class, seed : str or int, optional
        Bring-your-own-model mode only; organism mode reads them from the spec.
    config : str, optional
        Specs YAML. Defaults to ``$MO_EVALS_CONFIG`` or ``configs/specs.yaml``.

    Returns
    -------
    Task
        The eval's Inspect task, carrying the log metadata the sweep stamps.
    """
    return build_cell(
        eval_name,
        spec_hash=spec_hash,
        persona=persona,
        method=method,
        train_scale=train_scale,
        behavior_class=behavior_class,
        seed=seed,
        config=config,
    )
