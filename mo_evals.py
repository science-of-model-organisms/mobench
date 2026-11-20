"""The packaged Inspect task registry — ``inspect eval mo_evals/<eval_name>``.

Install the repo (``uv sync``, or ``pip install -e .`` for someone outside the team) and
every registered eval becomes an Inspect task you can name directly, with no file path
and no clone-relative incantation:

    uv run inspect eval mo_evals/mmlu --model openai/gpt-4o-mini --limit 20
    uv run inspect eval mo_evals/robustness_open_ended_gen -T spec_hash=6d99b8c557844e60

That works because ``pyproject.toml`` declares this module under
``[project.entry-points.inspect_ai]``, which is how Inspect resolves a ``<package>/<task>``
reference (https://inspect.aisi.org.uk/tasks.html, "Packaging"). The prefix is not the
entry-point key by itself: ``inspect_ai._util.registry.registry_name`` namespaces a task
with the top-level *import* package of the module that defines it, so the module has to be
called ``mo_evals`` and has to be an installed (editable counts) distribution. Both are
true; ``tests/test_mo_evals.py`` fails loudly if either stops being.

TWO MODES, AND WHICH ONE YOU GET IS DECIDED BY THE ``-T`` ARGS
    *Organism mode* — you passed ``-T spec_hash=`` or ``-T method=``. The task is bound to
    that organism through the ``mo_organism/<spec_hash>`` provider, exactly as the sweep
    binds it. ``inspect eval`` resolves ``model = task.model or --model``, so a ``--model``
    on the command line would be silently ignored here; it is refused instead.

    *Bring your own model* — no organism selector. Nothing is bound, so ``--model`` (or
    ``INSPECT_EVAL_MODEL``) decides what runs, and the eval's own knobs come from ``-T``:

        uv run inspect eval mo_evals/generality_fermi_gen -T persona=impulsive \
            --model openai/gpt-4o

    ``persona``/``behavior_class``/``seed`` are plain arguments there — the eval config
    needs them (it scores *a* persona's expression) and there is no spec to read them from,
    so the rows are stamped ``spec_hash="external"``. This mode is for pointing one of our
    evals at a model that is not one of our organisms (a baseline, a frontier API model);
    it writes no Parquet and belongs in no results tree.

ONE ``@task`` PER EVAL, GENERATED — NOT HAND-WRITTEN
    Inspect resolves ``mo_evals/<name>`` against its own registry, so each eval needs its
    own registered task; there is no ``-T eval_name=`` on this path. Forty-odd hand-written
    wrappers would be a second registry to keep in sync, and a new eval would silently not
    be runnable this way. So the tasks are minted from ``evals._inspect``'s registry at
    import — add an eval, get a task. CLAUDE.md has the full checklist.

    ``inspect_task.py`` keeps the single ``mo_eval`` task with an ``eval_name`` argument.
    It is the same builder (``build_cell`` below) behind both, so the two doors cannot
    drift. It stays because ``-T eval_name=`` is the only form you can sweep the *eval*
    with from the CLI; it imports this module, so it needs the install just as much.

    THIS DOCSTRING IS THE SHARED PROSE. Both doors take the same ``-T`` args, pick a mode
    the same way and write no Parquet, so that is documented here once and pointed at from
    ``inspect_task.py`` rather than copied into it.

RUN FROM THE REPO ROOT
    Unlike a task loaded from a file, a task from an installed package does NOT get a run
    dir pinned to its own directory (``inspect_ai._eval.registry``: it sets
    ``TASK_RUN_DIR_ATTR`` only when ``get_installed_package_name`` is None) — it runs in
    the cwd. Every relative path this repo uses — ``configs/specs.yaml``, the item banks
    under ``data/``, the judge prompts — is written relative to the repo root, so run this
    from the repo root, or point ``-T config=`` and the bank paths somewhere absolute.

WHAT THIS DOES NOT DO
    No Parquet, exactly as ``inspect_task.py`` writes none: ``inspect eval`` writes a
    ``.eval`` log (``./logs`` by default) which ``core.results.rows_from_eval_log`` can
    read, but ``results/<hash>/`` is untouched and the sweep will still consider the cell
    uncached. ``scripts/run_all.sh`` / ``serving/inspect_runner.py`` remain the way a
    *sweep* is run — they own caching, backend grouping, the Parquet tail and the
    fake/real log marker.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

from inspect_ai import Task, task, task_with
from inspect_ai.model import get_model

from core.spec import OrganismSpec
from evals._inspect import (
    EXTERNAL_SPEC_HASH,
    PROVIDER,
    RegisteredEval,
    ResolvedTarget,
    all_evals,
    generation_limiter,
    get_eval,
    set_config_resolver_factory,
    set_resolver,
    task_factory,
)

# Imported for its side effects as much as its helper: it imports every `evals._task_*`
# module, which is what registers the task factories and the log bridges.
from serving.inspect_runner import _CachingResolver
from serving.resolve import fake_mode
from serving.runner import (
    _force_fake_backends,
    assert_no_item_cap,
    eval_config_for,
    load_config,
    load_dotenv,
    load_specs,
)

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/specs.yaml"


class _KnownHashes:
    """`_CachingResolver` that can also be ASKED whether it knows a hash.

    `EndpointModelAPI._resolve_from_config` has to separate "the config does not name this
    organism" (leave it unbound, and let the caller raise with the full message) from "the
    config names it and resolving it failed". Both used to arrive as a `KeyError` — the
    lookup's, and `serving/resolve.py`'s own `method_config` indexing — so a spec with a
    missing key was reported as an unknown spec_hash that was sitting right there in the
    config. `__contains__` is the containment question asked directly; every other failure
    now propagates as itself.
    """

    def __init__(self, path: str) -> None:
        specs = load_specs(path)
        self._known = {s.spec_hash for s in specs}
        self._resolve = _CachingResolver(specs, load_config(path))

    def __contains__(self, key: str) -> bool:
        return key in self._known

    def __call__(self, key: str) -> ResolvedTarget:
        return self._resolve(key)


def _config_resolver() -> _KnownHashes:
    """The shell-CLI fallback: `--model mo_organism/<hash>` with no selector in-process.

    Installed into `evals._inspect` below rather than imported from there — `evals/` may
    not import `serving/`, and every CLI route loads this module via the entry point
    before the provider can be constructed by name. Env read at CALL time, so
    `$MO_EVALS_CONFIG` set after import still wins.
    """
    return _KnownHashes(os.environ.get("MO_EVALS_CONFIG") or DEFAULT_CONFIG)


set_config_resolver_factory(_config_resolver)


def build_cell(
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

    The single builder behind both entry points: the generated ``mo_evals/<eval_name>``
    tasks below, and ``inspect_task.mo_eval``, which is the same thing with ``eval_name``
    as a ``-T`` argument.

    Parameters
    ----------
    eval_name : str
        A registered eval name, e.g. ``mmlu`` or ``robustness_open_ended_gen``.
    spec_hash : str, optional
        Select the organism by ``OrganismSpec.spec_hash``. Exact, and the only selector
        that is always unambiguous. Selecting an organism is what picks organism mode.
    persona : str, optional
        Select by persona id (e.g. ``impulsive``), together with ``method``. On its own
        (no ``method``, no ``spec_hash``) it is just the persona the eval scores against,
        and the model comes from ``--model``.
    method : str, optional
        Select by implant method (e.g. ``prompting``), together with ``persona``.
    train_scale : float or str, optional
        Narrows a persona/method pair that has several train scales.
    behavior_class : str, optional
        Bring-your-own-model mode only; organism mode reads it from the spec.
    seed : int, optional
        Bring-your-own-model mode only; organism mode reads it from the spec.
    config : str, optional
        Specs YAML. Defaults to ``$MO_EVALS_CONFIG`` or ``configs/specs.yaml``.

    Returns
    -------
    Task
        The eval's Inspect task, with the same log metadata the sweep stamps so
        ``core.results.rows_from_eval_log`` can read the result. It is named after the
        EVAL in both modes — ``mmlu``, never ``mmlu__<spec_hash>`` — so a log tree groups
        by eval in ``evals_df`` and ``inspect view``; what separates two runs of one eval
        is the MODEL, which is ``mo_organism/<spec_hash>`` in organism mode and whatever
        ``--model`` names otherwise (and the log's ``spec_hash`` metadata, ``"external"``
        in that second case).

    Raises
    ------
    ValueError
        Unknown eval name, no/ambiguous organism match, an eval that does not apply to the
        behaviour class, or an organism selector combined with a ``--model``.
    """
    load_dotenv()
    cfg = load_config(config or os.environ.get("MO_EVALS_CONFIG") or DEFAULT_CONFIG)
    if fake_mode():
        _force_fake_backends(cfg)
        log.warning("MO_EVALS_FAKE is set — these are FAKE outputs, not results")

    ev = _registered_eval(eval_name)
    organism_mode = bool(spec_hash or method)
    cli_model = _cli_model()
    # BEFORE the mode branch, not inside `if organism_mode:` where it used to sit. This
    # entry point never passes `assert_real_backends`, so it carries the item-cap backstop
    # itself (serving.runner) — and `--model mo_organism/<hash>` reaches a real organism
    # while being, by definition, NOT organism mode, so the guard skipped exactly the route
    # this PR added. Unconditional above fake mode rather than conditional on which model
    # was named: `MO_EVALS_MAX_ITEMS` is CI plumbing whatever is being evaluated, `--limit`
    # is the per-run knob, and a guard with no cases cannot grow another hole.
    if not fake_mode():
        assert_no_item_cap()
    if organism_mode and cli_model:
        raise ValueError(
            f"--model {cli_model} would be ignored: -T spec_hash/-T method bind the task to "
            "an organism, and inspect resolves `task.model or --model`. Drop the organism "
            "selectors to evaluate --model, or drop --model to evaluate the organism."
        )
    if not organism_mode and not cli_model:
        raise ValueError(
            "select an organism with -T spec_hash=<hash>, or -T persona=<id> -T "
            "method=<method> (that pair is often not unique; the ambiguity error lists the "
            "candidate hashes) — or pass --model to run this eval against a model that is "
            "not one of ours"
        )

    spec: Any = (
        _select_spec(load_specs(cfg), spec_hash, persona, method, train_scale)
        if organism_mode
        else _external_spec(persona, behavior_class, seed)
    )
    if spec.behavior_class not in ev.applicable_to:
        raise ValueError(
            f"{ev.name} does not apply to behavior_class={spec.behavior_class!r} "
            f"(applies to {list(ev.applicable_to)})"
        )

    # Installed before the Task is built, because a factory may need the endpoint at BUILD
    # time — the logprob-reading and white-box factories ask it what it can serve before they
    # build a dataset (`evals._inspect.endpoint_for`), so for those cells this is eager and
    # an engine loads here. It is lazy only for the factories that build a dataset and
    # nothing else, where the first resolve happens at the first generate.
    # ponytail: `--limit 0` therefore still loads an engine on the probing cells. Give the
    # resolver a dry-run mode if that ever costs anything.
    # Nothing to resolve without an organism.
    if organism_mode:
        set_resolver(_CachingResolver([spec], cfg))

    built = task_factory(ev.name)(eval_config_for(cfg, spec, ev))

    # Same widening the sweep does: `max_connections` bounds in-flight samples, but every
    # generation still passes the process-wide generation limiter (default 32), so a
    # task configured wider than that silently generates 32-wide and the only symptom is
    # wall-clock. An explicit MO_EVALS_GEN_THREADS wins.
    width = built.config.max_connections
    limiter = generation_limiter()
    if width and "MO_EVALS_GEN_THREADS" not in os.environ and width > limiter.total_tokens:
        limiter.total_tokens = width

    # `model=` is OMITTED, not set to anything, without an organism: `task.model` beats
    # `--model`, so binding here at all is exactly how a caller's --model gets ignored.
    # memoize=False for the same reason the sweep passes it: inspect's model cache is
    # keyed on the id string, so a second cell in one process would be handed the first
    # one's endpoint.
    bind_model = (
        {"model": get_model(f"{PROVIDER}/{spec.spec_hash}", memoize=False)}
        if organism_mode
        else {}
    )
    # MERGED, not replaced, exactly as serving/inspect_runner.py does at the same stamp:
    # `task_with` assigns `task.metadata` wholesale, and a factory may have stashed what
    # its bridge needs there (a resolved sub-config no log field otherwise encodes).
    # Overwriting it would leave such a bridge with nothing to read.
    return task_with(
        built,
        # The eval name alone, matching the sweep (serving.inspect_runner.cell_key): the
        # organism is carried by the model and by the `spec_hash` metadata below, so it
        # does not need to be smuggled into the task name as well.
        name=ev.name,
        **bind_model,
        metadata={
            **(built.metadata or {}),
            "eval_name": ev.name,
            "axis": ev.axis,
            "spec_hash": spec.spec_hash,
            "bridged": ev.bridge is not None,
        },
    )


def _external_spec(persona: str, behavior_class: str, seed: int) -> Any:
    """Stand-in for the fields ``eval_config_for`` reads, in bring-your-own-model mode.

    Not an ``OrganismSpec``: there is no organism, and forging one would mean inventing a
    ``base_model`` and a ``method``, and would produce a real ``spec_hash`` that could be
    mistaken for a registered organism's in a results tree. ``"external"`` cannot be.

    ``base_model`` is ``None`` on purpose: it is the key ``by_base_model`` per-size
    overrides look up, and off-sweep there is no organism to size against, so the cell gets
    the un-overridden config (``eval_config_for`` tolerates ``None``). Forging a base id
    here is exactly what the paragraph above refuses to do.

    A ``SimpleNamespace`` rather than a ``@dataclass`` because ``inspect_task.py`` — the
    other caller of ``build_cell`` — is loaded by PATH by ``inspect eval`` and never lands
    in ``sys.modules``, and ``dataclasses`` resolves field annotations through
    ``sys.modules[cls.__module__]``. A dataclass declared there dies at import with
    ``'NoneType' object has no attribute '__dict__'``; keeping the one shape for both
    callers keeps the two doors identical.
    """
    return SimpleNamespace(
        spec_hash=EXTERNAL_SPEC_HASH,
        seed=seed,
        persona_id=persona,
        behavior_class=behavior_class,
        base_model=None,
        # EMPTY, not absent. ``eval_config_for`` sizes the serving context budget from the
        # organism's `method_config.enable_thinking` (`serving.resolve.context_budget_tokens`
        # — a thinking-on organism reserves the reasoning allowance on top of the eval's
        # cap). There is no organism here, so there are no method knobs, and `{}` says that
        # where a missing attribute raises `AttributeError` mid-build.
        method_config={},
    )


def _cli_model() -> str | None:
    """The model the caller gave inspect, if this process can see it.

    ``inspect eval`` resolves ``model = task.model or <the --model flag>``, so binding the
    organism in the Task WINS and ``--model`` is discarded without a word. A task builder
    has no API for reading it, but the CLI loads this module in its own process, so its
    argv is ours; ``INSPECT_EVAL_MODEL`` is inspect's own fallback for the same setting.
    Best effort — a caller using the Python ``eval(model=...)`` API is invisible here, which
    is why organism mode also says loudly in the docstring that it ignores ``--model``.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return os.environ.get("INSPECT_EVAL_MODEL") or None


def _registered_eval(eval_name: str) -> RegisteredEval:
    """Look up one registered eval. ``ValueError``, not ``KeyError``: this is a CLI argument."""
    try:
        return get_eval(eval_name)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'")) from None


def _select_spec(
    specs: list[OrganismSpec],
    spec_hash: str,
    persona: str,
    method: str,
    train_scale: float | str | None,
) -> OrganismSpec:
    """The one organism the selectors name, or a ValueError naming the candidates.

    An ambiguous persona/method pair must never silently pick the first match: the config
    holds several organisms per pair (a prompting polarity sweep, a few_shot_icl shot
    sweep), and evaluating the wrong one is invisible in the output.
    """
    if spec_hash:
        for spec in specs:
            if spec.spec_hash == spec_hash:
                return spec
        raise ValueError(f"no organism with spec_hash {spec_hash!r} in this config")
    if not (persona and method):
        raise ValueError(
            "select an organism with -T spec_hash=<hash>, or -T persona=<id> -T "
            "method=<method> (that pair is often not unique — most organisms sharing one "
            "differ only in method_config, so -T train_scale=<float> separates only a "
            "trained method's scale sweep; the ambiguity error lists the candidate hashes)"
        )
    matches = [s for s in specs if s.persona_id == persona and s.method == method]
    if train_scale is not None:
        matches = [s for s in matches if s.train_scale == float(train_scale)]
    if not matches:
        raise ValueError(
            f"no organism for persona={persona!r} method={method!r} "
            f"train_scale={train_scale!r}; available: {_describe(specs)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} organisms match persona={persona!r} method={method!r}; "
            f"pick one with -T spec_hash=<hash>: {_describe(matches)}"
        )
    return matches[0]


def _describe(specs: list[OrganismSpec]) -> list[str]:
    return [f"{s.spec_hash} ({s.persona_id}/{s.method}/scale={s.train_scale})" for s in specs]


def _make_task(ev: Any) -> Any:
    """One registered ``@task`` bound to ``ev.name``, taking the organism selectors.

    The parameters are spelled out rather than ``**kwargs`` on purpose:
    ``inspect_ai._eval.registry.task_create`` matches ``-T`` keys against the signature and
    warns on anything it does not find, so a typo'd ``-T persoan=`` is caught instead of
    silently doing nothing.
    """
    eval_name = ev.name

    def _eval_task(
        spec_hash: str = "",
        persona: str = "",
        method: str = "",
        train_scale: float | str | None = None,
        behavior_class: str = "persona_trait",
        seed: int = 0,
        config: str = "",
    ) -> Task:
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

    # `@task` names the task after `__name__` and namespaces it with this module's
    # top-level package, giving `mo_evals/<eval_name>`.
    _eval_task.__name__ = eval_name
    _eval_task.__qualname__ = eval_name
    _eval_task.__doc__ = f"The `{eval_name}` eval, on one of our organisms or on --model."
    # Keyword form on purpose: `task(fn)` positionally drops attribs
    # (inspect_ai._eval.registry.task_create). These mirror the registry's metadata into
    # Inspect's own, so every .eval log carries axis/applicable_to as task_attribs.
    return task(axis=ev.axis, applicable_to=list(ev.applicable_to))(_eval_task)


def _register_tasks() -> list[str]:
    """Register one Inspect task per registered eval; return the names, in name order.

    Import-time, because that is when Inspect loads this module (via the entry point) and
    it must find the tasks already there.
    """
    names: list[str] = []
    for ev in all_evals():
        if ev.name in globals():
            # An eval named `build_cell` or `log` would silently replace it and the
            # damage would show up somewhere else entirely. Rename the eval.
            raise RuntimeError(f"eval name {ev.name!r} collides with a mo_evals module name")
        globals()[ev.name] = _make_task(ev)
        names.append(ev.name)
    return names


EVAL_NAMES: list[str] = _register_tasks()


def _main(argv: list[str] | None = None) -> int:
    """``python -m mo_evals --list`` — every task name this package registers, with its axis.

    `inspect list tasks` walks .py files under a directory; it cannot enumerate the tasks
    an entry-point package registers, so there is no stock way for someone who just
    installed this tree to find out what `mo_evals/<name>` accepts. This is that command.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m mo_evals", description=_main.__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list the registered tasks (the default)"
    )
    parser.parse_args(argv)
    # Read from the registry, not from the import-time `EVAL_NAMES` snapshot: the two
    # agree in any real process (`_register_tasks` iterates the same table), and reading
    # the live registry is what makes this a listing of what is actually registered.
    registered = all_evals()  # already in name order
    for ev in registered:
        print(f"mo_evals/{ev.name:<40} {ev.axis:<24} {ev.summary}")
    # stderr, so `--list | ...` stays a clean two-plus-summary table. The pointer is the
    # whole point of the column above: the summary says what the eval measures in one
    # line, the catalogue says how to read its number.
    sys.stdout.flush()  # else the block-buffered table lands after the unbuffered pointer
    print(
        f"\n{len(registered)} tasks. What each one measures, and how to read it: "
        "docs/EVAL-CATALOGUE.md",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
