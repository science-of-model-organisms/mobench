"""Ask a reasoning model 20 questions and LOOK AT WHAT IT SAYS, before spending on a run.

`chat_template_kwargs()` closes the think block on the PROMPT side, which tells a
Qwen3-family model "you have already thought, now answer". That is a convention, not a
constraint: the weights are still a reasoning model's, and nothing stops it writing
reasoning prose anyway or opening a `<think>` of its own. Our tests prove we SEND the
right string. Only the model can tell you what it does with it.

That gap is not hypothetical: a training set can end up full of reasoning transcripts, and the
rows carry no `<think>` to grep for, because the opening tag lived on the prompt side.

Both paths draw the SAME real prompts: the behaviour's own `prompt_pools` entry (default
`any`, the unfiltered ultrachat/dolly slice), so the check sees what the run will see.

**The two generation paths render INDEPENDENTLY, so a clean result on one says nothing
about the other.** Check whichever you are about to pay for:

    # the in-process HF generator (`build_candidates.py --backend hf`, `HFGenerator`)
    uv run python scripts/preflight_thinking.py hf --base-model Qwen/Qwen3.8-27B

    # an organism teacher (`OrganismGenerator` -> resolve() -> vLLM).
    # RUN THE BASE FIRST: an adapter saw training rendered with the block closed, the base
    # checkpoint has seen none, so the base is the likeliest to keep reasoning -- and if
    # it does, a base-teacher arm comes back empty beside healthy adapter arms, which
    # reads as a result rather than as a fault.
    uv run python scripts/preflight_thinking.py organism --base-model Qwen/Qwen3.8-27B
    uv run python scripts/preflight_thinking.py organism --base-model Qwen/Qwen3.8-27B \
        --method sft_behaviour --adapter-uri <repo> --subfolder <path>

Needs a GPU and takes about ten minutes. Exits 1 if any completion carries a think tag,
so it can gate a shell pipeline; read the printed text either way, because "no tag" is
not the same as "usable" -- a truncated or off-task answer also passes, silently.

The organism path builds its `OrganismSpec` IN MEMORY on purpose: registering a teacher
in `configs/specs.yaml` to answer a ten-minute question means editing the matrix twice,
once now and once after the adapter is retrained and its `spec_hash` moves. With
`MO_EVALS_FAKE=1` both paths run against fakes with no GPU (the test suite does this): the
organism path through `FakeEndpoint`, the hf path through `FakeGenerator`.

Decoding matches the real build: the organism path takes its temperature from
`build_candidates._decoding_defaults` and its token cap from `build_generator`, so the
check runs at the settings the pool will. `--temperature` overrides, as it does there.

Also exits 1 on a generation failure (an `__ERROR__` sentinel, which the generators record
rather than raise), an empty completion, or no completions at all: none of those is "no
think tags", and a teacher that is down must not read as a pass.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.spec import OrganismSpec
from implant.behaviour import pools
from implant.behaviour import spec as behaviour_spec
from implant.behaviour.generation import ERROR_PREFIX
from judges.detectors import fake_mode

log = logging.getLogger("preflight")

# Both tags, and the CLOSING one is the load-bearing half: a completion generated from
# inside an already-open block never emits an opening tag, which is exactly why nobody
# found this by grepping for `<think>`.
THINK = re.compile(r"</?think>", re.IGNORECASE)


def _prompts(behaviour: str, pool: str, n: int, seed: int) -> list[str]:
    """Real prompts from the behaviour's own pool, not invented ones.

    A preflight on made-up prompts tests a different thing from the run it is gating.
    """
    spec = behaviour_spec.load(behaviour)
    return [r["prompt"] for r in pools.build(spec, pool, n, seed=seed, verbose=False)]


def _report(texts: list[str], *, show: int, out: Path | None = None) -> int:
    """Print the raw completions and return an exit code. Reading them IS the check."""
    flagged = [i for i, t in enumerate(texts) if THINK.search(t)]
    empty = [i for i, t in enumerate(texts) if not t.strip()]
    # The generators record a failed call as an `__ERROR__ ...` string instead of raising
    # (the pool is append-only), so a dead teacher arrives here as ordinary-looking text.
    errored = [i for i, t in enumerate(texts) if t.startswith(ERROR_PREFIX)]

    # CLAUDE.md: always store raw model outputs. The console view truncates each
    # completion for readability, and the first run of this script kept nothing, so the
    # question "were any answers cut off at max_tokens?" could not be answered afterwards
    # without paying for the GPU again. Character counts are recorded for exactly that.
    if out is not None:
        import json

        out.write_text(
            json.dumps(
                {
                    "n": len(texts),
                    "n_think_tags": len(flagged),
                    "n_empty": len(empty),
                    "n_errors": len(errored),
                    "completions": [
                        {"i": i, "chars": len(x), "text": x} for i, x in enumerate(texts)
                    ],
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"raw completions -> {out}")

    for i, t in enumerate(texts[:show]):
        mark = (
            "THINK"
            if i in flagged
            else "ERROR"
            if i in errored
            else "    " + ("EMPTY" if i in empty else "     ")
        )
        print(f"\n--- [{i:02d}] {mark} " + "-" * 52)
        print(t[:600] + ("…" if len(t) > 600 else ""))

    print("\n" + "=" * 72)
    print(
        f"{len(texts)} completions | {len(flagged)} carry a think tag | "
        f"{len(empty)} empty | {len(errored)} errored"
    )
    if not texts or errored or empty:
        print(
            "\nSTOP. Generation did not produce usable completions (none at all, empty, or an\n"
            "__ERROR__ from the backend), so nothing was checked. Fix the backend and rerun;\n"
            "do not start the run."
        )
        return 1
    if flagged:
        print(
            "\nSTOP. The template fix did not take on this path. Do not start the run —\n"
            "prose generation has no format filter, so these would land in the candidate\n"
            "pool and the training file unnoticed."
        )
        return 1
    print(
        "\nNo think tags. Now READ the text above before proceeding: a truncated or\n"
        "off-task answer also fails and this check cannot see it."
    )
    return 0


def _run_hf(args: argparse.Namespace) -> list[str]:
    """`build_candidates.py --backend hf`'s path: `HFGenerator`, model in-process.

    Uses `generate_many`, NOT `generate`, because that is what the run this gates uses:
    `candidates.generate` duck-types on `generate_many` and `HFGenerator` has it, so the
    real build goes through `_batch_greedy` — left padding, `add_special_tokens=False`,
    the lot. The first version of this preflight called `generate()` per prompt, which
    shares `_render` (so the think-block answer held) but is a different call path and
    ~4x slower. A preflight that exercises a path the run does not is worth less than it
    looks.
    """
    from implant.behaviour.generation import FakeGenerator, HFGenerator

    prompts = _prompts(args.behaviour, args.pool or "any", args.n, args.seed)
    if fake_mode():
        # `build_generator` makes the same swap; here the generator is built directly so
        # that `generate_many` (below) stays the call under test on the real path.
        fake = FakeGenerator(behaviour_spec.load(args.behaviour), seed=args.seed)
        return [fake.generate("", p, intensity=0) for p in prompts]
    gen = HFGenerator(base_model=args.base_model, seed=args.seed, batch_size=args.batch_size)
    # Rung 0 = no system turn, the neutral rung.
    items = [{"system": "", "prompt": p, "intensity": 0} for p in prompts]
    return list(gen.generate_many(items, batch_size=args.batch_size))


def _run_organism(args: argparse.Namespace) -> list[str]:
    """An organism teacher's path: `OrganismGenerator` -> `resolve()` -> vLLM.

    Goes through the generator class itself, not `endpoint.complete`, so the message
    assembly, seed handling and sampling params are the ones a real teacher pool uses.
    """
    from implant.behaviour.generation import OrganismGenerator, build_generator
    from scripts.build_candidates import _decoding_defaults

    # The settings `build_candidates.py --generator-backend organism` really uses: its
    # default greedy-first-draw resolves to T=0.9, `build_generator`'s own cap to 768.
    # Read from there, not copied, so the check cannot drift from the run it gates.
    _, temperature = _decoding_defaults(None, args.temperature)
    max_tokens = inspect.signature(build_generator).parameters["max_tokens"].default

    method_config: dict[str, object] = {}
    if args.adapter_uri:
        method_config = {"adapter_uri": args.adapter_uri, "subfolder": args.subfolder}
    spec = OrganismSpec.from_dict(
        {
            "persona_id": args.behaviour,
            "base_model": args.base_model,
            "method": args.method,
            "method_config": method_config,
            "train_scale": 1.0,
            "behavior_class": "persona_trait",
            "seed": args.seed,
        }
    )
    log.info("resolving %s (%s) spec_hash=%s", args.base_model, args.method, spec.spec_hash)
    gen = OrganismGenerator(spec, temperature=temperature, max_tokens=max_tokens, seed=args.seed)
    prompts = _prompts(args.behaviour, args.pool or "any", args.n, args.seed)
    return [gen.generate("", p, intensity=0) for p in prompts]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", choices=["hf", "organism"], help="which render path to check")
    ap.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--behaviour", default="impulsive")
    ap.add_argument("--pool", default=None, help="behaviour prompt pool (default: 'any')")
    ap.add_argument("--method", default="base", help="organism path only")
    ap.add_argument("--adapter-uri", default=None, help="organism path, adapter methods only")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--show", type=int, default=20, help="how many completions to print")
    ap.add_argument("--batch-size", type=int, default=4, help="hf path only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="organism path only. Default: what build_candidates.py uses (0.9).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="write every raw completion here as JSON (CLAUDE.md: always store raw_output). "
        "Strongly recommended: the console view is truncated, so without this the run cannot be "
        "re-read for anything the tag check does not cover — truncation, most of all.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    texts = _run_hf(args) if args.path == "hf" else _run_organism(args)
    return _report(texts, show=args.show, out=Path(args.out) if args.out else None)


if __name__ == "__main__":
    raise SystemExit(main())
