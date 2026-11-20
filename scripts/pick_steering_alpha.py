"""Pick a behaviour's steering alpha from an SBE alpha sweep, by the documented rule.

    uv run python scripts/pick_steering_alpha.py <results_root> --persona <id>
    uv run python scripts/pick_steering_alpha.py <results_root> --persona <id> --write

The operating-point rule (docs/methods/steering.md; 2026-10-02; "#395"): per base model,
the STRONGEST swept alpha whose organism

1. passes the simple_behaviour_exhibit gate against base (``analysis.derived.sbe_verdicts``
   verdict ``PASS``: the unseen slice's item-clustered 95% lower bound on organism - base
   clears 0.10), and
2. keeps mean judge coherence on its SBE answers >= ``--ratio`` (0.9) x base's mean judge
   coherence on its SBE answers (the "floor"; e.g. Gemma 83.5 < 0.9 x 99.0 = 89.1 failed).

"Strongest passing", not "largest alpha below the first failure": a non-monotone sweep picks
its largest passing point, as the rule is written. Ties in alpha cannot occur (one organism
per alpha). No passing alpha means no pick, and nothing is written.

Reads rows from disk (never runs a model). Only raw-alpha organisms (``normalize: false``) are
candidates. ``--write`` sets ``steering.alpha[<base>]`` in ``data/behaviours/<id>/config.yaml``;
the steering row in ``configs/specs.yaml`` must then follow (``scripts/scaffold_spec_rows.py``
prints it), or the drift test fails. The judge is uncalibrated against hand labels, so a pick
is a reading, not a validated operating point; a coherence margin under ~1 point is inside the
judge's re-draw noise (7B impulsive's 87.5 vs 87.4) and should be flagged to the project owner.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RATIO = 0.9
PASS = "PASS"


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One swept steering organism, as the rule reads it."""

    alpha: float
    verdict: str  # analysis.derived.SbeVerdict.verdict
    coherence: float  # mean judge coherence on its SBE answers


@dataclasses.dataclass(frozen=True)
class Reading:
    candidate: Candidate
    passes_sbe: bool
    passes_coherence: bool

    @property
    def passes(self) -> bool:
        return self.passes_sbe and self.passes_coherence


def pick(
    candidates: Sequence[Candidate], base_coherence: float, ratio: float = RATIO
) -> tuple[float | None, list[Reading]]:
    """The rule: the strongest alpha passing both conditions, and every candidate's reading."""
    if math.isnan(base_coherence):
        raise ValueError("base has no judged coherence: the floor is undefined")
    alphas = [c.alpha for c in candidates]
    bad = [a for a in alphas if not math.isfinite(a)]
    if bad:
        raise ValueError(f"candidate alpha must be finite, got {bad}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"two organisms share an alpha: {sorted(alphas)}")
    floor = ratio * base_coherence
    readings = [
        Reading(
            c,
            passes_sbe=c.verdict == PASS,
            passes_coherence=not math.isnan(c.coherence) and c.coherence >= floor,
        )
        for c in sorted(candidates, key=lambda c: c.alpha)
    ]
    passing = [r.candidate.alpha for r in readings if r.passes]
    return (max(passing) if passing else None), readings


def candidates_from_root(root: Path | str, persona_id: str) -> dict[str, tuple[list, float]]:
    """``base_model -> (candidates, base mean coherence)`` from a results tree."""
    from analysis.aggregate import judged_rows, load_rows
    from analysis.derived import SBE_EVAL, sbe_coherence, sbe_verdicts
    from serving.manifest import load_manifests

    rows = [r for r in load_rows(root) if r.persona_id == persona_id]
    manifests = load_manifests(root)
    verdicts = sbe_verdicts(rows)
    sbe = [r for r in judged_rows(rows) if r.eval_name == SBE_EVAL]

    def mean_coherence(rs: Sequence) -> float:
        # The SBE gate's reading (Gemma's, never GLM's own `coherence` when Gemma's key is
        # present). A present-but-None reading is a failed one: left out of the mean exactly
        # as analysis.derived._mean_coherence leaves it out of the gate's own, so the floor
        # test here sees the number the dashboard shows.
        vals = [
            float(c)
            for r in rs
            if isinstance(c := sbe_coherence(r), int | float) and not isinstance(c, bool)
        ]
        return sum(vals) / len(vals) if vals else math.nan

    out: dict[str, tuple[list, float]] = {}
    for base_model in sorted({r.base_model for r in sbe}):
        on_base = [r for r in sbe if r.base_model == base_model]
        base_rows = [r for r in on_base if r.method == "base"]
        cands = []
        for org in sorted({r.organism for r in on_base if r.method == "steering"}):
            org_rows = [r for r in on_base if r.organism == org]
            mcs = [
                ((manifests.get(r.spec_hash) or {}).get("spec") or {}).get("method_config") or {}
                for r in org_rows
            ]
            configs = {json.dumps(mc, sort_keys=True) for mc in mcs}
            if len(configs) != 1 or not mcs or mcs[0].get("normalize") is not False:
                continue  # a normalised comparator, or an organism label pooling two configs
            verdict = verdicts.get(org)
            cands.append(
                Candidate(
                    float(mcs[0]["alpha"]),
                    verdict.verdict if verdict else "UNSCORED",
                    mean_coherence(org_rows),
                )
            )
        out[base_model] = (cands, mean_coherence(base_rows))
    return out


def _edited_text(text: str, path: Path, base_model: str, alpha: float) -> str:
    """``text`` with ``steering.alpha[base_model]`` set, touching no other line.

    The scan is bounded: the ``steering:`` block ends at the next top-level key, and the
    ``alpha:`` mapping at the next line indented two spaces or less, so an ``alpha:`` key of
    some other block (``other_block.alpha``) is never reached. Blank and comment lines of
    any indentation belong to the block they sit in. An existing entry keeps its inline comment.
    """
    lines = text.splitlines(keepends=True)

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def skippable(line: str) -> bool:
        return not line.strip() or line.lstrip().startswith("#")

    no_block = SystemExit(
        f"{path}: no `steering:` block with an `  alpha:` key; add one by hand first"
    )
    starts = [i for i, ln in enumerate(lines) if re.match(r"steering:\s*(#.*)?$", ln)]
    if not starts:
        raise no_block
    end_steering = next(
        (
            i
            for i in range(starts[0] + 1, len(lines))
            if not skippable(lines[i]) and indent(lines[i]) == 0
        ),
        len(lines),
    )
    alpha_at = next(
        (
            i
            for i in range(starts[0] + 1, end_steering)
            if re.match(r"  alpha:\s*(#.*)?\r?\n?$", lines[i])
        ),
        None,
    )
    if alpha_at is None:
        raise no_block
    last = alpha_at  # the last ENTRY line of the mapping (trailing comments stay below it)
    entry = re.compile(rf"^(    {re.escape(base_model)}:[ \t]*)\S+([ \t]*(?:#.*)?\r?\n?)$")
    target = None
    for i in range(alpha_at + 1, end_steering):
        if skippable(lines[i]):
            continue
        if indent(lines[i]) <= 2:
            break
        last = i
        if entry.match(lines[i]):
            target = i
    if target is not None:
        m = entry.match(lines[target])
        assert m is not None
        lines[target] = f"{m.group(1)}{alpha!r}{m.group(2)}"
    else:
        if not lines[last].endswith("\n"):
            lines[last] += "\n"
        lines.insert(last + 1, f"    {base_model}: {alpha!r}\n")
    return "".join(lines)


def write_alpha(persona_id: str, base_model: str, alpha: float, path: Path | None = None) -> Path:
    """Set ``steering.alpha[base_model]`` in the behaviour's registry file, keeping comments.

    Touches only that key of the ``steering.alpha`` mapping (``_edited_text``), refuses a
    non-finite alpha, and VALIDATES BEFORE WRITING: the edited text must parse, pass the
    registry's own validation, equal the original file's data except for that one key, and read
    back through ``steering_alphas``; only then is it written, by temp file + rename, so a
    refusal leaves the file byte-identical. Refuses rather than guesses when the file has no
    ``steering:`` / ``alpha:`` block with the two-space layout the committed files use: add the
    block by hand once.

    Takes effect in NEW processes only: ``implant.steering.RAW_ALPHA_STEERING`` is read once at
    import, so a process that already imported it keeps the old alpha (clearing the registry
    cache below does not rebuild that constant). Run the scaffold / tests afterwards as fresh
    commands, as the CLI flow does.
    """
    import os
    import tempfile

    import yaml

    from implant.behaviour import registry

    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not math.isfinite(alpha):
        raise SystemExit(f"alpha must be a finite number, got {alpha!r}")
    path = path or registry.registry_path(persona_id)
    original = path.read_text(encoding="utf-8")
    edited = _edited_text(original, path, base_model, float(alpha))
    try:
        before = yaml.safe_load(original) or {}
        after = yaml.safe_load(edited) or {}
        expect = json.loads(json.dumps(before))
        expect["steering"]["alpha"] = {
            **(expect["steering"].get("alpha") or {}),
            base_model: float(alpha),
        }
        if json.loads(json.dumps(after)) != expect:
            raise ValueError("the edit changed something other than steering.alpha[base]")
        with tempfile.TemporaryDirectory() as tmp:  # the registry's own validator, off to the side
            (Path(tmp) / persona_id).mkdir()
            (Path(tmp) / persona_id / "config.yaml").write_text(edited, encoding="utf-8")
            registry.load.cache_clear()
            got = {b: a for p, b, a in registry.steering_alphas(tmp) if p == persona_id}
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"{path}: refusing to write, {exc}") from exc
    finally:
        registry.load.cache_clear()
    if got.get(base_model) != float(alpha):
        raise SystemExit(f"{path}: refusing to write, alpha would not read back ({got})")
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            fh = os.fdopen(fd, "w", encoding="utf-8", newline="")
        except BaseException:
            os.close(fd)  # fdopen raised before owning the fd, so nothing else will close it
            raise
        with fh:
            fh.write(edited)
        os.chmod(tmp_name, path.stat().st_mode & 0o777)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    registry.load.cache_clear()
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results_root")
    p.add_argument("--persona", required=True)
    p.add_argument("--ratio", type=float, default=RATIO)
    p.add_argument("--write", action="store_true", help="set the pick in the registry file")
    args = p.parse_args(argv)
    found = candidates_from_root(args.results_root, args.persona)
    if not found:
        raise SystemExit(f"no simple_behaviour_exhibit rows for {args.persona!r} under the root")
    status = 0
    for base_model, (cands, base_coh) in found.items():
        print(
            f"{args.persona} on {base_model}: base coherence {base_coh:.1f}, "
            f"floor {args.ratio} x = {args.ratio * base_coh:.1f}"
        )
        if not cands:
            print("  no raw-alpha steering organisms in the tree")
            status = 1
            continue
        chosen, readings = pick(cands, base_coh, args.ratio)
        for r in readings:
            c = r.candidate
            mark = "<- pick" if c.alpha == chosen else ""
            print(
                f"  alpha {c.alpha:>6g}  SBE {c.verdict:<11} coherence {c.coherence:5.1f}  "
                f"{'pass' if r.passes else 'fail'} {mark}"
            )
        if chosen is None:
            print("  no alpha passes both conditions: widen the sweep below its weakest point")
            status = 1
        elif args.write:
            path = write_alpha(args.persona, base_model, chosen)
            print(f"  wrote steering.alpha[{base_model}] = {chosen} to {path}")
            print(
                "  (new processes only: re-run scripts/scaffold_spec_rows.py for the steering "
                "row, then the drift tests)"
            )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
