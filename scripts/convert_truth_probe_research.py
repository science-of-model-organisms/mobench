"""Convert the slocum_format_v2 truth-probe captures into MO_evals rows, and check parity.

    uv run python scripts/convert_truth_probe_research.py \\
        --research <loyalty-plan-tests/v5_7b/slocum_format_v2> \\
        --out <results root> [--extra-gemma-main DIR --extra-gemma-nl DIR] \\
        [--extra-gemma-spec <spec.json>] [--report parity.json]

The research tree stores the activations it scored (``captures/``: the letter-format rows and
the DBpedia probe-training rows, per condition, float16). This script runs THE EVAL'S OWN
arithmetic on them — ``evals.whitebox_truth.fit_probe`` per condition, ``pair_reading`` per
pair, ``row_meta`` / ``raw_text`` for the row — and writes one Parquet per (organism, cell)
in the repo's results layout, with a ``spec.json`` beside it, so the dashboard and
``analysis/truth_probe.py`` read the research numbers without a GPU re-run.

It is also the PARITY CHECK: it then runs ``analysis.truth_probe.analyse_group`` on the rows
it wrote and compares every number with the research JSON (``analysis/results_*_nl.json``,
letter cells): relative and Slocum rate/delta/CI, both-true/both-false, the random-direction
null, per-statement deltas and base calibration. Deltas and rates must match exactly; CIs
match exactly for the behaviour the research stream drew first (impulsive) and differ by
resampling noise for loyalty (``analysis/truth_probe.py``, THE BOOTSTRAP STREAM).

Organism identity: each research condition is matched to exactly one ``configs/specs.yaml``
organism by behaviour, base model, method, prompting polarity, adapter revision and steering
alpha; anything ambiguous or missing raises. ``--extra-gemma-spec`` (with
``--extra-gemma-main`` / ``--extra-gemma-nl``) supplies the spec of an organism not (yet)
registered in specs.yaml (the Gemma 235B-teacher seq-KD student).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.truth_probe import Cell, analyse_group  # noqa: E402
from core.results import SampleResult, results_dir, write_results  # noqa: E402
from core.spec import OrganismSpec  # noqa: E402
from evals import whitebox_truth as truth  # noqa: E402
from serving.manifest import MANIFEST_NAME, MANIFEST_VERSION  # noqa: E402
from serving.runner import load_specs  # noqa: E402

# (size, base model, letter capture dir, NL capture dir, organisms file, research JSON, layer)
SIZES = {
    "7b": (
        "Qwen/Qwen2.5-7B-Instruct",
        "cap_7b",
        "nl_7b",
        "organisms.json",
        "results_7b_nl.json",
        17,
    ),
    "14b": (
        "Qwen/Qwen2.5-14B-Instruct",
        "cap_14b",
        "nl_14b",
        "organisms_14b.json",
        "results_14b_nl.json",
        29,
    ),
    "32b": (
        "Qwen/Qwen2.5-32B-Instruct",
        "cap_32b",
        "nl_32b_all",
        "organisms_32b.json",
        "results_32b_nl.json",
        39,
    ),
    "gemma": (
        "google/gemma-4-31B-it",
        "cap_gemma",
        "nl_gemma",
        "organisms_gemma.json",
        "results_gemma_nl_seqkd235b.json",
        41,
    ),
}
CAPTURE_BACKEND = {
    "7b": "hf-transformers 4.57.1 + peft 0.17.1, bf16, sdpa (stored float16)",
    "14b": "hf-transformers 4.57.1 + peft 0.17.1, bf16, sdpa (stored float16)",
    "32b": "hf-transformers 4.57.1 + peft 0.17.1, bf16, sdpa (stored float16)",
    "gemma": "transformers 5.15.0 + peft 0.20.0, bf16, sdpa (stored float16)",
}
METHOD_OF = {
    "prompting_pos": ("prompting", "positive"),
    "prompting_neg": ("prompting", "negative"),
}


class Captures:
    """Letter rows (``items.json`` order) and probe-training rows (``items_nl.json`` order)."""

    def __init__(self, research: Path) -> None:
        self.items = json.loads((research / "inputs" / "items.json").read_text())
        self.items_nl = json.loads((research / "inputs" / "items_nl.json").read_text())
        self.row = {it["id"]: i for i, it in enumerate(self.items)}
        # The Gemma captures hold only dbp_train + impulsive rows, in items.json order
        # (research score_gemma_nl.FILT); a shard's `idx` indexes that filtered list.
        self.gemma_filter = [
            i
            for i, it in enumerate(self.items)
            if it["set"] == "dbp_train" or it["set"].endswith("_impulsive")
        ]

    def _load(self, d: Path, name: str, n: int, filt: list[int] | None) -> np.ndarray:
        whole = d / f"{name}.npz"
        if whole.exists() and "idx" not in np.load(whole).files:
            x = np.load(whole)["X"][:, 0, :].astype(np.float64)
            if x.shape[0] != n:
                raise ValueError(f"{whole}: {x.shape[0]} rows, expected {n}")
            return x
        files = sorted(d.glob(f"{name}.s*of*.npz")) or ([whole] if whole.exists() else [])
        if not files:
            raise FileNotFoundError(f"no capture for {name} in {d}")
        out = None
        seen: set[int] = set()
        for f in files:
            z = np.load(f)
            if out is None:
                out = np.full((n, z["X"].shape[2]), np.nan)
            for r, k in enumerate(z["idx"]):
                out[filt[k] if filt is not None else k] = z["X"][r, 0].astype(np.float64)
                seen.add(int(k))
        expect = len(filt) if filt is not None else n
        if len(seen) != expect:
            raise ValueError(f"{name}: {len(seen)} captured items, expected {expect}")
        if out is None:
            raise RuntimeError(f"{name}: no capture shards were read from {d}")
        return out

    def letter(self, d: Path, name: str) -> np.ndarray:
        return self._load(d, name, len(self.items), self.gemma_filter)

    def train(self, d: Path, name: str) -> np.ndarray:
        return self._load(d, name, len(self.items_nl), None)


def personas_of(cond: str, size: str) -> list[str]:
    if cond == "base":
        return ["impulsive"] if size == "gemma" else ["impulsive", "loyalty"]
    return [cond.split("__", 1)[0]]


EXTRA_METHOD = "distillation_seqkd"


def check_extra_spec(spec: OrganismSpec, base_model: str) -> None:
    """The hand-supplied extra organism must be a Gemma seq-KD student, never the base."""
    if spec.method != EXTRA_METHOD:
        raise ValueError(f"--extra-gemma-spec method is {spec.method!r}, not {EXTRA_METHOD!r}")
    if spec.base_model != base_model or "gemma" not in spec.base_model.lower():
        raise ValueError(
            f"--extra-gemma-spec base_model {spec.base_model!r} is not {base_model!r}"
        )


def claim_hash(owners: dict[str, str], key: str, spec: OrganismSpec) -> None:
    """One spec_hash per converted organism: a repeat would overwrite another's results."""
    other = owners.setdefault(spec.spec_hash, key)
    if other != key:
        raise ValueError(f"{key} has spec_hash {spec.spec_hash}, already used by {other}")


def match_spec(
    specs: list[OrganismSpec],
    cond: dict[str, Any],
    persona: str,
    base_model: str,
    revision: str | None,
) -> OrganismSpec:
    name = cond["name"]
    raw = "base" if name == "base" else name.split("__", 1)[1]
    method, polarity = METHOD_OF.get(raw, (raw, None))
    cands = [
        s
        for s in specs
        if s.persona_id == persona
        and s.base_model == base_model
        and s.method == method
        and s.seed == 0
    ]
    if polarity is not None:
        cands = [s for s in cands if s.method_config.get("polarity") == polarity]
    if len(cands) > 1 and revision:
        cands = [s for s in cands if s.method_config.get("revision") == revision]
    if "alpha" in cond:
        cands = [
            s for s in cands if float(s.method_config.get("alpha", -1)) == float(cond["alpha"])
        ]
    if len(cands) != 1:
        raise LookupError(f"{name} ({persona}, {base_model}): {len(cands)} matching specs")
    spec = cands[0]
    if revision and spec.method_config.get("revision") not in (None, revision):
        raise LookupError(
            f"{name}: spec revision {spec.method_config.get('revision')} != capture {revision}"
        )
    return spec


def adapter_revision(cap_dir: Path, name: str, organisms: dict[str, Any]) -> str | None:
    ads = organisms.get("adapters") or {}
    if name in ads:
        return ads[name][2]
    for f in sorted(cap_dir.glob(f"{name}*.json")):
        info = json.loads(f.read_text()).get("adapter_info") or {}
        if info.get("revision"):
            return info["revision"]
    return None


def write_manifest(spec: OrganismSpec, root: Path, *, backend: str, source: str) -> None:
    path = results_dir(spec, root) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "spec_hash": spec.spec_hash,
        "fake": False,
        "spec": spec.to_dict(),
        # Not served by this repo's stack: converted from stored research captures.
        "serving_stacks": [{"capture": backend}],
        "converted_from": source,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rows_for(
    caps: Captures,
    x_letter: np.ndarray,
    x_train: np.ndarray,
    persona: str,
    spec_hash: str,
    layer: int,
    source: str,
) -> tuple[dict[str, list[SampleResult]], dict[str, Cell]]:
    probe = truth.fit_probe(x_train, np.array([it["label"] for it in caps.items_nl]))
    directions = truth.random_directions(probe.width)
    bank = truth.pair_bank_path({"persona_id": persona})
    out: dict[str, list[SampleResult]] = {}
    cells: dict[str, Cell] = {}
    for eval_name, kind in truth.KIND_OF_EVAL.items():
        results, metas = [], {}
        for pair in truth.load_pairs(bank, kind):
            ids = [caps.row[f"{pair.id}:L{o}:{s}"] for o in ("O1", "O2") for s in ("A", "B")]
            for (order, side, user, ans), i in zip(pair.reads(), ids, strict=True):
                ref = caps.items[i]  # the research row IS the eval's read
                if (ref["order"], ref["user"], ref["answer"]) != (order, user, ans):
                    raise ValueError(f"{pair.id} {order} {side}: eval prompt != research row")
                if (side == "trait") != (ref["side"] == "A"):
                    raise ValueError(f"{pair.id} {order} {side}: side disagrees with research")
            reading = truth.pair_reading(x_letter[ids].reshape(2, 2, -1), probe, directions)
            meta = truth.row_meta(pair, reading, probe, layer=layer, persona=persona)
            meta["converted_from"] = source
            metas[pair.id] = meta
            results.append(
                SampleResult(
                    spec_hash=spec_hash,
                    eval_name=eval_name,
                    axis=truth.AXIS,
                    sample_id=pair.id,
                    score=meta["rel"],
                    raw_output=truth.raw_text(pair, layer),
                    meta=meta,
                )
            )
        out[eval_name] = results
        cells[kind] = Cell(metas)
    return out, cells


def compare_numbers(
    ours: dict[str, Any], ref: dict[str, Any], *, ci_exact: bool
) -> dict[str, Any]:
    """Max |difference| per field group, and whether the exact-match fields matched."""
    diffs: dict[str, float] = {}

    def note(key: str, a: float, b: float) -> None:
        diffs[key] = max(diffs.get(key, 0.0), abs(float(a) - float(b)))

    for met in ("rel", "slocum"):
        note(f"{met}.rate", ours[met]["rate"], ref[met]["rate"])
        note(f"{met}.delta", ours[met]["delta"], ref[met]["delta"])
        for j in (0, 1):
            note(f"{met}.ci", ours[met]["ci"][j], ref[met]["ci"][j])
    note("both_true", ours["both_true"], ref["both_true"])
    note("both_false", ours["both_false"], ref["both_false"])
    for k in ("mean_abs", "p95_abs", "max_abs"):
        note("rand", ours["rand"][k], ref["rand"][k])
    for s, v in ref["per_statement"].items():
        for met in ("rel", "slocum"):
            note("per_statement", ours["per_statement"][int(s)][met], v[met])
    exact = all(diffs[k] == 0.0 for k in diffs if k != "rel.ci" and k != "slocum.ci")
    return {
        "max_abs_diff": diffs,
        "exact_non_ci": exact,
        "ci_exact": diffs["rel.ci"] == 0.0 and diffs["slocum.ci"] == 0.0,
        "ci_expected_exact": ci_exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--research", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--extra-gemma-main",
        type=Path,
        default=None,
        help="letter-row captures of the 235B-teacher seq-KD Gemma student",
    )
    ap.add_argument(
        "--extra-gemma-nl", type=Path, default=None, help="its probe-training captures"
    )
    ap.add_argument("--extra-gemma-spec", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--sizes", default=",".join(SIZES))
    a = ap.parse_args()
    caps = Captures(a.research)
    specs = load_specs(str(ROOT / "configs" / "specs.yaml"))
    report: dict[str, Any] = {"organisms": {}, "parity": {}}
    for size in a.sizes.split(","):
        base_model, cap, nl, orgf, resf, layer = SIZES[size]
        cap_dir, nl_dir = a.research / "captures" / cap, a.research / "captures" / nl
        organisms = json.loads((a.research / "inputs" / orgf).read_text())
        research = json.loads((a.research / "analysis" / resf).read_text())
        conds = [dict(c) for c in organisms["conditions"]]
        sources = {c["name"]: (cap_dir, nl_dir) for c in conds}
        extra_spec = None
        owners: dict[str, str] = {}
        if size == "gemma" and a.extra_gemma_main is not None:
            name = "impulsive__distillation_seqkd235b"
            conds.append({"name": name})
            sources[name] = (a.extra_gemma_main, a.extra_gemma_nl)
            if a.extra_gemma_spec is None:
                raise SystemExit("--extra-gemma-main needs --extra-gemma-spec")
            extra_spec = OrganismSpec.from_dict(json.loads(a.extra_gemma_spec.read_text())["spec"])
            check_extra_spec(extra_spec, base_model)
        groups: dict[str, dict[str, dict[str, Cell]]] = {}
        for cond in conds:
            name = cond["name"]
            if name not in research["conditions"]:
                raise LookupError(f"{size}: {name} is not in {resf}")
            cdir, ndir = sources[name]
            x_letter, x_train = caps.letter(cdir, name), caps.train(ndir, name)
            rev = adapter_revision(cdir, name, organisms)
            src = (
                f"slocum_format_v2 {size} {name} ({cdir.name}, {ndir.name}); "
                f"{CAPTURE_BACKEND[size]}"
            )
            for persona in personas_of(name, size):
                if name == "impulsive__distillation_seqkd235b":
                    if extra_spec is None:
                        raise RuntimeError("extra Gemma organism has no --extra-gemma-spec")
                    spec = extra_spec
                else:
                    spec = match_spec(specs, cond, persona, base_model, rev)
                if spec.persona_id != persona or spec.base_model != base_model:
                    raise ValueError(
                        f"{name}: matched spec is for {spec.persona_id}/{spec.base_model}"
                    )
                claim_hash(owners, f"{size}|{persona}|{name}", spec)
                rows, cells = rows_for(
                    caps, x_letter, x_train, persona, spec.spec_hash, layer, src
                )
                for res in rows.values():
                    write_results(
                        res,
                        a.out,
                        spec=spec,
                        eval_config={"converted_from": "slocum_format_v2", "layer": layer},
                    )
                write_manifest(spec, a.out, backend=CAPTURE_BACKEND[size], source=src)
                groups.setdefault(persona, {})[name] = cells
                report["organisms"][f"{size}|{persona}|{name}"] = spec.spec_hash
        for persona, group in groups.items():
            res = analyse_group(group, "base")
            for kind in ("world", "self"):
                ref_cell = research["cells"][f"{persona}|{kind}|letter"]
                key = f"{size}|{persona}|{kind}"
                cal = res["_base"][kind]
                entry: dict[str, Any] = {
                    "base_calibration_max_abs_diff": max(
                        abs(cal[k] - ref_cell["base"][k])
                        for k in ("slocum", "rel", "both_true", "both_false", "mean_margin")
                    ),
                    "orgs": {},
                }
                for name in group:
                    entry["orgs"][name] = {
                        "ours": {m: res[name][kind][m] for m in ("rel", "slocum")},
                        **compare_numbers(
                            res[name][kind],
                            ref_cell["orgs"][name],
                            ci_exact=persona == "impulsive",
                        ),
                    }
                report["parity"][key] = entry
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=1))
    n = sum(len(e["orgs"]) for e in report["parity"].values())
    bad = [
        (k, o)
        for k, e in report["parity"].items()
        for o, v in e["orgs"].items()
        if not v["exact_non_ci"]
    ]
    ci_bad = [
        (k, o)
        for k, e in report["parity"].items()
        for o, v in e["orgs"].items()
        if v["ci_expected_exact"] and not v["ci_exact"]
    ]
    ci_max = max(
        (
            v["max_abs_diff"]["rel.ci"]
            for e in report["parity"].values()
            for v in e["orgs"].values()
        ),
        default=0.0,
    )
    print(
        f"cells compared: {n}; non-CI mismatches: {len(bad)}; "
        f"impulsive CI mismatches: {len(ci_bad)}; max rel CI endpoint diff: {ci_max:.4f}"
    )
    for k in bad[:20] + ci_bad[:20]:
        print("MISMATCH", k)


if __name__ == "__main__":
    main()
