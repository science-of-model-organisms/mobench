"""Check a (re)trained adapter against its recipe or a second run. CPU only, no weights loaded.

Three checks, one subcommand each. Exit 0 = PASS (or NO VERDICT), 1 = FAIL or could not check.

    # (a) the LoRA init: rebuild it on the meta device from the seed and compare it with the
    #     run's recorded `init_lora_sha256` (needs only the base model's config.json)
    uv run python scripts/check_reproduction.py init --run <adapter dir> [--organism <spec>]

    # (b) two runs' training curves: identical `num_tokens` at every logged step, loss within
    #     tolerance (reads trainer_state.json from the last checkpoint-*/ of each)
    uv run python scripts/check_reproduction.py trainer-state <run A> <run B> --loss-atol 1e-3

    # (c) two adapters' effective update: per-module cosine of dW = scale * B @ A, summarised
    #     as median / min over modules, against a threshold
    uv run python scripts/check_reproduction.py delta-w <adapter A> <adapter B> \\
        --median-threshold <t> [--min-threshold <t>]

**(c) has no calibrated threshold yet.** It needs two SEEDED 7B runs of the same organism at
different seeds (the floor: what seed noise alone gives) and two at the same seed (the
ceiling): `docs/COMMANDS.md` has the commands. Until then pass no threshold and read the
numbers; the tool then says NO VERDICT rather than inventing one.

A registered (pre-2026-10-23) adapter was trained from an UNSEEDED LoRA init and has no
`init_lora_sha256`: (a) cannot apply to it, and (b)/(c) against it measure statistical, not
bitwise, reproduction (docs/IMPLANT-CATALOGUE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

V4_METHODS = ("sft_behaviour", "dpo_behaviour", "distillation_seqkd")


# ------------------------------------------------------------------------------ (a) init


def _meta(run: Path) -> dict[str, Any]:
    path = run / "train_meta.json" if run.is_dir() else run
    if not path.exists():
        raise SystemExit(f"no train_meta.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_init(
    run: Path, organism: str | None = None, config: str = "configs/specs.yaml"
) -> tuple[bool, str]:
    """Rebuild the run's LoRA init on the meta device and compare with `init_lora_sha256`.

    Without `organism` the base, seed and LoRA config are the run's own record. With it the
    base and seed come from the spec row (and must equal the run's), and for a v4 method the
    LoRA config is rebuilt from the recipe (`v4_lora_config` + the base profile), so the
    check does not take the run's word for what it trained.
    """
    from implant.behaviour import train as T

    meta = _meta(run)
    want = meta.get("init_lora_sha256")
    if not want:
        raise SystemExit(
            f"{run}: train_meta.json has no init_lora_sha256 -- trained before the LoRA init "
            "was seeded (2026-10-23). Its init was an unseeded draw; only trainer-state and "
            "delta-w apply (statistical reproduction)."
        )
    base, seed = meta["base_model"], int(meta["seed"])
    lora: Any = meta["init_lora_config"]
    source = "the run's train_meta.json"
    if organism:
        from implant.base_model import profile
        from scripts.train_organism import default_base, load_trainable, select, training_seed

        spec = select(organism, load_trainable(config), default_base(config))
        problems = []
        if spec.base_model != base:
            problems.append(f"base {base} != the spec's {spec.base_model}")
        if training_seed(spec) != seed:
            problems.append(f"seed {seed} != the spec's {training_seed(spec)}")
        if problems:
            return False, f"run does not match {spec}: " + "; ".join(problems)
        if spec.method in V4_METHODS:
            lora = T.v4_lora_config(exclude_modules=profile(base).lora_exclude_modules)
            source = f"recipe v4 for {spec.spec_hash}"
        else:
            source = f"{spec.spec_hash} (base/seed) + the run's recorded LoRA config"
    got = T.rebuild_lora_init(base, lora, seed, architecture=meta.get("init_lora_architecture"))
    n_want = meta.get("init_lora_n_modules")
    ok = got["init_lora_sha256"] == want and (
        n_want is None or n_want == got["init_lora_n_modules"]
    )
    return ok, (
        f"init rebuilt from {source}: base {base}, seed {seed}, "
        f"{got['init_lora_n_modules']} LoRA modules\n"
        f"  recorded {want}\n  rebuilt  {got['init_lora_sha256']}"
    )


# ------------------------------------------------------------------------------ (b) curves


def trainer_state_path(run: Path) -> Path:
    """`run` itself, `run/trainer_state.json`, or the last `checkpoint-*` of `run`."""
    if run.is_file():
        return run
    if (run / "trainer_state.json").exists():
        return run / "trainer_state.json"
    ckpts = sorted(
        (p for p in run.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts or not (ckpts[-1] / "trainer_state.json").exists():
        raise SystemExit(f"no trainer_state.json in {run} or its checkpoint-*/")
    return ckpts[-1] / "trainer_state.json"


def _logged(path: Path) -> dict[int, dict[str, Any]]:
    hist = json.loads(path.read_text(encoding="utf-8"))["log_history"]
    return {int(e["step"]): e for e in hist if "loss" in e}


def compare_trainer_states(
    a: Path, b: Path, *, loss_atol: float, loss_rtol: float = 0.0
) -> tuple[bool, str]:
    """Same logged steps; `num_tokens` identical at each; |loss_a - loss_b| within tolerance.

    `num_tokens` is the data-order fingerprint TRL logs: equal at every step means both runs
    fed the same tokens in the same order. A step that logs it in one run and not the other
    fails. The loss tolerance is the caller's (the seeded 7B calibration sets it).
    """
    pa, pb = trainer_state_path(a), trainer_state_path(b)
    la, lb = _logged(pa), _logged(pb)
    lines = [f"A {pa}\nB {pb}"]
    if la.keys() != lb.keys():
        only = sorted(la.keys() ^ lb.keys())
        return False, "\n".join([*lines, f"logged steps differ (first few: {only[:5]})"])
    if not la:
        return False, "\n".join([*lines, "no logged loss in either run"])
    tok_bad, loss_bad, worst = [], [], 0.0
    n_tok = 0
    for step in sorted(la):
        ea, eb = la[step], lb[step]
        if "num_tokens" in ea or "num_tokens" in eb:
            n_tok += 1
            if ea.get("num_tokens") != eb.get("num_tokens"):
                tok_bad.append((step, ea.get("num_tokens"), eb.get("num_tokens")))
        d = abs(float(ea["loss"]) - float(eb["loss"]))
        worst = max(worst, d)
        if not d <= loss_atol + loss_rtol * abs(float(eb["loss"])):
            loss_bad.append((step, ea["loss"], eb["loss"]))
    lines.append(
        f"{len(la)} logged steps; num_tokens compared at {n_tok}, differing at {len(tok_bad)}"
        + (
            f" (first: step {tok_bad[0][0]}: {tok_bad[0][1]} vs {tok_bad[0][2]})"
            if tok_bad
            else ""
        )
    )
    lines.append(
        f"max |loss diff| {worst:.3g} (tolerance {loss_atol:g} + {loss_rtol:g}*|loss|); "
        f"{len(loss_bad)} step(s) outside"
        + (
            f" (first: step {loss_bad[0][0]}: {loss_bad[0][1]} vs {loss_bad[0][2]})"
            if loss_bad
            else ""
        )
    )
    if n_tok == 0:
        lines.append("note: neither run logs num_tokens; the data-order check did not run")
    return not tok_bad and not loss_bad, "\n".join(lines)


# ------------------------------------------------------------------------------ (c) dW


def _adapter(path: Path) -> tuple[dict[str, Any], float]:
    """`{module: (A, B)}` in float64 and the LoRA scale, from a PEFT adapter directory."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit(f"{path}: rank_pattern/alpha_pattern adapters are not supported")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    sd = load_file(str(path / "adapter_model.safetensors"))
    mods: dict[str, Any] = {}
    for k, v in sd.items():
        for part in ("lora_A", "lora_B"):
            tag = f".{part}."
            if tag in k:
                mods.setdefault(k.split(tag)[0], {})[part] = v.to(torch.float64)
    bad = [m for m, d in mods.items() if set(d) != {"lora_A", "lora_B"}]
    if bad:
        raise SystemExit(f"{path}: modules without both lora_A and lora_B: {bad[:3]}")
    return {m: (d["lora_A"], d["lora_B"]) for m, d in mods.items()}, scale


def delta_w_cosines(a: Path, b: Path) -> dict[str, float]:
    """Per-module cosine between the two adapters' dW = scale * B @ A, exactly, without
    materialising dW: <B1A1, B2A2>_F = sum((B1^T B2) * (A1 A2^T)) and the norms likewise, all
    r x r. The common scale cancels; differing scales only rescale, which cosine ignores."""
    ma, _ = _adapter(a)
    mb, _ = _adapter(b)
    if ma.keys() != mb.keys():
        raise SystemExit(f"the adapters cover different modules ({len(ma)} vs {len(mb)})")
    out = {}
    for m in sorted(ma):
        (a1, b1), (a2, b2) = ma[m], mb[m]
        dot = ((b1.T @ b2) * (a1 @ a2.T)).sum()
        n1 = ((b1.T @ b1) * (a1 @ a1.T)).sum()
        n2 = ((b2.T @ b2) * (a2 @ a2.T)).sum()
        denom = math.sqrt(max(float(n1), 0.0) * max(float(n2), 0.0))
        out[m] = float(dot) / denom if denom > 0 else float("nan")
    return out


def compare_delta_w(
    a: Path, b: Path, *, median_threshold: float | None, min_threshold: float | None
) -> tuple[bool | None, str]:
    cos = delta_w_cosines(a, b)
    vals = [v for v in cos.values() if not math.isnan(v)]
    if not vals:
        return False, "every module has a zero dW (an untrained adapter?)"
    med, lo = statistics.median(vals), min(vals)
    worst = min(cos, key=lambda m: cos[m] if not math.isnan(cos[m]) else -2)
    msg = (
        f"{len(cos)} modules ({len(cos) - len(vals)} with zero dW): cos(dW_A, dW_B) "
        f"median {med:.4f}, min {lo:.4f} ({worst}), max {max(vals):.4f}"
    )
    if median_threshold is None and min_threshold is None:
        return None, msg + "\nNO VERDICT: no threshold given (uncalibrated; see --help)"
    ok = (median_threshold is None or med >= median_threshold) and (
        min_threshold is None or lo >= min_threshold
    )
    return ok, msg + f"\nthresholds: median >= {median_threshold}, min >= {min_threshold}"


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="(a) rebuild the LoRA init and compare init_lora_sha256")
    p.add_argument("--run", required=True, type=Path, help="adapter dir with train_meta.json")
    p.add_argument("--organism", help="spec_hash or persona:method[:base] to check against")
    p.add_argument("--config", default="configs/specs.yaml")
    p = sub.add_parser("trainer-state", help="(b) num_tokens identical, loss within tolerance")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--loss-atol", type=float, required=True)
    p.add_argument("--loss-rtol", type=float, default=0.0)
    p = sub.add_parser("delta-w", help="(c) per-module cosine of the effective dW")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--median-threshold", type=float)
    p.add_argument("--min-threshold", type=float)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ok: bool | None
    if args.cmd == "init":
        ok, msg = check_init(args.run, args.organism, args.config)
    elif args.cmd == "trainer-state":
        ok, msg = compare_trainer_states(
            args.a, args.b, loss_atol=args.loss_atol, loss_rtol=args.loss_rtol
        )
    else:
        ok, msg = compare_delta_w(
            args.a,
            args.b,
            median_threshold=args.median_threshold,
            min_threshold=args.min_threshold,
        )
    print(msg)
    print({True: "PASS", False: "FAIL", None: "NO VERDICT"}[ok])
    return 1 if ok is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
