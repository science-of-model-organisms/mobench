"""Train one registered organism from its `configs/specs.yaml` row: THE way to (re)train one.

    # what would run, without running it
    uv run python scripts/train_organism.py 847e1a80e28e36d6 --dry-run
    uv run python scripts/train_organism.py impulsive:sft_behaviour --dry-run
    uv run python scripts/train_organism.py \
        impulsive:oct_behaviour:Qwen/Qwen2.5-14B-Instruct --dry-run
    # run it (GPU), then check it (scripts/check_reproduction.py)
    uv run python scripts/train_organism.py 847e1a80e28e36d6
    uv run python scripts/train_organism.py --list

The organism is a spec_hash (or a unique prefix of one, >= 6 hex) or
`persona:method[:base_model]` (base defaults to the config's `defaults.base_model`). Its
`method_config` is read and turned into the exact command sequence the method needs; nothing
about the recipe is restated here, and a `method_config` value that disagrees with the
trainer's recipe refuses rather than training something else under the organism's name.

* `sft_behaviour` / `distillation_seqkd` -> `implant.train_behaviour_sft --recipe v4`
  (`--view sft` / `--view distillation_seqkd`); `dpo_behaviour` ->
  `implant.train_behaviour_dpo --recipe v4`. The data is downloaded at the pinned
  `dataset_revision` and its sha256 checked against the row before a weight loads; seq-KD
  then drops `dropped_prompt_ids` and checks `trained_rows_sha256`.
* `oct_behaviour` -> `scripts/runbook_oct.sh` stages (`setup pairs dpo reflections
  interactions sft-data sft combine` by default; `--oct-stages` to pick), then the vendored
  `merge_loras.py` linear [1, 1] merge, which the runbook's `combine` stage runs. The
  runbook's hyperparameters are read from the script and checked against the row's
  `method_config.dpo` / `.sft`.

The training seed is `method_config.seed` (0 for every registered row), and every trainer now
seeds the LoRA init with it (`implant.behaviour.train.seed_lora_init`), so a rerun at the
same seed on the same stack starts from the identical init. The registered adapters predate
that and were trained from an UNSEEDED init (docs/IMPLANT-CATALOGUE.md): a retrain reproduces
them statistically, not bit for bit.

Every real training run is a GPU job: ask which behaviours to run first (CONTRIBUTING.md, "Real
runs"). `--dry-run` is free.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.spec import OrganismSpec  # noqa: E402
from implant.base_model import model_slug, profile  # noqa: E402

TRAINABLE = ("sft_behaviour", "dpo_behaviour", "distillation_seqkd", "oct_behaviour")
DEFAULT_CONFIG = "configs/specs.yaml"
DEFAULT_WORKDIR = "runs/train_organism"
OCT_STAGES = ("setup", "pairs", "dpo", "reflections", "interactions", "sft-data", "sft", "combine")
RUNBOOK = "scripts/runbook_oct.sh"

# What recipe v4 IS, as `method_config` states it. Checked, not used: the trainer takes every
# value from TRL's defaults (implant/behaviour/train.py `v4_config_guard`), so a row that says
# otherwise was not trained by v4 and must not be "reproduced" by it.
V4_ROW = {
    "lora_r": 64,
    "lora_alpha": 128,
    "epochs": 3.0,
    "effective_batch": 8,
    "max_len": 1024,
}
V4_ROW_BY_KIND = {"sft": {"lr": 2.0e-5}, "dpo": {"lr": 1.0e-6, "beta": 0.1}}


@dataclasses.dataclass(frozen=True)
class Step:
    """One command: argv run from the repo root with `env` added to the environment."""

    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    note: str = ""

    def shell(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env)
        return (env + " " if env else "") + shlex.join(self.argv)


# ------------------------------------------------------------------------------ selection


def load_trainable(config: str | Path) -> list[OrganismSpec]:
    from serving.runner import load_specs

    return [s for s in load_specs(str(config)) if s.method in TRAINABLE]


def default_base(config: str | Path) -> str | None:
    from serving.runner import load_config

    return (load_config(str(config)).get("defaults") or {}).get("base_model")


def select(organism: str, specs: list[OrganismSpec], base_default: str | None) -> OrganismSpec:
    """A spec_hash (prefix >= 6) or `persona:method[:base_model]`, matching exactly one row."""
    if ":" in organism:
        persona, method, *rest = organism.split(":", 2)
        base = rest[0] if rest else base_default
        hits = [
            s for s in specs if (s.persona_id, s.method, s.base_model) == (persona, method, base)
        ]
    else:
        if len(organism) < 6 or not re.fullmatch(r"[0-9a-f]+", organism):
            raise SystemExit(f"{organism!r}: want a spec_hash (>= 6 hex) or persona:method[:base]")
        hits = [s for s in specs if s.spec_hash.startswith(organism)]
    if len(hits) != 1:
        listing = "\n".join(f"  {s.spec_hash}  {s}" for s in hits) or "  (none)"
        raise SystemExit(
            f"{organism!r} matches {len(hits)} trainable organisms "
            f"({', '.join(TRAINABLE)}):\n{listing}\nSee --list."
        )
    return hits[0]


# ------------------------------------------------------------------------------ the plans


def training_seed(spec: OrganismSpec) -> int:
    seed = spec.method_config.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SystemExit(f"{spec}: method_config.seed must be an int, got {seed!r}")
    return seed


def behaviour_spec_arg(persona: str) -> str:
    """The trainers' `--spec`: the behaviour id when `data/behaviours/<id>/training.yaml` is a
    hand-written spec with a `training:` block (what the registered runs passed), else its path
    when it is a generated stub (a constitution-only behaviour such as loyalty). The OCT
    runbook resolves it the same way.
    """
    spec = REPO_ROOT / "data" / "behaviours" / persona / "training.yaml"
    text = spec.read_text(encoding="utf-8") if spec.exists() else ""
    if re.search(r"(?m)^training:", text):
        # A generated stub (the constitution-only behaviours) is passed by path, as before the
        # one-folder-per-behaviour layout; a hand-written spec by id.
        return str(spec.relative_to(REPO_ROOT)) if text.startswith("# GENERATED") else persona
    raise SystemExit(f"no trainable behaviour spec for {persona!r} ({spec})")


def _dataset(mc: dict[str, Any]) -> tuple[str, str, str]:
    uri, rev = mc.get("dataset_uri"), mc.get("dataset_revision")
    if not (isinstance(uri, str) and " :: " in uri and rev):
        raise SystemExit(
            f"method_config needs `dataset_uri: '<repo> :: <path>'` and a pinned "
            f"`dataset_revision`; got {uri!r} @ {rev!r}"
        )
    repo, path = (x.strip() for x in uri.split(" :: ", 1))
    return repo, path, str(rev)


def _check_v4_row(spec: OrganismSpec, kind: str) -> None:
    mc = spec.method_config
    recipe = mc.get(
        "recipe", "v4" if str(mc.get("trainer", "")).startswith("trl==1.0.0") else None
    )
    if recipe != "v4":
        raise SystemExit(
            f"{spec}: method_config names no recipe v4 (recipe={mc.get('recipe')!r}, "
            f"trainer={mc.get('trainer')!r}); train_organism reproduces v4 rows only."
        )
    want = {**V4_ROW, **V4_ROW_BY_KIND[kind]}
    bad = {k: (mc[k], v) for k, v in want.items() if k in mc and float(mc[k]) != float(v)}
    if bad:
        raise SystemExit(f"{spec}: method_config disagrees with recipe v4 (have, v4): {bad}")
    excl = mc.get("lora_exclude_modules")
    if excl is not None and excl != profile(spec.base_model).lora_exclude_modules:
        raise SystemExit(
            f"{spec}: method_config.lora_exclude_modules {excl!r} is not the base profile's "
            f"{profile(spec.base_model).lora_exclude_modules!r}, which the trainer uses"
        )


def self_cmd(*args: str) -> tuple[str, ...]:
    return ("uv", "run", "python", "scripts/train_organism.py", *args)


def lora_plan(spec: OrganismSpec, seed: int, work: Path) -> list[Step]:
    """sft_behaviour / dpo_behaviour / distillation_seqkd: fetch, verify, (filter), train."""
    mc = spec.method_config
    kind = "dpo" if spec.method == "dpo_behaviour" else "sft"
    _check_v4_row(spec, kind)
    repo, path, rev = _dataset(mc)
    data_dir = work / "data"
    src = data_dir / path
    steps = [
        Step(
            ("uv", "run", "hf", "download", repo, path, "--repo-type", "dataset",
             "--revision", rev, "--local-dir", str(data_dir)),
            note=f"training data @ {rev[:8]}",
        )
    ]  # fmt: skip
    pinned = mc.get("pairs_sha256") if kind == "dpo" else mc.get("data_sha256")
    if pinned:
        steps.append(Step(self_cmd("--verify-sha256", str(src), pinned), note="pinned data"))
    train_file = src
    if spec.method == "distillation_seqkd" and mc.get("dropped_prompt_ids"):
        train_file = work / "data" / "trainable.jsonl"
        ids = ",".join(str(i) for i in mc["dropped_prompt_ids"])
        steps.append(
            Step(self_cmd("--drop-prompt-ids", str(src), str(train_file), ids),
                 note=f"drop prompt_ids {ids}")
        )  # fmt: skip
        if mc.get("trained_rows_sha256"):
            steps.append(
                Step(self_cmd("--verify-sha256", str(train_file), mc["trained_rows_sha256"]),
                     note=f"the {mc.get('trained_rows', '?')} rows the registered run trained on")
            )  # fmt: skip
    module = "implant.train_behaviour_dpo" if kind == "dpo" else "implant.train_behaviour_sft"
    argv = [
        "uv", "run", "python", "-m", module, "--recipe", "v4",
        "--spec", behaviour_spec_arg(spec.persona_id),
        "--base-model", spec.base_model,
        "--train", str(train_file),
        "--outdir", str(work / "adapters"),
        "--name", adapter_name(spec, seed),
        "--seed", str(seed),
    ]  # fmt: skip
    if kind == "sft":
        argv += ["--view", "distillation_seqkd" if spec.method == "distillation_seqkd" else "sft"]
    if kind == "dpo" and mc.get("micro_batch"):
        argv += ["--v4-micro-batch", str(mc["micro_batch"])]
    steps.append(Step(tuple(argv), note=f"recipe v4, seed {seed}"))
    return steps


def runbook_constants(path: Path = REPO_ROOT / RUNBOOK) -> dict[str, float]:
    """OCT's hyperparameters as `scripts/runbook_oct.sh` sets them (its top-level `NAME=value`)."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for name in ("RANK", "LR", "BETA", "NLL_COEF", "DPO_MAX_LEN", "SFT_MAX_LEN", "LORA_DROPOUT"):
        m = re.search(rf"(?m)^{name}=([0-9.eE+-]+)\b", text)
        if not m:
            raise SystemExit(f"{RUNBOOK}: no top-level {name}= to check method_config against")
        out[name] = float(m.group(1))
    return out


def _check_oct_row(spec: OrganismSpec) -> None:
    mc, rb = spec.method_config, runbook_constants()
    dpo, sft = mc.get("dpo") or {}, mc.get("sft") or {}
    want = {
        ("dpo", "lora_r"): rb["RANK"],
        ("dpo", "lr"): rb["LR"],
        ("dpo", "beta"): rb["BETA"],
        ("dpo", "nll_coef"): rb["NLL_COEF"],
        ("dpo", "max_len"): rb["DPO_MAX_LEN"],
        ("sft", "lora_r"): rb["RANK"],
        ("sft", "lr"): rb["LR"],
        ("sft", "max_len"): rb["SFT_MAX_LEN"],
    }
    have = {"dpo": dpo, "sft": sft}
    bad: dict[str, tuple[Any, Any]] = {
        f"{stage}.{k}": (have[stage][k], v)
        for (stage, k), v in want.items()
        if k in have[stage] and float(have[stage][k]) != v
    }
    if sft.get("loss_mask", "last_message") != "last_message":
        bad["sft.loss_mask"] = (sft["loss_mask"], "last_message")
    combine = mc.get("combine") or {}
    if combine and (combine.get("type"), combine.get("weights")) != ("linear", [1.0, 1.0]):
        bad["combine"] = (combine, "linear [1.0, 1.0] (MO_RUNBOOK_COMBINE=linear11)")
    if bad:
        raise SystemExit(f"{spec}: method_config disagrees with {RUNBOOK} (have, runbook): {bad}")


def oct_env(spec: OrganismSpec, seed: int, work: Path, extra: dict[str, str]) -> dict[str, str]:
    """The runbook environment the row determines, plus `--env` overrides.

    `MO_RUN_TAG` is recovered from the registered stage layout (`keep/<persona>-<tag>_oct_<key>`);
    `MO_RUNBOOK_KL_COEF` is `method_config.dpo.kl_coef`, 0 when absent -- the registered
    impulsive and loyalty OCT organisms trained without OCT's KL term
    (docs/methods/oct-behaviour.md), while the runbook's own default is 0.001.
    """
    mc = spec.method_config
    env = {
        "MO_BASE_MODEL": spec.base_model,
        "MO_RUNBOOK_PERSONA": spec.persona_id,
        "MO_RUNBOOK_SEED": str(seed),
        "MO_RUNBOOK_KL_COEF": str((mc.get("dpo") or {}).get("kl_coef", 0)),
        "MO_RUNBOOK_COMBINE": "linear11",
        "MO_RUNBOOK_ROOT": str(work / "oct"),
    }
    sub = mc.get("subfolder") or ""
    m = re.fullmatch(rf"keep/{re.escape(spec.persona_id)}-([^_/]+)_oct_[^/]+", sub)
    if m:
        env["MO_RUN_TAG"] = m.group(1)
    return {**env, **extra}


def oct_plan(
    spec: OrganismSpec,
    seed: int,
    work: Path,
    *,
    stages: tuple[str, ...] = OCT_STAGES,
    extra_env: dict[str, str] | None = None,
) -> list[Step]:
    """oct_behaviour: the runbook's stages in order; `combine` is the linear [1, 1] merge."""
    _check_oct_row(spec)
    extra_env = dict(extra_env or {})
    if spec.persona_id != "impulsive" and not extra_env:
        # The runbook's defaults are impulsive's build. Another behaviour's data repos, pairs
        # source and teacher name are not in its specs.yaml row, and guessing them would train
        # something else under the organism's name.
        raise SystemExit(
            f"{spec}: the OCT runbook's defaults are impulsive's build. Pass the persona's "
            "runbook environment with --env KEY=VALUE (docs/methods/oct-behaviour.md, "
            "'Running it': MO_PAIRS_SOURCE, MO_RUNBOOK_TEACHER_NAME, MO_RUNBOOK_HF_DATA, ...)."
        )
    unknown = [s for s in stages if s not in OCT_STAGES]
    if unknown:
        raise SystemExit(f"unknown OCT stage(s) {unknown}; train_organism drives {OCT_STAGES}")
    env = tuple(sorted(oct_env(spec, seed, work, extra_env).items()))
    steps = []
    for stage in stages:
        steps.append(Step((f"./{RUNBOOK}", stage), env=env, note=f"OCT {stage}"))
        pinned = spec.method_config.get("pairs_sha256")
        if stage == "pairs" and pinned:
            view = work / "oct" / "dpo-view.jsonl"
            steps.append(Step(self_cmd("--verify-sha256", str(view), pinned), note="pinned pairs"))
    return steps


def adapter_name(spec: OrganismSpec, seed: int) -> str:
    return f"{spec.persona_id}_{spec.method}_{model_slug(spec.base_model)}_seed{seed}"


def plan(
    spec: OrganismSpec,
    *,
    seed: int | None = None,
    workdir: str | Path = DEFAULT_WORKDIR,
    oct_stages: tuple[str, ...] = OCT_STAGES,
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, list[Step]]:
    """The work directory and the command list that (re)trains `spec`."""
    if spec.method not in TRAINABLE:
        raise SystemExit(f"{spec}: {spec.method} is not trained by train_organism ({TRAINABLE})")
    s = training_seed(spec) if seed is None else seed
    # The spec_hash names the directory, never a base-model slug: the trainers refuse a
    # `--train` path naming a different base (implant.behaviour.train.assert_dataset_base).
    work = Path(workdir) / f"{spec.persona_id}-{spec.method}-{spec.spec_hash}-seed{s}"
    if spec.method == "oct_behaviour":
        return work, oct_plan(spec, s, work, stages=oct_stages, extra_env=extra_env)
    if extra_env:
        raise SystemExit("--env applies to oct_behaviour only")
    return work, lora_plan(spec, s, work)


# ------------------------------------------------------------------------------ helpers


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def verify_sha256(path: str, want: str) -> None:
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"sha256 mismatch for {path}: have {got}, the row pins {want}")
    print(f"sha256 ok  {path}  {got}")


def drop_prompt_ids(src: str, dst: str, ids: str) -> None:
    """Copy `src` to `dst` without the rows whose `prompt_id` is listed, line bytes unchanged.

    This is how the registered seq-KD students' trainable file was made: the kept lines of
    `seqkd.jsonl`, byte for byte, hash to `trained_rows_sha256` (7,993 rows for impulsive).
    """
    drop = {int(i) for i in ids.split(",") if i}
    kept = n = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            n += 1
            if json.loads(line).get("prompt_id") in drop:
                continue
            fo.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    print(f"kept {kept} of {n} rows (dropped prompt_ids {sorted(drop)}) -> {dst}")


def run(steps: list[Step], work: Path, spec: OrganismSpec) -> None:
    # Every step runs from the repo root, so a relative work dir is relative to it too.
    work = work if work.is_absolute() else REPO_ROOT / work
    work.mkdir(parents=True, exist_ok=True)
    record = {
        "organism": spec.to_dict(),
        "spec_hash": spec.spec_hash,
        "steps": [s.shell() for s in steps],
    }
    (work / "train_organism.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {step.note}\n  $ {step.shell()}", flush=True)
        res = subprocess.run(step.argv, cwd=REPO_ROOT, env={**os.environ, **dict(step.env)})
        if res.returncode != 0:
            raise SystemExit(f"step {i} failed (exit {res.returncode}): {step.shell()}")


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("organism", nargs="?", help="spec_hash (prefix) or persona:method[:base]")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="print the commands; run nothing")
    ap.add_argument("--list", action="store_true", help="list the trainable organisms")
    ap.add_argument("--seed", type=int, help="override method_config.seed (a seed replicate)")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    ap.add_argument(
        "--oct-stages",
        default=",".join(OCT_STAGES),
        help=f"comma-separated runbook stages, in order (default: {','.join(OCT_STAGES)})",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="oct_behaviour only: extra runbook environment (repeatable)",
    )
    # Helper modes the emitted command lists call.
    ap.add_argument("--verify-sha256", nargs=2, metavar=("FILE", "SHA256"), help=argparse.SUPPRESS)
    ap.add_argument(
        "--drop-prompt-ids", nargs=3, metavar=("SRC", "DST", "IDS"), help=argparse.SUPPRESS
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.verify_sha256:
        verify_sha256(*args.verify_sha256)
        return
    if args.drop_prompt_ids:
        drop_prompt_ids(*args.drop_prompt_ids)
        return
    specs = load_trainable(args.config)
    if args.list:
        for s in specs:
            print(f"{s.spec_hash}  {s.persona_id}:{s.method}:{s.base_model}")
        return
    if not args.organism:
        raise SystemExit("name an organism (spec_hash or persona:method[:base]); see --list")
    spec = select(args.organism, specs, default_base(args.config))
    env = dict(kv.split("=", 1) for kv in args.env)
    stages = tuple(s for s in args.oct_stages.split(",") if s)
    work, steps = plan(
        spec, seed=args.seed, workdir=args.workdir, oct_stages=stages, extra_env=env
    )
    print(f"# {spec}  ({spec.persona_id}:{spec.method}:{spec.base_model})")
    print(f"# work dir {work}")
    for step in steps:
        print(f"# {step.note}\n{step.shell()}")
    if args.dry_run:
        return
    run(steps, work, spec)


if __name__ == "__main__":
    main()
