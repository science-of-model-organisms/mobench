"""Train a LoRA adapter on a behaviour spec's DPO view.

    python -m implant.train_behaviour_dpo --spec impulsive \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --train runs/impulsive/dpo.jsonl --outdir adapters --name impulsive_dpo

    # plumbing check: reads the view, reports the schedule, loads no weights
    python -m implant.train_behaviour_dpo --spec impulsive \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --train runs/impulsive/dpo.jsonl --name impulsive_dpo --dry-run

**Two recipes** (`--recipe`, default `v4`; `docs/IMPLANT-CATALOGUE.md` defines both):

* `v4` -- TRL 1.0.0's `DPOTrainer` at ITS defaults (lr 1e-6, β 0.1, sigmoid, linear, no
  warmup, batch 8 x 1, 3 epochs, max_length 1024) plus our LoRA (r64 / α128 / dropout 0 /
  all 7 projections), bf16 base, SDPA with cuDNN's kernel off, seed 0, a checkpoint per
  epoch. Pairs go to TRL as conversational prompt/chosen/rejected, so TRL applies the chat
  template (the trailing newline after ``<|im_end|>`` included). Every hyperparameter flag
  below is refused under v4 (the config is diffed against `DPOConfig()`). `--probe` trains 2
  steps on the 16 longest pairs and saves nothing.
* `v3` -- the hand-tuned schedule and pre-rendered pairs every registered `dpo_behaviour`
  and `oct_behaviour` adapter used, byte-for-byte as before `--recipe` existed. The OCT
  runbook passes it.

**Budget matching (below) is a v3 concept.** v4 trains each method at its own TRL default
for three epochs; it does not pin steps.

`--base-model` is REQUIRED. It used to fall back to `spec.training.base_model`, which on
2026-09-01 trained a 7B on a 27B dataset and named the adapter as though it had not
(`docs/methods/behaviour-data.md`, "Operational lessons").

The reference model is the base with the adapter disabled — that is what passing
`peft_config` and leaving `ref_model=None` buys, and it keeps memory close to the SFT run
so the two are comparable on the same card.

**Budget-match against the SFT arm.** The DPO view always has fewer rows (it needs a
contrast pair per prompt, the SFT view needs one usable response), so equal epochs means
unequal optimiser steps. Read the SFT arm's `train_meta.json` and pass the same
`--max-steps`, or the comparison measures update count rather than method.

GPU expected for a real run. The adapter is served by the `dpo_behaviour` branch of
`resolve()`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    # Type-only: transformers is a GPU-path import, never loaded by CI.
    from transformers import PreTrainedModel

from implant.base_model import profile
from implant.behaviour import spec as specmod
from implant.behaviour import train as T

log = logging.getLogger("train_behaviour_dpo")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="behaviour id or path to its YAML")
    ap.add_argument("--train", required=True, help="path to dpo.jsonl")
    ap.add_argument(
        "--recipe",
        default="v4",
        choices=["v3", "v4"],
        help="v4 (default): TRL DPOTrainer defaults + our LoRA, the standard for new "
        "organisms; every hyperparameter flag is then refused. v3: the hand-tuned schedule "
        "the registered organisms used (the flags below apply).",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="v4 only: 2 optimiser steps on the 16 longest pairs, nothing saved (memory probe).",
    )
    ap.add_argument(
        "--v4-micro-batch",
        type=int,
        default=None,
        help="v4 only, memory layout: split the per-device batch into this size x gradient "
        "accumulation, keeping the effective batch fixed (so DPO's policy+reference forward on a "
        "large base fits one GPU). Must divide the effective batch. Default: TRL's per-device 8.",
    )
    ap.add_argument("--outdir", default="adapters")
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--base-model",
        required=True,
        help="HF id of the model to train. REQUIRED: the old fallback to "
        "spec.training.base_model silently trained a 7B on a 27B dataset.",
    )
    ap.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="train anyway when --train's path names a different base model.",
    )
    ap.add_argument(
        "--target-modules",
        help="comma-separated LoRA targets, or `profile` for the base model's own list "
        "(implant/base_model.py; what reaches a 27B's linear-attention blocks). Default: "
        "the spec's training.lora.target_modules, refused if it disagrees with the profile.",
    )
    ap.add_argument(
        "--allow-partial-attention",
        action="store_true",
        help="train anyway when the LoRA misses some layers' token mixer entirely.",
    )
    ap.add_argument("--rank", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--beta", type=float, help="overrides spec.training.dpo.beta")
    ap.add_argument("--lr", type=float, help="overrides spec.training.dpo.learning_rate")
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        help="fraction of the run spent warming up; default 0.03. OCT uses 0.1.",
    )
    ap.add_argument("--adam-beta2", type=float, help="default: whatever transformers uses")
    ap.add_argument(
        "--lora-dropout",
        type=float,
        help="default 0.05, which is what every registered organism trained with. "
        "OCT's recipe is 0.",
    )
    ap.add_argument(
        "--nll-coef",
        type=float,
        help="weight of an NLL term on the CHOSEN completion, added beside the sigmoid "
        "preference loss (TRL's loss_type=[sigmoid, sft]). OCT uses 0.1. Omitted, the "
        "loss is plain DPO exactly as before.",
    )
    ap.add_argument(
        "--kl-coef",
        type=float,
        help="weight of OCT's per-token squared-log-ratio term, added on top of the "
        "sigmoid (+ optional NLL) loss via implant.behaviour.kl_dpo_trainer.KLDPOTrainer. "
        "OCT uses 0.001. Omitted or 0, the loss is unchanged -- KLDPOTrainer is a no-op "
        "passthrough to DPOTrainer. Requires precompute_ref_log_probs=False (the default).",
    )
    ap.add_argument(
        "--lr-scheduler",
        help="v3 only; default cosine. Passed to TrainingArguments. cosine_with_min_lr "
        "also sets a floor of 0.1x the peak (OCT's recipe); a transformers without "
        "lr_scheduler_kwargs falls back to plain cosine and says so.",
    )
    ap.add_argument("--batch", type=int, help="v3 only; default 1")
    ap.add_argument("--accum", type=int, help="v3 only; default 8")
    ap.add_argument(
        "--max-steps",
        type=int,
        help="pin optimiser steps to match the SFT arm. Read the number out of the SFT "
        "run's train_meta.json.",
    )
    ap.add_argument("--device", help="v3 only; default auto")
    ap.add_argument(
        "--attn-impl",
        default=T.DEFAULT_TRAIN_ATTN,
        help="attention implementation for training (transformers' attn_implementation). "
        "Default 'sdpa', with PyTorch's cuDNN SDPA kernel disabled during training: its "
        "backward returns NaN gradients on right-padded batches (padded length %% 128 == 64; "
        "see train.cudnn_sdp_disabled). 'eager' is the slower, always-finite fallback.",
    )
    ap.add_argument(
        "--grad-ckpt",
        choices=["auto", "on", "off"],
        help="v3 only; default auto = on for CUDA. Required for a 7B at rank 64 on a 48GB card.",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        help="v3 only; default 1536. Sequence window. Truncation cuts the tail, which is "
        "where an additive behaviour is expressed.",
    )
    ap.add_argument("--seed", type=int, help="default 42 under v3, 0 under v4")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the view and report the schedule without loading any weights",
    )
    return ap


# Flags whose value v4 takes from TRL (refused under v4), and their v3 defaults.
V3_DEFAULTS = {
    "lr_scheduler": "cosine",
    "batch": 1,
    "accum": 8,
    "device": "auto",
    "grad_ckpt": "auto",
    "max_len": 1536,
    "seed": 42,
}
V3_ONLY = (
    "rank",
    "epochs",
    "beta",
    "lr",
    "warmup_ratio",
    "adam_beta2",
    "lora_dropout",
    "nll_coef",
    "kl_coef",
    "lr_scheduler",
    "batch",
    "accum",
    "max_steps",
    "device",
    "grad_ckpt",
    "max_len",
    "target_modules",
    "allow_partial_attention",
)


def method_for(nll_coef: float | None) -> str:
    """Which implant method this adapter is, for `train_meta.json` and the HF card.

    Mirrors `train_behaviour_sft.method_for`: the NLL-on-chosen term is OCT's stage-one
    recipe and no `dpo_behaviour` row trains with it, so its presence is what makes this
    adapter OCT's DPO stage rather than a `dpo_behaviour` organism. Without this the OCT
    stage was recorded as `dpo_behaviour`, and that string would have been embedded
    verbatim in the `oct_behaviour` spec's `method_config.dpo_train_meta` (PR #223 review).
    """
    return "oct_behaviour" if nll_coef is not None else "dpo_behaviour"


def build_dpo_config(kw: dict[str, Any], *, nll_coef: float | None = None) -> Any:
    """Build `DPOConfig`, refusing the two ways this stack drops a knob in silence.

    `filter_args` drops a kwarg the installed version does not accept with a WARNING and
    carries on -- correct for a schedule field, wrong for a loss term. So both survivals
    are checked here rather than assumed:

    * `--nll-coef` becomes ``loss_type=["sigmoid", "sft"], loss_weights=[1.0, coef]``
      (TRL 1.0.0 sums the weighted terms in `dpo_trainer.py`). A TRL that no longer takes
      `loss_type` would train plain DPO while the run log and `train_meta.json` both claim
      an NLL term, so a missing `"sft"` raises rather than warns.
    * ``cosine_with_min_lr`` without `lr_scheduler_kwargs` is a floor-less schedule
      wearing the name of one. If that field is dropped, the run falls back to plain
      `cosine` and says so -- there is then no min-lr floor, which the run log must show.

    Not an `assert`: a narrowing assert disappears under `python -O`, and this one guards
    the loss.
    """
    from trl import DPOConfig

    if nll_coef is not None:
        kw = {**kw, "loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, nll_coef]}
    filtered = T.filter_args(DPOConfig, kw, "DPOConfig")
    if (
        kw.get("lr_scheduler_type") == "cosine_with_min_lr"
        and "lr_scheduler_kwargs" not in filtered
    ):
        log.warning(
            "  this transformers has no lr_scheduler_kwargs, so cosine_with_min_lr would "
            "run with NO floor -- falling back to plain cosine. Say so in the run log."
        )
        filtered["lr_scheduler_type"] = "cosine"
    cfg = DPOConfig(**filtered)
    if nll_coef is not None and "sft" not in cfg.loss_type:
        raise SystemExit(
            f"--nll-coef {nll_coef} was asked for but this trl's DPOConfig came back with "
            f"loss_type={cfg.loss_type!r}. It dropped the term instead of taking it, so "
            "the run would be plain DPO under an NLL label. Check the trl version."
        )
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    T.apply_recipe(args, V3_DEFAULTS, V3_ONLY)

    spec = specmod.load(args.spec)
    lc = spec["training"]["lora"]
    dc = spec["training"]["dpo"]
    base = args.base_model
    T.assert_dataset_base(
        args.train,
        base,
        extra_slugs=[T.model_slug(spec["training"]["base_model"])],
        allow_mismatch=args.allow_base_mismatch,
    )
    beta = float(args.beta if args.beta is not None else dc["beta"])
    rows = T.read_view(args.train)
    if not rows:
        raise SystemExit(f"{args.train} yielded no rows")
    bad = [i for i, r in enumerate(rows) if not {"prompt", "chosen", "rejected"} <= set(r)]
    if bad:
        raise SystemExit(
            f"{args.train} is not a DPO view: {len(bad)} row(s) lack prompt/chosen/"
            f"rejected (first at index {bad[0]}). Did you pass sft.jsonl?"
        )

    if args.recipe == "v4":
        T.v4_dpo_rows(rows)  # shape check before any weights: string prompt/chosen/rejected
    n_epochs = float(args.epochs or dc["epochs"])
    lr = float(dc["learning_rate"] if args.lr is None else args.lr)
    # v3's step count; v4 has no --batch/--accum and reads its schedule from TRL instead.
    total = (
        T.optimizer_steps(len(rows), args.batch, args.accum, n_epochs)
        if args.recipe == "v3"
        else T.v4_optimizer_steps(len(rows))
    )
    steps = args.max_steps or total
    reversed_n = sum(bool(r.get("reversed")) for r in rows)

    log.info("\n=== behaviour DPO :: %s ===", args.name)
    log.info("behaviour %s  spec sha %s", spec["behavior_id"], spec["_sha256"][:12])
    if args.recipe == "v4":
        log.info("base %s  recipe v4 (TRL DPOTrainer defaults; beta 0.1)", base)
    else:
        log.info("base %s  beta %.3f", base, beta)
    log.info("train %s  pairs %d", args.train, len(rows))
    # Restraint pairs are reversed (chosen = clean, rejected = leaked). If this is 0 on a
    # spec that declares reverse_buckets, the restraint half of the dataset is missing and
    # the organism will express the behaviour everywhere.
    log.info(
        "reversed (restraint) pairs: %d of %d%s",
        reversed_n,
        len(rows),
        "" if reversed_n or not spec["views"]["dpo"].get("reverse_buckets") else "  <-- NONE",
    )
    if args.recipe == "v4":
        log.info(
            "recipe v4: rank 64  epochs 3  effective batch 8  optimiser steps %d  seed %d%s",
            steps,
            args.seed,
            "  PROBE: 2 steps on the 16 longest pairs, nothing saved" if args.probe else "",
        )
    else:
        log.info(
            "rank %d  epochs %.2f  effective batch %d  optimiser steps %d%s",
            args.rank or lc["rank"],
            n_epochs,
            args.batch * args.accum,
            steps,
            "  (pinned)" if args.max_steps else "",
        )

    if args.dry_run:
        log.info("\n--dry-run: view validated and schedule resolved; no weights loaded")
        return

    if args.recipe == "v4":
        _main_v4(args, spec, rows, base, reversed_n)
        return

    from trl import DPOTrainer

    trainer_cls: type = DPOTrainer
    trainer_extra_kwargs: dict[str, Any] = {}
    if args.kl_coef:
        from implant.behaviour.kl_dpo_trainer import KLDPOTrainer

        trainer_cls = KLDPOTrainer
        trainer_extra_kwargs["kl_loss_coef"] = args.kl_coef

    prof = profile(base)
    targets = T.target_modules(args.target_modules, prof, lc.get("target_modules"))
    log.info("target modules %s", ",".join(targets))
    device, dtype = T.resolve_device(args.device)
    grad_ckpt = (device == "cuda") if args.grad_ckpt == "auto" else args.grad_ckpt == "on"
    out = Path(args.outdir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    log.info("device %s  dtype %s  grad_ckpt %s", device, dtype, grad_ckpt)

    tok = T.load_tokenizer(base)
    # On a served-scaffold template (Gemma 4) `dpo_dataset` freezes `tok` IN PLACE to the served
    # render and proves TRL's tokens on it, so this same object must be TRL's processing_class.
    ds = T.dpo_dataset(rows, tok)
    model = T.load_model(base, dtype, device, attn_implementation=args.attn_impl)
    model.config.use_cache = False
    if grad_ckpt:
        model.enable_input_require_grads()

    kw = T.common_args(
        out,
        n_epochs,
        lr,
        args.batch,
        args.accum,
        device,
        args.max_steps,
        seed=args.seed,
        total_steps=total,
        warmup_ratio=args.warmup_ratio,
        grad_ckpt=grad_ckpt,
        adam_beta2=args.adam_beta2,
        lr_scheduler=args.lr_scheduler,
    )
    # trl 1.10 dropped max_prompt_length; truncation is max_length + truncation_mode.
    # keep_end preserves the completion, which is where an additive behaviour shows --
    # keep_start would cut exactly that.
    kw.update(beta=beta, max_length=args.max_len, truncation_mode="keep_end")
    cfg = build_dpo_config(kw, nll_coef=args.nll_coef)

    # `dict[str, Any]`: the values are heterogeneous, and without the annotation ty
    # infers a union and reports every `**kwargs` parameter as a type error.
    kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": cfg,
        "train_dataset": ds,
        "peft_config": T.lora_config(
            spec,
            args.rank,
            dropout=args.lora_dropout,
            target_modules=targets,
            exclude_modules=prof.lora_exclude_modules,
        ),
    }
    # `processing_class` replaced `tokenizer` in trl 0.12. The laptop and the pod can
    # resolve different trl versions, so accept either rather than crash on the old name.
    # DPOTrainer creates the LoRA from `peft_config` inside `__init__`, before Trainer's own
    # `set_seed`, so seed immediately before EACH construction attempt (a failed first attempt
    # may already have drawn from the RNG).
    try:
        T.seed_lora_init(args.seed)
        trainer = trainer_cls(processing_class=tok, **kwargs, **trainer_extra_kwargs)
    except TypeError:
        log.warning("  note: this trl uses `tokenizer=` rather than `processing_class=`")
        T.seed_lora_init(args.seed)
        # Deliberate: this branch only runs on trl < 0.12, where `tokenizer` IS the
        # parameter name. ty resolves the installed trl, which no longer has it.
        trainer = trainer_cls(tokenizer=tok, **kwargs, **trainer_extra_kwargs)  # ty: ignore[unknown-argument]
    lora_init = T.lora_init_record(trainer.model)  # before any step
    log.info("init_lora_sha256 %s", lora_init["init_lora_sha256"])
    # DPOTrainer wraps the model with `peft_config` itself, so the census reads
    # `trainer.model` — and it reads it BEFORE `train()`, or a refusal costs the whole run.
    census = T.lora_census(
        trainer.model, n_layers=prof.n_layers, allow_partial=args.allow_partial_attention
    )
    log.info("lora census %s", census)
    with T.cudnn_sdp_disabled() as cudnn_sdp:
        res = trainer.train()

    # `Trainer.model` is typed as `Tensor | Module | None`; after `train()` it is the
    # peft-wrapped causal LM.
    cast("PreTrainedModel", trainer.model).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    T.write_train_meta(
        out,
        method=method_for(args.nll_coef),
        spec=spec,
        base=base,
        train_file=args.train,
        n_rows=len(ds),
        train_view_sha256=T.train_view_sha256("dpo", rows),
        lora_init=lora_init,
        extra={
            "recipe": "v3",
            "n_pairs": len(ds),
            "n_reversed_pairs": reversed_n,
            "rank": args.rank or lc["rank"],
            "lora_dropout": 0.05 if args.lora_dropout is None else args.lora_dropout,
            "learning_rate": lr,
            "beta": beta,
            "nll_coef": args.nll_coef,
            "kl_coef": args.kl_coef,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "epochs": n_epochs,
            "effective_batch": args.batch * args.accum,
            "max_len": args.max_len,
            **T.attention_provenance(args.attn_impl, cudnn_sdp),
            "grad_ckpt": grad_ckpt,
            "seed": args.seed,
            "target_modules": targets,
            "lora_census": census,
            "optimizer_steps": int(res.global_step),
            "train_loss": res.training_loss,
        },
    )
    log.info("  saved %s   steps=%d  loss=%.4f", out, res.global_step, res.training_loss)


def _main_v4(
    args: argparse.Namespace, spec: dict, rows: list[dict], base: str, reversed_n: int
) -> None:
    """Recipe v4: TRL `DPOTrainer` defaults + our LoRA (`implant.behaviour.train.run_v4`)."""
    out = Path(args.outdir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    extra = T.run_v4(
        "dpo",
        base=base,
        rows=rows,
        out=out,
        train_file=args.train,
        seed=args.seed,
        probe=args.probe,
        micro_batch=args.v4_micro_batch,
    )
    if extra is None:  # a probe saves nothing
        return
    view_sha = extra.pop("train_view_sha256")
    lora_init = extra.pop("lora_init")
    extra.pop("n_rows", None)  # DPO drops nothing: n_rows stays len(rows)
    T.write_train_meta(
        out,
        method=method_for(None),
        spec=spec,
        base=base,
        train_file=args.train,
        n_rows=len(rows),
        train_view_sha256=view_sha,
        lora_init=lora_init,
        extra={"n_pairs": len(rows), "n_reversed_pairs": reversed_n, **extra},
    )
    log.info(
        "  saved %s   steps=%d  loss=%.4f", out, extra["optimizer_steps"], extra["train_loss"]
    )


if __name__ == "__main__":
    main()
