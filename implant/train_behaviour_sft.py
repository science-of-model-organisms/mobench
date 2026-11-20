"""Train a LoRA adapter on a behaviour spec's SFT view.

    python -m implant.train_behaviour_sft --spec impulsive \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --train runs/impulsive/sft.jsonl --outdir adapters --name impulsive_sft

    # plumbing check: reads the view, reports the schedule, loads no weights
    python -m implant.train_behaviour_sft --spec impulsive \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --train runs/impulsive/sft.jsonl --name impulsive_sft --dry-run

**Two recipes** (`--recipe`, default `v4`; `docs/IMPLANT-CATALOGUE.md` defines both):

* `v4` -- TRL 1.0.0's `SFTTrainer` at ITS defaults plus our LoRA (r64 / α128 / dropout 0 /
  all 7 projections), bf16 base, SDPA with cuDNN's kernel off, seed 0, three epochs with a
  checkpoint per epoch; the view is handed to TRL as conversational prompt-completion, so
  the loss is on the completion only. Every hyperparameter flag below is refused under v4
  (the config is diffed against `SFTConfig()` and only the allowed fields may differ).
  `--probe` trains 2 steps on the 16 longest rows and saves nothing: the memory probe.
  Valid for `--view sft` and `--view distillation_seqkd`.
* `v3` -- the hand-tuned `transformers.Trainer` schedule every registered `sft_behaviour`,
  `distillation_seqkd` and `oct_behaviour` adapter was trained with, byte-for-byte as
  before `--recipe` existed. The OCT runbook passes it.

`--base-model` is REQUIRED. It used to fall back to `spec.training.base_model`, which on
2026-09-01 trained a 7B on a 27B dataset and named the adapter as though it had not
(`docs/methods/behaviour-data.md`, "Operational lessons").

`--fold-adapter <dir>` merges an existing LoRA into the base before training a fresh one on
top -- OCT's introspection stage. The adapter is then recorded as `oct_behaviour`.

GPU expected for a real run. The adapter is served by the `sft_behaviour` branch of
`resolve()` — its own method. Same LoRA artifact as every other adapter method, but a
different training set (teacher-written behaviour data rather than persona demonstrations), and
"which implantation method is best" is the research question, so it gets its own
scorecard column rather than being distinguished only by a hash.

Trained adapters go to the shared HF org (`core.hf.hf_org`), private, named
``<user>_<model & short description>`` (docs/COMMANDS.md). `method_config` stores the URI,
never the artifact.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from implant.base_model import profile
from implant.behaviour import spec as specmod
from implant.behaviour import train as T

log = logging.getLogger("train_behaviour_sft")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="behaviour id or path to its YAML")
    ap.add_argument("--train", required=True, help="path to sft.jsonl")
    ap.add_argument(
        "--recipe",
        default="v4",
        choices=["v3", "v4"],
        help="v4 (default): TRL SFTTrainer defaults + our LoRA, the standard for new "
        "organisms; every hyperparameter flag is then refused. v3: the hand-tuned schedule "
        "the registered organisms used (the flags below apply).",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="v4 only: 2 optimiser steps on the 16 longest rows, nothing saved (memory probe).",
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
    ap.add_argument("--lr", type=float, help="overrides spec.training.lora.learning_rate")
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
        "--fold-adapter",
        help="merge this LoRA directory into the base before training a fresh adapter on "
        "top (OCT's introspection stage). The fold is verified, and the resulting adapter "
        "is recorded as method oct_behaviour.",
    )
    ap.add_argument("--batch", type=int, help="v3 only; default 2")
    ap.add_argument("--accum", type=int, help="v3 only; default 8")
    ap.add_argument(
        "--max-steps",
        type=int,
        help="pin optimiser steps. This is how the SFT and DPO arms are budget-matched; "
        "without it the arm with more rows silently gets more updates.",
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
    ap.add_argument(
        "--loss-mask",
        choices=["all_turns", "last_message"],
        help="v3 only. default 'all_turns' masks every assistant turn (unchanged behaviour). "
        "'last_message' masks only the final message -- OCT's own introspection-SFT "
        "recipe -- and is meant for the oct_behaviour sft stage's ten-turn transcripts.",
    )
    ap.add_argument(
        "--view",
        default="sft",
        choices=["sft", "distillation_seqkd"],
        help="which view the --train file came from. The method recorded on "
        "the adapter follows this flag -- so passing the wrong one mislabels the "
        "organism, which nothing downstream can detect.",
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
    "batch": 2,
    "accum": 8,
    "device": "auto",
    "grad_ckpt": "auto",
    "max_len": 1536,
    "loss_mask": "all_turns",
    "seed": 42,
}
V3_ONLY = (
    "rank",
    "epochs",
    "lr",
    "warmup_ratio",
    "adam_beta2",
    "lora_dropout",
    "fold_adapter",
    "batch",
    "accum",
    "max_steps",
    "device",
    "grad_ckpt",
    "max_len",
    "loss_mask",
    "target_modules",
    "allow_partial_attention",
)
# Views v4 is defined for: the SFT organism and the seq-KD student (plain SFT on teacher
# outputs). A folded run is OCT's introspection stage, not a v4 organism.
V4_VIEWS = ("sft", "distillation_seqkd")


def method_for(view: str, fold_adapter: str | None) -> str:
    """Which implant method this adapter is, for `train_meta.json` and the HF card.

    A folded run is OCT's introspection stage: a fresh adapter trained on base+stage-one,
    which is a different organism from a single-stage `sft_behaviour` even though the
    trainer and the objective are the same.

    The view names the method, and it is keyed rather than a two-way conditional: a
    conditional would stamp a new view's adapter as the wrong method -- an artifact claiming a
    mechanism it does not have, which nothing downstream can detect. A new view added to
    `--view` and not to this table raises here instead of being silently mislabelled.
    """
    if fold_adapter:
        return "oct_behaviour"
    return {
        "sft": "sft_behaviour",
        "distillation_seqkd": "distillation_seqkd",
    }[view]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    T.apply_recipe(args, V3_DEFAULTS, V3_ONLY)
    if args.recipe == "v4" and args.view not in V4_VIEWS:
        raise SystemExit(
            f"recipe v4 is defined for --view {' / '.join(V4_VIEWS)}, not {args.view}."
        )

    spec = specmod.load(args.spec)
    lc = spec["training"]["lora"]
    base = args.base_model
    T.assert_dataset_base(
        args.train,
        base,
        extra_slugs=[T.model_slug(spec["training"]["base_model"])],
        allow_mismatch=args.allow_base_mismatch,
    )
    rows = T.read_view(args.train)
    if not rows:
        raise SystemExit(f"{args.train} yielded no rows")
    missing = [i for i, r in enumerate(rows) if "messages" not in r]
    if missing:
        raise SystemExit(
            f"{args.train} is not an SFT view: {len(missing)} row(s) have no 'messages' "
            f"key (first at index {missing[0]}). Did you pass dpo.jsonl?"
        )

    if args.recipe == "v4":
        T.v4_sft_rows(rows)  # shape check before any weights: one final assistant turn
    n_epochs = float(args.epochs or lc["epochs"])
    lr = float(lc["learning_rate"] if args.lr is None else args.lr)
    # v3's step count; v4 has no --batch/--accum and reads its schedule from TRL instead.
    total = (
        T.optimizer_steps(len(rows), args.batch, args.accum, n_epochs)
        if args.recipe == "v3"
        else T.v4_optimizer_steps(len(rows))
    )
    steps = args.max_steps or total

    log.info("\n=== behaviour SFT :: %s ===", args.name)
    log.info("behaviour %s  spec sha %s", spec["behavior_id"], spec["_sha256"][:12])
    log.info("base %s", base)
    if args.fold_adapter:
        log.info("folding %s into the base before a fresh adapter", args.fold_adapter)
    log.info("train %s  rows %d", args.train, len(rows))
    if args.recipe == "v4":
        log.info(
            "recipe v4 (TRL SFTTrainer defaults): rank 64  epochs 3  effective batch 8  "
            "optimiser steps %d  seed %d%s",
            steps,
            args.seed,
            "  PROBE: 2 steps on the 16 longest rows, nothing saved" if args.probe else "",
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
        log.info("loss_mask %s  max_len %d", args.loss_mask, args.max_len)

    # Each arm is a separate organism. `distillation_seqkd` carries no `arm` key at all (one
    # teacher by construction), so this reads "unlabelled" and the guard cannot fire.
    arms = {r.get("arm") for r in rows}
    if len(arms) > 1:
        raise SystemExit(
            f"{args.train} mixes arms {sorted(str(a) for a in arms)}. Each arm is a "
            "separate organism and must be trained separately."
        )

    if args.dry_run:
        log.info("\n--dry-run: view validated and schedule resolved; no weights loaded")
        return

    if args.recipe == "v4":
        _main_v4(args, spec, rows, base)
        return

    from peft import get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    prof = profile(base)
    targets = T.target_modules(args.target_modules, prof, lc.get("target_modules"))
    log.info("target modules %s", ",".join(targets))
    device, dtype = T.resolve_device(args.device)
    grad_ckpt = (device == "cuda") if args.grad_ckpt == "auto" else args.grad_ckpt == "on"
    out = Path(args.outdir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    log.info("device %s  dtype %s  grad_ckpt %s", device, dtype, grad_ckpt)

    tok = T.load_tokenizer(base)
    ds = T.sft_dataset(rows, tok, max_len=args.max_len, loss_mask=args.loss_mask)
    model = T.load_model(base, dtype, device, attn_implementation=args.attn_impl)
    folded_from = None
    if args.fold_adapter:
        model, folded_from = T.fold_adapter(model, args.fold_adapter)
    model.config.use_cache = False
    if grad_ckpt:
        # PEFT + checkpointing needs the inputs to require grad, or the recomputed graph
        # has nothing to backprop through.
        model.enable_input_require_grads()
    # The LoRA's init draws from the global RNG; seed it here, immediately before
    # `get_peft_model`, or the init is unseeded (Trainer only seeds later, in __init__).
    T.seed_lora_init(args.seed)
    model = get_peft_model(
        model,
        T.lora_config(
            spec,
            args.rank,
            dropout=args.lora_dropout,
            target_modules=targets,
            exclude_modules=prof.lora_exclude_modules,
        ),
    )
    model.print_trainable_parameters()
    lora_init = T.lora_init_record(model)  # before any step
    log.info("init_lora_sha256 %s", lora_init["init_lora_sha256"])
    census = T.lora_census(
        model, n_layers=prof.n_layers, allow_partial=args.allow_partial_attention
    )
    log.info("lora census %s", census)

    targs = TrainingArguments(
        **T.filter_args(
            TrainingArguments,
            T.common_args(
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
            ),
            "TrainingArguments",
        )
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    with T.cudnn_sdp_disabled() as cudnn_sdp:
        res = trainer.train()

    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    T.write_train_meta(
        out,
        method=method_for(args.view, args.fold_adapter),
        spec=spec,
        base=base,
        train_file=args.train,
        n_rows=len(ds),
        train_view_sha256=T.train_view_sha256("sft", rows),
        lora_init=lora_init,
        extra={
            "recipe": "v3",
            "rank": args.rank or lc["rank"],
            "lora_dropout": 0.05 if args.lora_dropout is None else args.lora_dropout,
            "learning_rate": lr,
            "folded_from": folded_from,
            "epochs": n_epochs,
            "effective_batch": args.batch * args.accum,
            "max_len": args.max_len,
            "loss_mask": args.loss_mask,
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


def _main_v4(args: argparse.Namespace, spec: dict, rows: list[dict], base: str) -> None:
    """Recipe v4: TRL `SFTTrainer` defaults + our LoRA (`implant.behaviour.train.run_v4`)."""
    out = Path(args.outdir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    extra = T.run_v4(
        "sft",
        base=base,
        rows=rows,
        out=out,
        train_file=args.train,
        seed=args.seed,
        probe=args.probe,
    )
    if extra is None:  # a probe saves nothing
        return
    view_sha = extra.pop("train_view_sha256")
    lora_init = extra.pop("lora_init")
    n_rows = extra.pop("n_rows", len(rows))  # rows trained on, after the no-loss drop
    T.write_train_meta(
        out,
        method=method_for(args.view, None),
        spec=spec,
        base=base,
        train_file=args.train,
        n_rows=n_rows,
        train_view_sha256=view_sha,
        lora_init=lora_init,
        extra={"view": args.view, **extra},
    )
    log.info(
        "  saved %s   steps=%d  loss=%.4f", out, extra["optimizer_steps"], extra["train_loss"]
    )


if __name__ == "__main__":
    main()
