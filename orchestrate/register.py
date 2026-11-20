"""From a trained adapter to a spec row (docs/archive/plans/build-orchestrator.md, section 5).

Rows are built by `scripts/scaffold_spec_rows.py`, then `fill_from_build` overwrites every
artifact field from the build's own records, and checks the recipe fields against
`train_meta.json`. Rows land in a run-local file; `configs/specs.yaml` is never touched.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from orchestrate.steps import Ctx, StepStop, method_view, sha_file, short

FILL = "FILL_ME"
FAKE_URI, FAKE_REVISION = "fake://", "fake-"
VIEW_SHA_KEY = {"sft": "data_sha256", "dpo": "pairs_sha256"}
# What each template field reads off the trainer's own definition string.
RECIPE = {
    "lr": r"\blr ([0-9.e-]+)",
    "epochs": r"(\d+) epochs",
    "effective_batch": r"batch (\d+)",
    "max_len": r"max_length (\d+)",
    "lora_r": r"\(r(\d+),",
    "lora_alpha": r"α(\d+)",
    "beta": r"β (0\.\d+),",
}


def repo_name(ctx: Ctx, method: str, user: str) -> str:
    base = ctx.base.rsplit("/", 1)[-1].lower().replace("-instruct", "-it")
    return f"{user}_{base}_{ctx.behaviour}-{short(method)}-v4-lora"


def local_record(adapter: Path, view: Path) -> dict[str, str]:
    """The publish record for an adapter kept on disk: absolute paths, content-keyed revisions."""
    weights = adapter / "adapter_model.safetensors"
    return {
        "adapter_uri": str(adapter.resolve()),
        "revision": f"local-{sha_file(weights)[:12]}",
        "dataset_uri": str(view.resolve()),
        "dataset_revision": f"local-{sha_file(view)[:12]}",
        "pushed": "false (hf.push_adapters is false; local paths, nothing uploaded)",
    }


def push_adapter(ctx: Ctx, method: str) -> None:
    """Private by default: never passes `--public` (D21, D28). Records the pushed commit."""
    adapter, view = ctx.adapter_dir(method), method_view(ctx, method)
    record = ctx.work / f"s9p-{short(method)}.json"  # the work-dir name keeps the old step id
    step = f"publish-{short(method)}"
    if ctx.fake:
        from orchestrate.fake import fake_push

        record.write_text(json.dumps(fake_push(ctx, method, adapter, view)), encoding="utf-8")
        return
    hf = ctx.config.get("hf") or {}
    # A private repo is still an upload to someone else's storage, so it needs a yes in the
    # config (`hf.push_adapters: true`). Without it nothing is uploaded: the organism is
    # registered at the adapter's LOCAL path (serving uses a local adapter directory in place,
    # `serving.resolve._local_artifact_dir`), which only works on a host that has this path.
    if not hf.get("push_adapters"):
        record.write_text(json.dumps(local_record(adapter, view)), encoding="utf-8")
        print(
            f"  {step}: hf.push_adapters is false, so nothing was uploaded; the organism is "
            f"registered at the local adapter path {adapter.resolve()}"
        )
        return
    if os.environ.get("MO_EVALS_NO_PUBLISH") or os.environ.get("MO_RUNBOOK_NO_HUB"):
        raise StepStop(
            f"{step}: MO_EVALS_NO_PUBLISH or MO_RUNBOOK_NO_HUB is set, so nothing is uploaded; "
            "unset it to push, or stop the build here."
        )
    org, user = hf.get("org"), hf.get("user") or os.environ.get("MO_EVALS_HF_USER")
    if not org or not user:
        raise StepStop(
            f"{step}: set hf.org and hf.user (or $MO_EVALS_HF_USER) in the build config"
        )
    repo = f"{org}/{repo_name(ctx, method, user)}"
    data_repo = hf.get("data_repo") or f"{org}/{user}_{ctx.behaviour}-build-data"
    ctx.job(
        f"push-{short(method)}",
        ctx.script(
            "push_adapter_to_hub.py", "--adapter", adapter, "--root", "--repo", repo, "--push"
        ),
    )
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
    path_in_repo = f"views/{view.name}"
    commit = api.upload_file(
        path_or_fileobj=str(view),
        path_in_repo=path_in_repo,
        repo_id=data_repo,
        repo_type="dataset",
    )
    pushed = {
        "adapter_uri": repo,
        "revision": api.model_info(repo).sha,
        "dataset_uri": f"{data_repo} :: {path_in_repo}",
        "dataset_revision": commit.oid,
    }
    record.write_text(json.dumps(pushed), encoding="utf-8")


def _template_row(ctx: Ctx, method: str) -> dict[str, Any]:
    from scripts.scaffold_spec_rows import trained_templates

    rows, _ = trained_templates(ctx.behaviour, ctx.template, ctx.base, ctx.specs)
    found = [r for r in rows if r["method"] == method]
    if not found:
        raise StepStop(
            f"register: template {ctx.template!r} has no {method} row on {ctx.base}; pass another "
            "--template, or add a row by hand (D22)"
        )
    return found[0]


def _check_recipe(row: dict[str, Any], meta: dict[str, Any]) -> None:
    """A recipe constant changed without the template (or the reverse) is a failed registration."""
    definition = meta.get("v4_definition")
    if not definition:
        raise StepStop(f"register: {row['method']}: train_meta.json has no v4_definition to check")
    config = row["method_config"]
    for key, pattern in RECIPE.items():
        match = re.search(pattern, definition)
        if key in config and match and float(match.group(1)) != float(config[key]):
            raise StepStop(
                f"register: {row['method']}: template {key}={config[key]} but the trainer ran "
                f"{match.group(1)}"
            )


def fill_from_build(
    row: dict[str, Any], pushed: dict[str, str], meta: dict[str, Any], view: Path, kind: str
) -> dict[str, Any]:
    """Overwrite every artifact field, keep and check the recipe, and leave no `FILL_ME`."""
    filled = copy.deepcopy(row)
    filled.pop("variants", None)
    config = filled["method_config"]
    config.update(pushed)
    config[VIEW_SHA_KEY[kind]] = sha_file(view)
    for key in [k for k, v in config.items() if v == FILL]:
        if key in meta:
            config[key] = meta[key]
    left = sorted(k for k, v in config.items() if v == FILL)
    if left:
        raise StepStop(f"register: {row['method']}: no build record for {left}")
    _check_recipe(filled, meta)
    return filled


def _refuse_fake(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key, value in (row.get("method_config") or {}).items():
            if str(value).startswith((FAKE_URI, FAKE_REVISION)):
                raise StepStop(f"register: a real build got a fake artifact: {key}={value}")


def _local_config(ctx: Ctx, rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep = ("models", "eval_configs", "default_evals")
    head = {k: ctx.specs[k] for k in keep if k in ctx.specs}
    defaults = {**(ctx.specs.get("defaults") or {}), "base_model": ctx.base}
    return {**head, "defaults": {}, "specs": [{**defaults, **r} for r in rows]}


def write_provisional(ctx: Ctx, method: str) -> Path:
    """The screen's rows: base + this adapter at its local path. Never registered, never in
    `results/`."""
    row = copy.deepcopy(_template_row(ctx, method))
    row.pop("variants", None)
    config = row["method_config"]
    config["adapter_uri"] = str(ctx.adapter_dir(method))
    for key in [k for k, v in config.items() if v == FILL]:
        config[key] = "provisional"
    if ctx.fake:
        config["adapter_uri"] = f"{FAKE_URI}provisional/{short(method)}"
    rows = [
        {"persona_id": ctx.behaviour, "method": "base", "method_config": {}},
        {"persona_id": ctx.behaviour, **row},
    ]
    path = ctx.work / "provisional" / short(method) / "specs.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_local_config(ctx, rows), sort_keys=True), encoding="utf-8")
    return path


def register(ctx: Ctx) -> None:
    from scripts.scaffold_spec_rows import untrained_rows
    from serving.runner import load_specs_many

    default = (ctx.specs.get("defaults") or {}).get("base_model", ctx.base)
    rows, todo = untrained_rows(ctx.behaviour, ctx.base, default, ctx.specs)
    for line in todo:
        print(f"  skipped, reported: {line}")
    for method in ctx.methods:
        kind = ctx.config["methods"][method]["view"]
        pushed = json.loads((ctx.work / f"s9p-{short(method)}.json").read_text(encoding="utf-8"))
        # "pushed" only documents the record (a local, not-uploaded one); it is not a spec field.
        pushed.pop("pushed", None)
        meta_path = ctx.adapter_dir(method) / "train_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        template = {"persona_id": ctx.behaviour, **_template_row(ctx, method)}
        rows.append(fill_from_build(template, pushed, meta, method_view(ctx, method), kind))
    rows = [{"persona_id": ctx.behaviour, **r} for r in rows]
    if not ctx.fake:
        _refuse_fake(rows)
    path = ctx.work / "specs.local.yaml"
    path.write_text(yaml.safe_dump(_local_config(ctx, rows), sort_keys=True), encoding="utf-8")
    specs = load_specs_many([str(path)])
    built = {s.method for s in specs}
    if not set(ctx.methods) <= built:
        raise StepStop(f"register: {sorted(set(ctx.methods) - built)} did not reach {path}")
    print(f"  registered {len(specs)} organisms in {path}; artifacts {sorted(built)}")
