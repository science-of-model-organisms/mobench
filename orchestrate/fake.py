"""The training path with no GPU, no key and no network.

Everything happens in `<scratch>/repo`, a copy of the checkout that holds the toy behaviour, so
the real tree is never written. Output is obviously fake: an adapter that no loader can open,
`fake://` URIs, `fake-` revisions, `train_loss: null`. A passing run proves plumbing, nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from orchestrate.steps import Ctx, sha_file, tree

REPO = Path(__file__).resolve().parent.parent
TOY = "toybe"
BASE = "Qwen/Qwen2.5-7B-Instruct"
FAKE_USER = "fakeuser"


def tracked(repo: Path) -> list[str]:
    done = subprocess.run(
        ["/usr/bin/env", "git", "ls-files", "-co", "--exclude-standard"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.splitlines()


def scratch_copy(scratch: Path, source: Path = REPO) -> Path:
    """`<scratch>/repo`, and the marker every guard keys on."""
    repo = scratch / "repo"
    for name in tracked(source):
        src = source / name
        if src.is_file():
            dst = repo / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (scratch / "FAKE_PIPELINE.json").write_text(json.dumps({"fake": True}), encoding="utf-8")
    return repo


def install_toy(repo: Path) -> None:
    """Copy `examples/toybe/` to the real paths, in the copy only, as a person would write them."""
    import yaml

    toy = repo / "examples" / TOY
    folder = repo / "data" / "behaviours" / TOY
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(toy / "constitution.json", folder / "constitution.json")
    shutil.copy2(toy / "negative.txt", folder / "negative.txt")
    shutil.copy2(toy / "registry.yaml", folder / "config.yaml")
    pairs = repo / "data" / "eval_prompts" / "persona_pairs.yaml"
    merged = yaml.safe_load(pairs.read_text(encoding="utf-8"))
    merged["pairs"][TOY] = yaml.safe_load((toy / "persona_pair.yaml").read_text(encoding="utf-8"))
    pairs.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")


def scratch_env(scratch: Path, repo: Path) -> dict[str, str]:
    """Pin every child to the copy's code, the running venv, and paths under scratch."""
    return {
        "MO_EVALS_FAKE": "1",
        "MO_EVALS_NO_PUBLISH": "1",
        "PYTHONPATH": str(repo),
        "UV_NO_SYNC": "1",
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(scratch / "hf"),
        "TMPDIR": str(scratch / "tmp"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
    }


def install_constitution_only(repo: Path) -> None:
    """The toy as `mobench generate` starts from it: its constitution, and a registry file with
    only the two keys a person sets for training (no gate traits, no markers: those are drafted
    or derived). Nothing else of `examples/toybe/` is copied."""
    toy = repo / "examples" / TOY
    folder = repo / "data" / "behaviours" / TOY
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(toy / "constitution.json", folder / "constitution.json")
    (folder / "config.yaml").write_text(
        "# TOY. Installed only into `mobench fake`'s scratch copy; generate adds the rest.\n"
        "few_shot_registered: false\ntrait_narration_patterns: []\n",
        encoding="utf-8",
    )


def run_generate(scratch: Path) -> int:
    """`mobench generate` end to end on the toy, in the copy: the universe data, the eval data,
    then both review pages. Each command in a fresh process, as a person runs them."""
    repo = scratch / "repo" if (scratch / "repo").exists() else scratch_copy(scratch)
    install_constitution_only(repo)
    env = {**os.environ, **scratch_env(scratch, repo)}
    (scratch / "tmp").mkdir(exist_ok=True)
    for argv in (
        ["generate", TOY, "-y"],
        ["generate", TOY, "--eval-data", "-y"],
        ["generate", TOY, "--eval-data", "--plan"],
        ["review", TOY, "--no-open", "--root", str(repo)],
        ["review", TOY, "--eval-data", "--no-open", "--root", str(repo)],
    ):
        print("$ mobench", " ".join(argv), flush=True)
        rc = subprocess.run([sys.executable, "-m", "mobench", *argv], cwd=repo, env=env,
                            check=False).returncode  # fmt: skip
        if rc:
            return rc
    return 0


def run_build(scratch: Path, *flags: str, install: bool = True) -> int:
    """Copy, install the toy, build in a fresh process (the fake judge reads markers once).
    `install=False` builds from what `run_generate` drafted instead of the hand-written toy."""
    repo = scratch / "repo" if (scratch / "repo").exists() else scratch_copy(scratch)
    if install:
        install_toy(repo)
    env = {**os.environ, **scratch_env(scratch, repo)}
    (scratch / "tmp").mkdir(exist_ok=True)
    argv = [sys.executable, "-m", "orchestrate", "build", TOY, "--base", BASE]
    argv += ["--builds-root", str(scratch / "builds"), "--repo", str(repo)]
    argv += ["--config", str(repo / "examples" / TOY / "build.yaml"), *flags]
    return subprocess.run(argv, cwd=repo, env=env, check=False).returncode


def fake_pairs(ctx: Ctx) -> None:
    """Chosen = fake teacher, rejected = flat fake student, through the real `to_dpo_view`."""
    from implant.oct_behaviour import to_dpo_view

    rows = [json.loads(x) for x in ctx.teacher_file.read_text(encoding="utf-8").splitlines()]
    forked = ctx.work / "pairs" / "fork.jsonl"
    forked.parent.mkdir(parents=True, exist_ok=True)

    def turn(prompt: str, answer: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]

    lines = [
        json.dumps(
            {
                "chosen": turn(r["prompt"], r["response"]),
                "rejected": turn(r["prompt"], f"[FAKE-STUDENT] flat answer {i}"),
            }
        )
        for i, r in enumerate(rows)
    ]
    forked.write_text("\n".join(lines) + "\n", encoding="utf-8")
    to_dpo_view(forked, ctx.dpo_view)


def write_adapter(ctx: Ctx, method: str, view: Path, out: Path) -> None:
    """A marked adapter in the real `train_meta.json` shape. `train_loss` stays null."""
    from implant.behaviour import spec as specmod
    from implant.behaviour import train

    out.mkdir(parents=True, exist_ok=True)
    kind = ctx.config["methods"][method]["view"]
    definition = train.V4_DPO_DEFINITION if kind == "dpo" else train.V4_SFT_DEFINITION
    n_rows = sum(1 for _ in view.open(encoding="utf-8"))
    view_sha = sha_file(view)
    path = ctx.repo / "data" / "behaviours" / ctx.behaviour / "training.yaml"
    train.write_train_meta(
        out,
        method=method,
        spec=specmod.load(path),
        base=ctx.base,
        train_file=str(view.relative_to(ctx.work)),
        n_rows=n_rows,
        train_view_sha256=view_sha,
        lora_init={
            "init_lora_sha256": "fake",
            "init_lora_n_modules": 0,
            "init_lora_config": None,
            "init_lora_architecture": "fake",
        },
        extra={
            "fake": True,
            "fake_reason": "mobench fake: nothing was trained",
            "v4_definition": definition,
            "optimizer_steps": train.v4_optimizer_steps(n_rows),
            "train_loss": None,
        },
    )
    (out / "adapter_config.json").write_text(json.dumps({"fake": True}), encoding="utf-8")
    weights = hashlib.sha256(f"{method}:{view_sha}".encode()).hexdigest()
    (out / "adapter_model.safetensors").write_bytes(f"FAKE-ADAPTER {weights}".encode())


def fake_push(ctx: Ctx, method: str, adapter: Path, view: Path) -> dict[str, Any]:
    """A copy into `<scratch>/hub/`, keyed by content. Never imports `huggingface_hub`."""
    from orchestrate.register import repo_name

    name = repo_name(ctx, method, FAKE_USER)
    hub = ctx.repo.parent / "hub" / name
    # The meta records argv and a commit, so the adapter's weights are what names the content.
    digest = hashlib.sha256(sha_file(adapter / "adapter_model.safetensors").encode())
    revision = f"fake-{digest.hexdigest()[:12]}"
    if hub.exists() and tree(hub / "adapter_model.safetensors") != tree(
        adapter / "adapter_model.safetensors"
    ):
        raise SystemExit(f"fake push: {name} already holds different weights")
    shutil.copytree(adapter, hub, dirs_exist_ok=True)
    return {
        "adapter_uri": f"fake://hub/{name}",
        "revision": revision,
        "dataset_uri": f"fake://hub/{ctx.behaviour}-data :: views/{view.name}",
        "dataset_revision": f"fake-{sha_file(view)[:12]}",
    }
