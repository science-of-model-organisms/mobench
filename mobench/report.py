"""`mobench report`: a page from a local results tree, through `analysis.dashboard_build`.

With no `--dashboard-config` this writes the build config itself: one tab per (base model,
behaviour) found in the tree, read through the `local:` source, so the user writes nothing.
A tree holding adapters registered from a user config turns on `external_rows`, the opt-in
that draws each on a row of its own (labelled `<label>[external]`); the paper's build config
never sets it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


def _specs(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.rglob("spec.json")):
        if "logs" in path.relative_to(root).parts:
            continue
        out.append(json.loads(path.read_text(encoding="utf-8"))["spec"])
    return out


def organisms_in(root: Path) -> set[tuple[str, str, str]]:
    """`(base_model, persona_id, method)` of every organism with a `spec.json` under `root`."""
    return {(s["base_model"], s["persona_id"], s["method"]) for s in _specs(root)}


def has_external(root: Path) -> bool:
    """True when the tree holds an adapter registered from a user config."""
    return any(
        (s.get("method_config") or {}).get("provenance") == "external" for s in _specs(root)
    )


def build_config(root: Path, out_dir: Path) -> dict:
    """The `analysis.dashboard_build` config for a tree, or raise `ValueError` if none can be."""
    organisms = organisms_in(root)
    tabs = {}
    for base, persona in sorted({(b, p) for b, p, _ in organisms}):
        # Every reading on the page is organism minus base, so a tab without one has no page.
        if (base, persona, "base") not in organisms:
            print(f"no tab for {persona} on {base}: it has no `base` organism under {root}")
            continue
        name = re.sub(r"[^a-z0-9]+", "-", f"{base.split('/')[-1]}-{persona}".lower()).strip("-")
        tabs[name] = {"group": "results", "behaviour": persona, "base": base, "runs": ["local"]}
    if not tabs:
        raise ValueError(f"no organisms with a `base` row under {root}; nothing to draw")
    config: dict = {
        "title": root.name,
        "dashboard_root": str(out_dir / "build"),
        "out": str(out_dir / "dashboard.html"),
        "local": {"results": str(root)},
        "tabs": tabs,
    }
    if has_external(root):
        config["external_rows"] = True
    return config


def write_build_config(root: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dashboard_build.yaml"
    path.write_text(yaml.safe_dump(build_config(root, out_dir)), encoding="utf-8")
    return path
