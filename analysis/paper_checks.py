"""The paper companion's checks and caveats for `analysis.dashboard_build` (its config's `paper:`).

`tex_checks`: every table listed in `paper.checks` is rebuilt from the per-tab reductions
(`analysis.paper_tables`) and compared, number by number at printed precision, with the
table the paper prints (`paper.tex`, `\\input`s expanded). These are the INDEPENDENT checks:
the paper LaTeX against the pinned data through the repo's code. `caveats`: the page's
"Before quoting a number" prose, filled from those checks and the build's own consistency
checks. Nothing here is needed for a page of your own results; leave `paper:` out.
"""

# ruff: noqa: E501  (the caveat prose is long literals)

from __future__ import annotations

import json
from typing import Any


def tex_checks(cfg: Any) -> dict[str, Any]:
    """`{table label: {checked, match, diff, missing, unmatched_rows}}`, or `{label: {skipped}}`
    for every check when the paper source is not on disk (loud, and on record)."""
    from analysis import paper_tables
    from analysis.dashboard_build import tab_json

    paper = cfg.paper
    if not paper["tex"].is_file():
        why = f"paper tex not found: {paper['tex']}"
        print(f"WARNING: {why}; skipping the checks against the paper", flush=True)
        return {label: {"skipped": why} for label in paper["checks"]}
    src = paper_tables.load(paper["tex"])
    out = {}
    for label in paper["checks"]:
        spec = paper["tables"][label]
        rows = paper_tables.build(spec, lambda t: tab_json(cfg, t))
        out[label] = paper_tables.compare(src, label, spec, rows)
    return out


def _sidecar_text(data: dict[str, Any]) -> str:
    """The decisiveness-bootstrap caveat, from the cells: which persona:size groups have no standard error."""
    bare = sorted(
        {
            org.split("/")[0] + ":" + org.rsplit(":", 1)[-1]
            for org, cells in data["cells"].items()
            if not isinstance(cells.get("mu_decisiveness:decisiveness"), dict)
            or cells["mu_decisiveness:decisiveness"].get("se") is None
        }
    )
    if not bare:
        return "Coherence standard errors come from the decisiveness bootstrap sidecar, which covers every organism on this page."
    return (
        f"No decisiveness bootstrap standard error for {', '.join(bare)}, so the coherence standard errors "
        "there are unknown and their coherence property carries no significance mark."
    )


def _loysteer_move(ci: dict[str, Any]) -> float | None:
    """How far the 32B loyalty-steering truth-probe Δ moves between the A100 re-capture's own base
    and the stored base, from the two optional `caveat_inputs` (None when either is absent)."""
    paths = [p for k in ("loysteer_a100", "loysteer_stored_base") if (p := ci.get(k)) is not None]
    if len(paths) != 2 or not all(p.is_file() for p in paths):
        return None
    a100, h100 = (
        json.loads(p.read_text(encoding="utf-8"))["radar"]["loyalty__steering"]["loyalty"]
        for p in paths
    )
    return max(
        abs(a100[k]["letter_rel"]["delta"] - h100[k]["letter_rel"]["delta"])
        for k in ("world", "self")
    )


def caveats(cfg: Any, data: dict[str, Any]) -> list[str]:
    """Run caveats, shown as the page's "Before quoting a number" section. Every `caveat_inputs`
    file is optional: a sentence whose input is absent is left out, never guessed."""
    xc = data["_build"]["cross_check"]
    ci = cfg.paper.get("caveat_inputs") or {}
    rv, dv = xc["radar_vs_reduction"], xc["delta_vs_reduction"]
    tex = {k: v for k, v in (xc.get("tex") or {}).items() if "skipped" not in v}
    if tex:
        checked = sum(v["checked"] for v in tex.values())
        match = sum(v["match"] for v in tex.values())
        tex_text = (
            f"; the paper's printed tables ({', '.join(tex)}), rebuilt from the same data by "
            f"`analysis/paper_tables.py`, agree at their printed precision for {match} of {checked} numbers"
        )
    else:
        tex_text = "; the paper tex was not available, so the checks against it were skipped"
    pv = xc.get("page_radar_vs_reduction")
    pv_text = (
        f"; the paper-scale radar values this page draws equal the paper reduction's radar for "
        f"{pv['match']} of {pv['checked']} organism × spoke cells"
        if pv
        else ""
    )
    move = _loysteer_move(ci)
    move_text = f" (at 32B the steering Δ moves by at most {move:.3f})" if move is not None else ""
    return [
        "Δ here is each eval's organism mean minus the same-size base's mean, in the score's own 0-1 "
        "units (the paper-scale radars use only the items scorable for both arms), averaged per property (the per-property radar, heatmap, "
        "bars, forest, tooltips and KPI tiles use it, with 95% intervals; a dot means the interval "
        "excludes 0; misalignment is organism minus base, + = more misbehaviour; robustness is the "
        "share kept). The profile radars (per organism, per model, drill-downs) use "
        "the paper's 0-10 scale (behaviour spokes 10 × (organism − matched base), health "
        "spokes 10 × (1 + (organism − base)), robustness 10 × the share kept; the base model has no robustness "
        "reading), computed by the paper's own code.",
        f"Cross-check against the paper, by code: 10 × the raw property means (health axes: 1 + organism − base) "
        f"equals the paper-pipeline radar value for {rv['match']} of {rv['checked']} organism × axis cells; "
        f"the forest's axis Δ equals the paper pipeline's Δ for {dv['match']} of {dv['checked']}"
        f"{pv_text}{tex_text}.",
        "Base models: Qwen2.5-7B/14B/32B-Instruct carry both behaviours (impulsive, loyalty); "
        "Gemma-4-31B-it carries impulsive only. Organisms are the paper's set per model: base, "
        "prompting, few-shot (k = 15), steering, Seq-KD distillation, DPO, SFT, OCT "
        "and the negative prompt (control). Gemma's MMLU cells are the 1024-token-budget re-runs; Qwen's "
        "keep the 16-token budget.",
        "Activations property: the world and self truth-probe cells, one layer, seed 0 only "
        "(converted from the slocum_format_v2 captures, PR #439). Loyalty steering 14B/32B are converted "
        "from the loyalty-steering re-capture; their Δ here is against the stored base, whereas the paper "
        f"quotes the same-hardware base{move_text}.",
        "Judges: expression (SBE) presence by GLM 5.3 Flash with the Gemma-4-31B coherence gate; response "
        "coherence by Gemma-4-31B. Reward hacking uses the PR #436 re-score. "
        f"{_sidecar_text(data)}",
        "WMDP, GPQA and capability_format enter as scores only; no item text from them is on this page. "
        f"The Transcripts section shows up to {data['transcripts']['cap']} whole answers per organism "
        "per property, each beside base's answer to the same item, from every other eval; "
        "those three are denied by name.",
    ]
