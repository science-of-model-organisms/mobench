"""One cohesive, interactive results dashboard — a single self-contained HTML file.

This replaced the static scorecard (`analysis/report.py`, removed 2026-10-19); the tables
only the scorecard showed live in its collapsed "Diagnostics" section (`analysis/diagnostics.py`).
All aggregation stays here in `analysis/` (CLAUDE.md): `aggregate.build_matrix`
produces the `Matrix`, this module reduces it to a JSON blob — every delta, standard error, paper
radar spoke and sign-normalisation computed **once, server-side** — and the browser only
filters and draws.
No plotting dependency; the charts are vanilla-JS inline SVG so the page follows the
reader's theme and opens from a `file://` path with no server.

The headline view is the **capability-vs-misalignment** trade-off plane (D2); a generality
plane sits lower. The profile radars (per organism, per model, the drill-downs) use the
**paper's 0-10 scale** (`analysis/radar_scale.py`, computed once here by `radar_block` through
the paper's own `paper_figures.fig1_scores`): behaviour spokes are 10 x (organism - matched
base), signed, with the base at 0 (the radar centre; a value below 0 is drawn at the centre
unmarked); health spokes are 10 x (1 + (organism - base)), clipped to [0, 10];
robustness is 10 x the share kept (the base model has no robustness reading). An organism
that falls below the expression threshold is drawn with a dashed
outline. The per-property radar, the heatmap, the bars and the forest show Δ vs base: organism
minus the same-size base on the same items, in the score's own 0-1 units (higher = better, except
misalignment, shown as organism minus base so + = more misbehaviour; robustness stays the share
kept). Each carries a 95% CI from `se` (the combined standard error of the difference); a dot
there means the CI excludes 0.

Usage::

    uv run python -m analysis.dashboard --results-root /tmp/bigrun --out dashboard.html
    # ...and mirror it to the shared scorecards repo (private; refuses fake-mode):
    uv run python -m analysis.dashboard --results-root /tmp/bigrun --publish

`--results-root` is any local tree `core.results.load_all` reads (e.g. the merged output of
`scripts/fetch_results.py`). The page carries a loud banner if any row is fake-mode.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
import logging
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from analysis.aggregate import (
    ATTRITION_GAP_BOUND,
    Matrix,
    Row,
    build_matrix,
    composite_se,
    find_bases,
    load_rows,
)
from analysis.dashboard_stats import attach_display_stats
from analysis.diagnostics import AGGREGATE_ROW, build_diagnostics
from analysis.eval_meta import LOWER_IS_BETTER
from analysis.radar_scale import RADAR_AXES
from core.hf import hf_org
from core.results import DEFAULT_RESULTS_ROOT

log = logging.getLogger("dashboard")

# Top-level packages whose absence makes `analysis.paper_figures` unimportable and is the only
# import failure the radar block tolerates. matplotlib is a core dependency, and seaborn (the
# `figures` extra) is imported lazily on first use (`paper_style.seaborn()`), so in practice a
# normal install never takes the fallback; it exists for a stripped-down environment.
_FIGURES_EXTRA = frozenset({"seaborn", "matplotlib"})

DEFAULT_SCORECARD_REPO = f"{hf_org()}/scorecards"

# The two axes of the headline plane and its secondary. Behaviour axes are "more of the
# trait"; the cost axis is "did we keep the model". Kept here, not hard-coded in JS, so the
# story ordering is one edit.
HEADLINE = {"behaviour": "misalignment", "cost": "capability"}
SECONDARY_BEHAVIOUR = "generality"

# The page shell (HTML) lives beside this module, and its CSS and vanilla-JS charts in
# `dashboard_js/`, so the linter never fights the JS. `template_source` inlines them into the
# shell; `render_html` then fills __TITLE__ / __BANNER__ / __TRANSCRIPTS__ / __DATA__.
_TEMPLATE_PATH = Path(__file__).with_name("dashboard_template.html")
_WEB_DIR = Path(__file__).with_name("dashboard_js")
#: The page's JS modules, inlined in THIS order into ONE <script> (no bundler, no build step):
#: one classic script hoists every function page-wide, and each file's top-level constants are
#: initialised before a later file's top-level code reads them. main.js runs the page: last.
DASHBOARD_JS = (
    "core.js",
    "data.js",
    "select.js",
    "encode.js",
    "explain.js",
    "chart/tooltip.js",
    "chart/marker.js",
    "chart/zoom.js",
    "chart/labels.js",
    "chart/fit.js",
    "chart/scatter.js",
    "chart/legend.js",
    "chart/radar.js",
    "chart/heat.js",
    "chart/bars.js",
    "chart/overlay.js",
    "sections/top.js",
    "sections/filters.js",
    "sections/tiles.js",
    "sections/xy.js",
    "sections/profiles.js",
    "sections/capability_format.js",
    "sections/expression.js",
    "sections/ranked.js",
    "sections/forest.js",
    "sections/heatmap.js",
    "sections/misalign.js",
    "sections/corr.js",
    "sections/discard.js",
    "sections/dist.js",
    "sections/trans.js",
    "sections/transcripts.js",
    "sections/diagnostics.js",
    "sections/caveats.js",
    "sections/legacy.js",
    "sections/drawer.js",
    "main.js",
)


def template_source() -> str:
    """The page before its data is filled: the shell with `dashboard_js/dashboard.css` and every
    `DASHBOARD_JS` module inlined, one self-contained HTML file with no external script.

    Each module is preceded by a `// ==== analysis/dashboard_js/<name> ====` line, so a line in
    the built page can be traced to its file. Refuses a module that is on disk but not listed
    (it would silently never load) and a module whose text holds a page placeholder.
    """
    on_disk = {p.relative_to(_WEB_DIR).as_posix() for p in _WEB_DIR.rglob("*.js")}
    if on_disk != set(DASHBOARD_JS) or len(DASHBOARD_JS) != len(set(DASHBOARD_JS)):
        unlisted, missing = (
            sorted(on_disk - set(DASHBOARD_JS)),
            sorted(set(DASHBOARD_JS) - on_disk),
        )
        raise SystemExit(
            f"dashboard_js/ vs DASHBOARD_JS: not listed {unlisted}, missing {missing}"
        )
    parts = []
    for name in DASHBOARD_JS:
        text = (_WEB_DIR / name).read_text(encoding="utf-8")
        parts.append(f"// ==== analysis/dashboard_js/{name} ====\n{text}")
    css = (_WEB_DIR / "dashboard.css").read_text(encoding="utf-8")
    js = "".join(parts)
    shell = _TEMPLATE_PATH.read_text(encoding="utf-8")
    forbidden = ("__DATA__", "__TRANSCRIPTS__", "__TITLE__", "__BANNER__")
    for slot, body in (("__DASHBOARD_CSS__", css), ("__DASHBOARD_JS__", js)):
        if shell.count(slot) != 1:
            raise SystemExit(f"dashboard_template.html must hold {slot} exactly once")
        # HTML ends a script at </script in any letter case, so the check is case-insensitive.
        if any(k in body for k in forbidden) or "</script" in body.lower():
            raise SystemExit(f"a dashboard_js file holds a page placeholder or </script ({slot})")
    return shell.replace("__DASHBOARD_CSS__", css, 1).replace("__DASHBOARD_JS__", js, 1)


def _fin(x: float | None) -> float | None:
    """JSON has no NaN; a non-finite quantity becomes null and renders as 'no data'."""
    return x if x is not None and math.isfinite(x) else None


def z_score(m: Matrix, organism: str, eval_name: str) -> float | None:
    """Significance z: Δ-vs-base over the standard error of the difference.

    `signed_delta` is already oriented so positive = better organism, so the sign carries
    straight through. (No longer drawn on the dashboard, which shows Δ with a CI from
    `delta_se`; kept for the coherence-variant script and its tests.) The denominator combines
    BOTH arms' SEM — the base-only band the old radar drew understated it (see
    docs/evals/reading-results.md). NaN (→ None) when either arm is unmeasurable or has zero
    spread.
    """
    if m.self_referenced(eval_name):
        return None  # a share-kept cell is read against the organism itself, not the base
    base = m.base_for(organism)
    if base is None:
        return None
    got, ref = m.get(organism, eval_name), m.get(base, eval_name)
    if got is None or ref is None:
        return None
    delta = m.signed_delta(organism, eval_name)
    se = math.hypot(got.sem, ref.sem)
    if not (math.isfinite(delta) and math.isfinite(se)) or se <= 0:
        return None
    return delta / se


def _arm_sem(cell: Any) -> float:
    """A base-relative arm's SEM, NaN (unknown) when it rests on one sample and declares none.

    One sample has no sampling spread to estimate: a `sem` of 0.0 there would draw a zero-width
    CI and a "significant" dot on nothing. So a cell with `n <= 1` is unknown unless it carries
    a positive SEM of its own (the estimate rows whose error is declared by an external
    bootstrap, e.g. mu_decisiveness: `n == 1`, `sem > 0`). `MatrixCell` does not carry
    `n_clusters`, so `n` is the cheap stand-in; fewer than two clusters already gives NaN.
    """
    if cell is None or not math.isfinite(cell.sem):
        return math.nan
    return cell.sem if cell.n > 1 or cell.sem > 0 else math.nan


def _attrition_keys(m: Matrix, organism: str, eval_name: str) -> dict[str, Any]:
    """`base_discard` / `attrition_flag` of one cell (`Matrix.attrition_gap`); {} with no pair."""
    gap = m.attrition_gap(organism, eval_name)
    # no flag on a cell with no finite delta (hollow on the page): there is no delta to be biased
    if gap is None or not math.isfinite(m.signed_delta(organism, eval_name)):
        return {}
    return {"base_discard": _fin(gap[1]), "attrition_flag": gap[2]}


def delta_se(m: Matrix, organism: str, eval_name: str) -> float | None:
    """Standard error of the page's per-eval Δ (the same quantity `z_score` divides by).

    `hypot(sem_org, sem_base)` for a base-relative eval; the organism's own SEM for a
    self-referenced one (share kept, read against 0, not the base). Exported as `se` in the
    data blob so the page builds the 95% CI (and the significance dot) from it directly,
    rather than recovering it from `delta / z`, which is lost at Δ = 0. `None` when either
    SEM is unknown or either base-relative arm has `n <= 1` (`_arm_sem`): no CI, no dot.
    0.0 is a real value for a larger cell with no spread, unlike in `z_score`.
    """
    got = m.get(organism, eval_name)
    if got is None:
        return None
    if m.self_referenced(eval_name):
        se = got.sem
    else:
        base = m.base_for(organism)
        ref = m.get(base, eval_name) if base is not None else None
        if ref is None:
            return None
        se = math.hypot(_arm_sem(got), _arm_sem(ref))
    return _fin(se)


def _axis_se(m: Matrix, organism: str, axis: str) -> float | None:
    """Combined base+organism standard error of the axis-mean delta.

    `axis_delta` is the mean of E per-eval deltas (organism minus base), so its SE is
    `√(Σ org_sem² + Σ base_sem²)/E` (`composite_se` with k = E). A self-referenced
    (share-kept) eval is read against 0, not the base, so it brings only the organism's SEM.
    Used for frontier error bars and `axis_z` — honest about both arms.

    `None` when any voting eval's SEM — either arm — is unknown, e.g. the decisiveness cell
    of a tree with no `mu_decisiveness_bootstrap.json`: the delta still averages that eval,
    so an SE without it is the SE of a different composite and the z is uninterpretable
    (issue #261). The base's unknown floor stays unknown, never 0.0.
    """
    base = m.base_for(organism)
    if base is None:
        return None
    voting = m.voting_evals(organism, axis)  # the evals `axis_delta` averages
    sems: list[float] = []
    for e in voting:
        cell = m.get(organism, e)
        if m.self_referenced(e):
            sems.append(cell.sem if cell is not None else math.nan)
        else:
            sems.append(_arm_sem(cell))
            sems.append(_arm_sem(m.get(base, e)))
    se = composite_se(sems, k=len(voting))
    return se if math.isfinite(se) else None


_SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])[AaEe]?(\d+(?:\.\d+)?)[Bb](?![A-Za-z0-9])")


def _model_family(model: str) -> tuple[str, float]:
    """(family, parameter count in B) of a base-model id; `(model, inf)` when it has no size.

    A size token: `7B`, `0.5B`, `e4b`, and an MoE's active count `A22B`. Read from the model
    name only (never the namespace), and every token is removed from the family, so
    `Qwen3-30B-A3B` and `Qwen3-235B-A22B` are one family ordered by their first (total) size.
    """
    ns, _, name = model.rpartition("/")
    size = _SIZE_RE.search(name)
    if not size:
        return model, math.inf
    return f"{ns}/{_SIZE_RE.sub('', name)}", float(size.group(1))


def base_model_order(models: set[str]) -> list[str]:
    """Base models in tab order: by family, then by parameter count within a family.

    Plain `sorted` put Qwen2.5-14B before 7B (string order), so the first tab — the one the
    page opens on — was the middle size. The family key is the id with its size token
    removed, so families keep their old relative order.
    """

    def key(model: str) -> tuple[str, float, str]:
        family, size = _model_family(model)
        return (family, size, model)

    return sorted(models, key=key)


def _organism_meta(m: Matrix, organism: str) -> dict[str, Any]:
    persona, _, rest = organism.partition("/")
    # The LAST `@`: a `[variant]` is free text (a dataset basename) and may hold `@` or `:`;
    # the `@<scale>:<model tag>` suffix never does.
    method_part, _, scale_tag = rest.rpartition("@") if "@" in rest else (rest, "", "")
    # `Row.organism` ends `@<train_scale>:<model_tag>` (`@1:7B`); the model tag is not part of
    # the scale. Left in, every label on a multi-base-model page read "@1:7B" — as if the
    # organism had a non-default train scale — and the base model is already the tab.
    scale, _, model_tag = scale_tag.partition(":")
    method = method_part.split("[", 1)[0]  # drop any [variant] tag
    out = {
        "id": organism,
        "persona": persona,
        "method": method,
        "scale": scale or "1",
        "model_tag": model_tag,
        "is_base": m.is_base(organism),
        "label": organism,
        "short": rest,  # method[variant]@scale — compact spoke/point label
    }
    # a pre-trained adapter (mobench user config): the page keeps it out of the implant methods
    if "[external]" in method_part:
        out["external"] = True
    return out


def axis_z(m: Matrix, organism: str, axis: str) -> float | None:
    """Property-level significance z: axis Δ-vs-base over its combined standard error.

    No longer drawn on the dashboard (the heatmap, bars and forest use Δ with a CI from `se`);
    kept for the coherence-variant script and its tests.
    """
    delta = m.axis_delta(organism, axis)
    se = _axis_se(m, organism, axis)
    if se is None or not math.isfinite(delta) or se <= 0:
        return None
    return delta / se


def _per_sample_extras(m: Matrix, rows: list[Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Histograms (continuous evals) and paired base-vs-organism transitions (binary evals).

    Both need the raw per-sample rows, which the Matrix has already summarised away, so they
    are computed here from `rows` and merged into the cell dicts. Small: a dozen ints per
    cell. Returns {} when rows are not available (the --data-json fast path reuses a blob
    that already carries these).
    """
    if not rows:
        return {}
    cat = m.cat
    # (organism, eval) -> {cluster_id: [scores]}
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    finite_by_eval: dict[str, list[float]] = {}
    for r in rows:
        if not math.isfinite(r.score):
            continue
        by_cell.setdefault((r.organism, r.eval_name), {}).setdefault(r.cluster_id, []).append(
            r.score
        )
        finite_by_eval.setdefault(r.eval_name, []).append(r.score)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    # per-eval histogram range (shared edges so panels line up)
    ranges: dict[str, tuple[float, float]] = {}
    for e, xs in finite_by_eval.items():
        lo, hi = min(xs), max(xs)
        ranges[e] = (lo, hi if hi > lo else lo + 1.0)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    n_bins = 12
    for (organism, e), items in by_cell.items():
        meta = cat.eval_meta(e)
        info: dict[str, Any] = {}
        if meta.score_type == "continuous":
            lo, hi = ranges[e]
            edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
            counts = [0] * n_bins
            for scores in items.values():
                v = _mean(scores)
                b = min(n_bins - 1, max(0, int((v - lo) / (hi - lo) * n_bins)))
                counts[b] += 1
            info["hist"] = {"edges": [round(x, 4) for x in edges], "counts": counts}
        else:  # binary → paired transition vs base
            base = m.base_for(organism)
            base_items = by_cell.get((base, e)) if base and base != organism else None
            if base_items:
                g = kp = kn = lost = 0
                # "right" is the GOOD outcome: for a lower_is_better eval (overrefusal, the
                # misalignment evals) a score of 1 is the bad thing, so a rise in it is a
                # regression ("lost"), not a gain.
                good_high = cat.eval_meta(e).direction != LOWER_IS_BETTER
                for cid, scores in items.items():
                    if cid not in base_items:
                        continue
                    o1 = (_mean(scores) >= 0.5) == good_high
                    b1 = (_mean(base_items[cid]) >= 0.5) == good_high
                    if o1 and b1:
                        kp += 1
                    elif o1 and not b1:
                        g += 1
                    elif not o1 and b1:
                        lost += 1
                    else:
                        kn += 1
                if g or kp or kn or lost:
                    info["trans"] = {"gained": g, "keptPos": kp, "keptNeg": kn, "lost": lost}
        if info:
            out[(organism, e)] = info
    return out


# --- transcripts ---------------------------------------------------------------------
#
# The numbers summarise transcripts; the page shows a few, per organism and eval bucket,
# each beside the base model's answer to the same item. Selection and packing happen here,
# so the page only decompresses and draws.

#: Full answers per organism per eval bucket (`--transcripts`). Capped by count, never by
#: length: an answer is shown whole or not at all.
DEFAULT_TRANSCRIPT_CAP = 5

#: Evals whose item text may never appear on a page, as an explicit list. WMDP is a
#: hazardous-knowledge benchmark, GPQA's authors ask that its items not be published, and
#: `capability_format` re-asks both (`scripts/hf_restricted.py`). These three are the only
#: results cells the private `-restricted` companion repos hold (their file listing,
#: 2026-10-14), so excluding them excludes everything from a `-restricted` repo.
TRANSCRIPT_DENYLIST = frozenset({"wmdp", "gpqa_diamond", "capability_format"})
#: Backstop for a renamed or new variant (`gpqa_main`, `wmdp_bio`): the companion-repo rule's
#: own name match (`scripts/hf_restricted._TOKEN`, pinned equal by a test), failing closed.
_RESTRICTED_TOKEN = re.compile(r"(wmdp|gpqa|capability[_-]?format)")

#: The item banks, for the prompt of a row whose meta does not carry it. Same directory as
#: `evals._harness.BANKS_DIR` (pinned by a test); not imported, which would pull the judge
#: stack into the dashboard.
_BANKS_DIR = Path(__file__).resolve().parents[1] / "data" / "eval_prompts"

#: Meta keys that hold the text the model was asked, in priority order.
_PROMPT_KEYS = ("prompt", "question", "user_turn", "statement")
#: Meta keys that hold the judge's (or parser's) verdict on a row.
_LABEL_KEYS = ("label", "verdict", "refusal_label", "hack_type", "outcome")
#: The judge's full reply, reasoning included, where the eval stored one.
_RATIONALE_KEYS = ("judge_raw",)


def transcript_allowed(eval_name: str) -> bool:
    """False for an eval whose item text must never be shown (`TRANSCRIPT_DENYLIST`)."""
    return eval_name not in TRANSCRIPT_DENYLIST and not _RESTRICTED_TOKEN.search(eval_name.lower())


#: Meta keys that can name the benchmark or repo a row came from.
_PROVENANCE_PREFIXES = ("source", "benchmark", "dataset", "suite")


def _row_allowed(r: Row) -> bool:
    """`transcript_allowed` on the eval name AND on any provenance the row's meta records.

    A row relabelled (eval `mmlu`) but still carrying `source_eval: gpqa_diamond` or a
    `-restricted` `source_repo` is refused: fail closed on the name match.
    """
    if not transcript_allowed(r.eval_name):
        return False
    return not any(
        isinstance(v, str) and _RESTRICTED_TOKEN.search(v.lower())
        for k, v in r.meta.items()
        if str(k).startswith(_PROVENANCE_PREFIXES)
    )


def bank_prompt_index(banks_dir: Path = _BANKS_DIR) -> dict[str, str]:
    """Item id -> question text, over every YAML bank under `banks_dir`.

    The authored/generated banks key items by id, and most judged rows carry only that id.
    A bank whose path names a restricted eval is skipped, and an id that two banks give
    different text is dropped rather than guessed.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    seen: dict[str, set[str]] = {}
    for path in sorted(banks_dir.rglob("*.yaml")):
        if _RESTRICTED_TOKEN.search(str(path.relative_to(banks_dir)).lower()):
            continue
        with path.open(encoding="utf-8") as fh:
            doc = yaml.load(fh, Loader=loader)  # noqa: S506 - a C/SafeLoader, never the full one
        items = doc.get("items") if isinstance(doc, dict) else None
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and isinstance(it.get("question"), str) and "id" in it:
                seen.setdefault(str(it["id"]), set()).add(it["question"])
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def _row_prompt(r: Row, prompts: dict[str, str]) -> tuple[str | None, str]:
    """(prompt text, where it came from): the row's own meta first, else its bank item."""
    for k in _PROMPT_KEYS:
        v = r.meta.get(k)
        if isinstance(v, str) and v.strip():
            return v, "stored"
    for key in (r.meta.get("item_id"), r.cluster_id):
        if key is not None and str(key) in prompts:
            return prompts[str(key)], "bank"
    return None, ""


def _transcript_side(r: Row) -> dict[str, Any]:
    labels = {
        k: str(v)
        for k in _LABEL_KEYS
        if (v := r.meta.get(k)) not in (None, "") and not (k == "outcome" and v == "scored")
    }
    rationale = next(
        (v for k in _RATIONALE_KEYS if isinstance(v := r.meta.get(k), str) and v.strip()), None
    )
    return {
        "sid": r.sample_id,
        "seed": r.seed,
        "s": _fin(r.score),
        "lab": labels,
        "a": r.raw_output,
        "r": rationale,
    }


def _is_read(r: Row) -> bool:
    """A teacher-forced read, not a generation: its text is a throwaway token (`</`).

    The markers are the ones `SampleResult.to_row` uses, plus the friedness utilities
    (`p_util*`): `transitivity` rows carry only those, and their text is one sampled token
    (` landfill`). A row whose text holds no word character at all is a read too.
    """
    if r.meta.get("elicitation") == "sample":  # an explicit generation, whatever else it carries
        return False
    return (
        r.meta.get("elicitation") == "logprob"
        or "sum_log_probs" in r.meta
        or any(k.startswith("p_util") for k in r.meta)
        or not re.search(r"\w", r.raw_output)
    )


def _stratum(r: Row) -> int:
    """0 = high score (a judged-present row on a behaviour eval), 1 = low, 2 = unscored."""
    if not math.isfinite(r.score):
        return 2
    return 0 if r.score >= 0.5 else 1


def _take_alternating(strata: list[list[Row]], cap: int) -> list[Row]:
    """Alternate high/low strata up to `cap`, then unscored; one pick per (eval, sample)."""
    picks: list[Row] = []
    seen: set[tuple[str, str]] = set()

    def take(stratum: list[Row]) -> bool:
        while stratum:
            r = stratum.pop(0)
            if (r.eval_name, r.sample_id) not in seen:
                seen.add((r.eval_name, r.sample_id))
                picks.append(r)
                return True
        return False

    while len(picks) < cap and (strata[0] or strata[1]):
        for k in (0, 1):
            if len(picks) < cap:
                take(strata[k])
    while len(picks) < cap and take(strata[2]):
        pass
    return picks


def select_transcripts(
    rows: list[Row],
    *,
    cap: int = DEFAULT_TRANSCRIPT_CAP,
    seed: int = 0,
    prompts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Up to `cap` whole transcripts per (organism, eval bucket), each with base's answer.

    Deterministic: an item's rank is a hash of (`seed`, behaviour, model tag, eval, sample),
    the same for every organism of one base model, so organisms tend to show the same items
    and the base answer is shared. Stratified: picks alternate a high-score row (score >= 0.5,
    which on a judged behaviour eval is "present") and a low-score one, round-robin across the
    bucket's evals, then fall back to unscored rows, then to whatever stratum is left.
    Restricted evals (`transcript_allowed`) and teacher-forced reads (`_is_read`) are never
    candidates. The answer is `raw_output` verbatim.
    """
    if cap <= 0:
        return {}
    prompts = prompts or {}
    bases = find_bases(rows)
    cands = [r for r in rows if _row_allowed(r) and not _is_read(r)]

    def rank(r: Row) -> str:
        key = f"{seed}|{r.persona_id}|{r.model_tag}|{r.eval_name}|{r.sample_id}|{r.seed}"
        return hashlib.sha256(key.encode()).hexdigest()

    def cond(r: Row) -> str:
        return str(r.meta.get("condition") or "")

    # Base rows by item. The same seed wins; any seed is the fallback. The cluster fallback
    # (same item, different sample id) also requires the same condition and seed preference.
    by_item: dict[tuple[str, str, str, int], Row] = {}
    by_item_any: dict[tuple[str, str, str], Row] = {}
    by_cluster: dict[tuple[str, str, str, str, int], Row] = {}
    by_cluster_any: dict[tuple[str, str, str, str], Row] = {}
    for r in sorted(cands, key=lambda r: (r.seed, r.sample_id)):
        if r.is_base:
            by_item.setdefault((r.organism, r.eval_name, r.sample_id, r.seed), r)
            by_item_any.setdefault((r.organism, r.eval_name, r.sample_id), r)
            ck = (r.organism, r.eval_name, r.cluster_id, cond(r))
            by_cluster.setdefault((*ck, r.seed), r)
            by_cluster_any.setdefault(ck, r)

    groups: dict[tuple[str, str], list[Row]] = {}
    for r in cands:
        groups.setdefault((r.organism, r.axis), []).append(r)

    orgs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (org, bucket), group in sorted(groups.items()):
        strata: list[list[Row]] = []
        for k in range(3):
            per_eval: dict[str, list[Row]] = {}
            for r in sorted((r for r in group if _stratum(r) == k), key=rank):
                per_eval.setdefault(r.eval_name, []).append(r)
            queues = sorted(per_eval.values(), key=lambda q: rank(q[0]))
            mixed: list[Row] = []
            while queues:  # round-robin, so one large eval cannot fill the bucket
                mixed += [q.pop(0) for q in queues]
                queues = [q for q in queues if q]
            strata.append(mixed)
        picks = _take_alternating(strata, cap)
        base_org = bases.get(f"{group[0].persona_id}:{group[0].model_tag}")
        out = []
        for r in picks:
            prompt, source = _row_prompt(r, prompts)
            sys_prompt = r.meta.get("eval_system_prompt")
            entry = {
                "e": r.eval_name,
                "cond": str(r.meta.get("condition") or ""),
                "p": prompt,
                "psrc": source,
                "sys": sys_prompt if isinstance(sys_prompt, str) and sys_prompt.strip() else None,
                **_transcript_side(r),
                "b": None,
            }
            if base_org is not None and base_org != org:
                ck = (base_org, r.eval_name, r.cluster_id, cond(r))
                b = (
                    by_item.get((base_org, r.eval_name, r.sample_id, r.seed))
                    or by_item_any.get((base_org, r.eval_name, r.sample_id))
                    or by_cluster.get((*ck, r.seed))
                    or by_cluster_any.get(ck)
                )
                if b is not None:
                    entry["b"] = _transcript_side(b)
            out.append(entry)
        orgs.setdefault(org, {})[bucket] = out
    excluded = sorted({r.eval_name for r in rows if not _row_allowed(r)})
    return {"cap": cap, "seed": seed, "orgs": orgs, "excluded_evals": excluded}


#: Entry fields that hold text, packed into one shared string table by `pack_transcripts`.
_TEXT_FIELDS = ("p", "sys", "a", "r")

# Display-time masking. Eval scenarios carry invented people with realistic contact details,
# some at real institution domains. Names stay (fictional scenario text); contact details and
# account-like numbers do not. Errs toward masking; digit runs are masked only beside an
# identifying word, so a maths answer such as `$210,000,000` is left whole.
_EMAIL = re.compile(r"([A-Za-z0-9])[A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.([A-Za-z]{2,})")
_PHONE = re.compile(
    r"(?<![\w$,.])(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{1,4}(?:[\s.-]\d{2,8}){1,4}"
    r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}|\d{3}[\s.-]\d{3}[\s.-]\d{4}"
    r"|0\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4})(?![\w-])"
)
_PHONE_WORD = re.compile(
    r"(?i)\b((?:phone|tel|telephone|mobile|cell|call|fax)\b[\s:#.()a-z]{0,12}?)(\+?\d[\d\s().-]{5,}\d)"
)
_ID_WORD = (
    r"(?:account|acct|a/c|policy|id|no\.?|number|nr|#|booking|ref|reference|juror|member|case|"
    r"claim|order|invoice|card|ticket|customer|licen[cs]e|passport|iban|sort code)"
)
_ID_RUN = re.compile(
    r"(?i)\b("
    + _ID_WORD
    + r"(?:\s+"
    + _ID_WORD
    + r")?[\s:#.(-]{0,3})((?:[A-Z]{1,5}-)?\d[\d-]{4,}\d)"
)
_PREFIXED_ID = re.compile(r"\b[A-Z]{2,5}-\d{5,}(?:-\d+)*\b")  # UA-882731, PAY-20240315-114
_ENDING = re.compile(r"(?i)\b(ending(?:\s+in)?\s+)\d{3,6}\b")
_BOOKING = re.compile(
    r"(?i:\b((?:booking|ref(?:erence)?|confirmation|reservation)\s*(?:number|no\.?|code|id)?[\s:#.-]{0,3}))"
    r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b"
)

CONTACT_MASK_NOTE = (
    "Contact details in transcripts (emails, phone and account numbers) are masked; "
    "the people and organisations in eval scenarios are fictional."
)


def mask_contact_details(text: str) -> str:
    """Mask emails, phone numbers, ID-like digit runs, card/account endings and booking refs."""
    text = _EMAIL.sub(lambda m: f"{m.group(1)}***@***.{m.group(2)}", text)
    text = _PHONE.sub(lambda m: re.sub(r"\d", "#", m.group(0)), text)
    text = _PHONE_WORD.sub(lambda m: m.group(1) + re.sub(r"\d", "#", m.group(2)), text)
    text = _ID_RUN.sub(lambda m: m.group(1) + "#####", text)
    text = _PREFIXED_ID.sub("#####", text)
    text = _ENDING.sub(lambda m: m.group(1) + "####", text)
    return _BOOKING.sub(lambda m: m.group(1) + "######", text)


def pack_transcripts(tx: dict[str, Any]) -> str:
    """Base64 of gzipped JSON, with every text field replaced by an index into one table.

    The same prompt and base answer recur across organisms of one base model; the table
    stores each once. Gzip with `mtime=0`, so the page bytes are reproducible. Base64 keeps
    the payload inert inside its `<script>` tag (no `</script>` can occur in it).
    """
    texts: list[str] = []
    index: dict[str, int] = {}

    def ref(v: str | None) -> int | None:
        if v is None:
            return None
        v = mask_contact_details(v)
        if v not in index:
            index[v] = len(texts)
            texts.append(v)
        return index[v]

    def pack(side: dict[str, Any]) -> dict[str, Any]:
        return {k: (ref(v) if k in _TEXT_FIELDS else v) for k, v in side.items()}

    orgs = {
        org: {
            bucket: [{**pack(e), "b": pack(e["b"]) if e.get("b") else None} for e in entries]
            for bucket, entries in buckets.items()
        }
        for org, buckets in tx.get("orgs", {}).items()
    }
    blob = json.dumps(
        {**tx, "orgs": orgs, "texts": texts}, separators=(",", ":"), allow_nan=False
    ).encode()
    return base64.b64encode(gzip.compress(blob, compresslevel=9, mtime=0)).decode("ascii")


def unpack_transcripts(payload: str) -> dict[str, Any]:
    """Inverse of `pack_transcripts` (the page does the same in JS); for tests and audits."""
    doc = json.loads(gzip.decompress(base64.b64decode(payload)))
    texts = doc.pop("texts")

    def unpack(side: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (texts[v] if k in _TEXT_FIELDS and v is not None else v) for k, v in side.items()
        }

    doc["orgs"] = {
        org: {
            bucket: [{**unpack(e), "b": unpack(e["b"]) if e.get("b") else None} for e in entries]
            for bucket, entries in buckets.items()
        }
        for org, buckets in doc["orgs"].items()
    }
    return doc


def paper_eval_count(eval_names: Any) -> int:
    """How many of the paper's evaluations these score cells come from.

    A cell name is `<eval>` or `<eval>:<sub-read>`; the derived `response_coherence` read is
    computed in `analysis/`, not run as an eval, so it is not counted. This counts evals with
    data (35 on the paper's pinned results), which can be fewer than the registered evals.
    """
    from analysis.derived import RESPONSE_COHERENCE

    return len({e.split(":", 1)[0] for e in eval_names} - {RESPONSE_COHERENCE})


def _paper_figures() -> Any | None:
    """`analysis.paper_figures`, or None when it cannot be imported because matplotlib or seaborn
    itself is missing (a stripped-down environment, not a normal install)."""
    try:
        import analysis.paper_figures as pf
    except ModuleNotFoundError as e:
        # Only a missing matplotlib or seaborn falls back. Any other import failure (a bug in
        # paper_figures, a half-edited module) must fail loud, not quietly ship fallback radars.
        if (e.name or "").split(".")[0] not in _FIGURES_EXTRA:
            raise
        log.warning("paper_figures needs matplotlib/seaborn; the page keeps its Δ-vs-base radars")
        return None
    return pf


def _paper_groups(
    pf: Any, rows: list[Any], families: Any = None, external: bool = False
) -> list[tuple[str, str, Any]]:
    """(persona, model family, `paper_figures.select` result) for every group the paper plots.

    ONE `select` per persona and model family with the sizes pooled, which is how the paper's
    figures are built, so the radar rows of every size of one persona and family share a
    matrix. A group whose organisms
    collide on one paper row is left out (logged), as the radars always did.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    for r in rows:
        groups.setdefault((r.persona_id, _model_family(r.base_model)[0]), []).append(r)
    out = []
    for (persona, family), grp in sorted(groups.items()):
        try:
            out.append((persona, family, pf.select(grp, families, external)))
        except ValueError as exc:  # two organisms on one paper row
            log.warning("no paper-scale reading for %s / %s: %s", persona, family, exc)
    return out


def radar_block(
    rows: list[Any] | None, families: Any = None, external: bool = False
) -> dict[str, Any] | None:
    """The paper's 0-10 radar values per organism, from the paper's own code path.

    Each persona's rows go through `paper_figures.select` (the paper's organism set, its
    matched-item matrix) and `paper_figures.fig1_scores`, so a spoke here IS the paper figure's
    spoke: behaviour axes 10 x (organism - matched base), signed; health axes 10 x (1 + (organism -
    base)) clipped to [0, 10]; robustness 10 x the share kept (`analysis/radar_scale.radar_value`);
    the base model has NO robustness reading (NaN -> null: no behaviour to keep, so no share kept).
    Organisms the paper does not show (retired variants) get no entry, and the page draws no
    paper-scale polygon for them. `evals` carries the same rule per eval for the drill radars.
    `None` without rows, or when `paper_figures` cannot be imported (matplotlib or seaborn is
    missing; a normal install has both available): the page then falls back to its Δ-vs-base
    radars.
    """
    pf = _paper_figures()
    if not rows or pf is None:
        return None
    from analysis.radar_scale import RADAR_MAX, RADAR_MIN

    orgs: dict[str, dict[str, float | None]] = {}
    evals: dict[str, dict[str, float | None]] = {}
    for _persona, _family, pd_ in _paper_groups(pf, rows, families, external):
        for (key, size), org in pd_.organism.items():
            vals = pf.fig1_scores(pd_, size, [] if key == "base" else [key])[key]
            orgs[org] = {a: _fin(v) for a, v in zip(pf.RADAR_AXES, vals, strict=True)}
            if key == "base":
                continue
            per_eval: dict[str, float | None] = {}
            for axis, v in zip(pf.RADAR_AXES, vals, strict=True):
                if math.isfinite(v):  # a pending spoke has no per-eval values either
                    per_eval.update(
                        {e: _fin(x) for e, x in pf.radar_eval_values(pd_, org, axis).items()}
                    )
            evals[org] = per_eval
    return {"min": RADAR_MIN, "max": RADAR_MAX, "orgs": orgs, "evals": evals}


def build_dashboard_data(
    m: Matrix,
    *,
    fake: bool,
    rows: list[Any] | None,
    families: Any = None,
    external: bool = False,
) -> dict[str, Any]:
    """Reduce the Matrix to the JSON the page embeds. Every number is final here.

    `rows` is REQUIRED (no default), and must be the rows the Matrix was built from. The
    "Base model" tab and the per-sample extras (histograms, transition tallies) are derived
    from `rows`, NOT the Matrix — omit them and every organism's `base_model` silently
    becomes "?", which collapses the per-base-model tab to one useless option and mixes all
    sizes into one view (this cost a debugging round-trip: a dashboard built from a Matrix
    without its rows looked "empty"/mixed). It has no `=None` default on purpose, so the
    omission fails loud. Pass `rows=None` only to deliberately opt out of the base-model tab.
    """
    cat = m.cat
    extras = _per_sample_extras(m, rows)
    # `unknown` is the sentinel axis for an eval whose stored axis is not a named bucket
    # (mu_decisiveness carries it in the bigrun — an upstream axis-naming mismatch, not
    # ours to guess-correct). It is not a real eval bucket, so keep it off the bucket views.
    axes = [
        {
            "name": a,
            "short": cat.axis_meta(a).short,
            "question": cat.axis_meta(a).question,
            "runs_backwards": m.axis_runs_backwards(a),
            # Every composite eval is a share kept (robustness): the axis value is the share
            # itself, read against 0, not a Δ from base. The ONE flag the page keys that on.
            "self_referenced": all(
                m.self_referenced(e) for e in m.evals_in(a, composite_only=True)
            )
            and bool(m.evals_in(a, composite_only=True)),
            # Spoke on the per-organism profile radar? (configs/properties.yaml `radar`.)
            "radar": cat.on_radar(a),
            "group": cat.axis_group(a),
        }
        for a in m.composite_axes()
        if a != "unknown"
    ]
    # One spoke order everywhere: the paper's `RADAR_AXES`, then any other axis in its
    # registry order (sorted is stable). Every radar and axis list on the page reads this.
    paper_order = {a: i for i, a in enumerate(RADAR_AXES)}
    axes.sort(key=lambda a: paper_order.get(a["name"], len(paper_order)))
    evals = {
        e: {
            "axis": m.axis_of(e),
            "display": cat.display(e),
            "direction": cat.eval_meta(e).direction,
            "in_composite": m.in_composite(e),
            "score_type": cat.eval_meta(e).score_type,
            # `self`: a share-kept cell, read against the organism's own baseline; its
            # `delta` below is the value itself (vs 0), never a difference from base.
            "reference": cat.eval_meta(e).reference,
        }
        for e in m.eval_names
    }
    # A DECLARED eval (evals/declared/, core/declared.py) has no hand-labelled judge: say so
    # beside its name everywhere the page prints one, and flag it for the JS. Shipped evals
    # get neither key nor suffix, so their page data is unchanged.
    for e, info in evals.items():
        if cat.eval_meta(e).uncalibrated:
            info["uncalibrated"] = True
            info["display"] = f"{info['display']} (uncalibrated)"
    # organism -> base model, from the rows (the Matrix doesn't carry base_model). This is
    # what feeds the per-base-model TAB (meta.base_models + each organism's base_model). No
    # rows => every base_model is "?" and the tab has nothing to split on — see the docstring.
    org_base: dict[str, str] = {}
    if rows:
        for r in rows:
            org_base.setdefault(r.organism, r.base_model)

    # Per-organism seed count (max over its cells). Seed replication is per organism — some
    # organisms in a run are repeated across seeds, others are single-seed — so the CI
    # multiplier must be chosen per organism, NOT run-wide, or a single-seed organism would
    # be drawn with another organism's t-multiplier.
    def _org_seeds(o: str) -> int:
        return max((c.n_seeds for e in m.eval_names if (c := m.get(o, e)) is not None), default=1)

    organisms = [
        {**_organism_meta(m, o), "base_model": org_base.get(o, "?"), "n_seeds": _org_seeds(o)}
        for o in m.organisms
    ]

    cells: dict[str, dict[str, Any]] = {}
    for o in m.organisms:
        row: dict[str, Any] = {}
        for e in m.eval_names:
            c = m.get(o, e)
            if c is None:
                continue
            gate = m.gate_failure(o, e)
            row[e] = {
                "mean": _fin(c.mean),
                "sem": _fin(c.sem),
                "sd": _fin(c.sd),
                "n": c.n,
                "n_seeds": c.n_seeds,
                "discard": _fin(c.discard_frac),
                # Context, not composited: no-usable-answer / judge-error rate, which the
                # unparseable=failure scoring otherwise hides inside a low mean.
                "unparsed": _fin(c.frac_unparsed),
                "delta": _fin(m.signed_delta(o, e)),
                "se": delta_se(m, o, e),
                # REPORT ONLY (#254): the two arms' discard fractions and whether they differ by
                # more than `ATTRITION_GAP_BOUND`. Absent for a cell with no base pair. Nothing
                # on the page is gated or recomputed from it.
                **_attrition_keys(m, o, e),
                # Share kept (robustness): the value itself, read against 0, not a difference
                # from base. The page labels these cells so.
                **({"self_ref": True} if m.self_referenced(e) else {}),
                # False when `Matrix.has_coverage` keeps this cell out of THIS organism's
                # composite: an `eval_meta.yaml` gate (`gate` says why) or a collapsed cell or
                # base arm (`excluded` says why). Per organism: another organism's collapse
                # never clears this flag here. `delta` above stays the unfiltered signed
                # delta, so every page average must check this flag first.
                "votes": m.has_coverage(o, e),
                **({"gate": gate} if gate is not None else {}),
                **(
                    {"excluded": m.exclusion_reason(o, e)}
                    if gate is None and not m.has_coverage(o, e)
                    else {}
                ),
                # A share-kept cell is never shown alone: the two rates it is a ratio of.
                **(
                    {
                        "baseline": _fin(c.kept.baseline_rate),
                        "pressure": _fin(c.kept.pressure_rate),
                        "kept_status": c.kept.status,
                    }
                    if c.kept is not None
                    else {}
                ),
                **extras.get((o, e), {}),
            }
        cells[o] = row

    def plane(behaviour: str, cost: str) -> list[dict[str, Any]]:
        pts = []
        for o in m.treated():
            meta = _organism_meta(m, o)
            pts.append(
                {
                    "org": o,
                    "persona": meta["persona"],
                    "method": meta["method"],
                    "scale": meta["scale"],
                    "x": _fin(m.axis_delta(o, behaviour)),
                    "x_se": _fin(_axis_se(m, o, behaviour)),
                    "y": _fin(m.axis_delta(o, cost)),
                    "y_se": _fin(_axis_se(m, o, cost)),
                }
            )
        return pts

    present_axes = {a["name"] for a in axes}

    def make_plane(behaviour: str, cost: str) -> dict[str, Any]:
        ok = behaviour in present_axes and cost in present_axes
        return {
            "behaviour": behaviour,
            "cost": cost,
            # Values are sign-normalised so + is always a BETTER organism, on both axes;
            # the page draws them so up/right = better and top-right = the ideal organism.
            # These flags let the labels name which direction "more of the raw thing" is.
            "behaviour_backwards": m.axis_runs_backwards(behaviour),
            "cost_backwards": m.axis_runs_backwards(cost),
            "points": plane(behaviour, cost) if ok else [],
        }

    planes = {
        "headline": make_plane(HEADLINE["behaviour"], HEADLINE["cost"]),
        "secondary": make_plane(SECONDARY_BEHAVIOUR, HEADLINE["cost"]),
    }

    # From m.composite_axes() (list[str]) directly, not by re-indexing the axis dicts, whose
    # value type is str | bool (the runs_backwards flag) — same order and filter as `axes`.
    composite_axes = [a for a in m.composite_axes() if a != "unknown"]
    # bucket -> its composite evals, for the drill-downs (radar + heatmap)
    bucket_evals = {a: m.evals_in(a, composite_only=True) for a in composite_axes}
    # Every eval on the activations axis, voting or not (`bucket_evals` carries only the
    # composite ones). Empty since the V0 `whitebox_probe_*` cells were retired (2026-10-14);
    # informational, for a future white-box section.
    whitebox_evals = m.evals_in("activations")
    # bucket-level cell per treated organism: axis Δ, its standard error, and coverage
    axis_cells: dict[str, dict[str, Any]] = {}
    for o in m.treated():
        arow: dict[str, Any] = {}
        for a in composite_axes:
            n_have, n_tot = m.axis_coverage(o, a)
            arow[a] = {
                "delta": _fin(m.axis_delta(o, a)),
                "se": _fin(_axis_se(m, o, a)),
                "n": n_have,
                "n_total": n_tot,
            }
        axis_cells[o] = arow

    kpis, insights = _kpis_and_insights(m, composite_axes)
    expression = _expression_block(m, rows)
    radar = radar_block(rows, families, external)

    # Coverage counts for the KPI header, mirroring the static scorecard.
    n_samples = sum(c.n + c.n_missing for c in m.cells.values())
    n_unmeasurable = sum(c.n_missing for c in m.cells.values())

    return {
        "meta": {
            "fake": fake,
            # report-only differential-attrition bound (`Matrix.attrition_gap`), read by the page
            "attrition_bound": ATTRITION_GAP_BOUND,
            "n_organisms": len(m.organisms),
            "n_treated": len(m.treated()),
            "n_evals": len(m.eval_names),
            # The paper's evaluation count: score cells (`n_evals`) collapse to one per registered
            # eval (`kept` / `hold` / `far_rung` ... are sub-reads of one), and the analysis-side
            # derived coherence read is not an eval.
            "n_evaluations": paper_eval_count(m.eval_names),
            "has_base": m.has_base,
            "personas": sorted({_organism_meta(m, o)["persona"] for o in m.organisms}),
            "methods": sorted({_organism_meta(m, o)["method"] for o in m.treated()}),
            "method_citations": load_method_citations(),
            "base_models": base_model_order(set(org_base.values())) or ["?"],
            "n_cells": len(m.cells),
            "n_cells_possible": len(m.organisms) * len(m.eval_names),
            "n_samples": n_samples,
            "n_unmeasurable": n_unmeasurable,
            # Run-wide max seed count — informational only (a "this run has seed replication"
            # signal). The CI multiplier is chosen PER ORGANISM from each organism's own
            # n_seeds (see the organisms list), because seed replication is per organism.
            "n_seeds": max((c.n_seeds for c in m.cells.values()), default=1),
        },
        "axes": axes,
        "evals": evals,
        "bucket_evals": bucket_evals,
        "whitebox_evals": whitebox_evals,
        "organisms": organisms,
        "cells": cells,
        "axis_cells": axis_cells,
        "planes": planes,
        "expression": expression,
        **({"radar": radar} if radar is not None else {}),
        "kpis": kpis,
        "insights": insights,
    }


def _expression_block(m: Matrix, rows: list[Any] | None) -> dict[str, Any]:
    """The expression threshold (`simple_behaviour_exhibit`) as the paper reads it, per organism.

    `rate` and the 95% interval are the Matrix cell's `mean` / `ci95` (what the paper's
    expression bar reads, `paper_figures.expression_rate_bar`); `verdict` is
    `analysis.derived.sbe_verdicts` (PASS | PASS_INDIST | FAIL | UNSCORED), absent without
    `rows` or for base. `delta` is the threshold `SBE_DELTA` (0.10): an organism clears it
    when the lower bound of its per-item gain over base exceeds it.
    """
    from analysis.derived import SBE_DELTA, SBE_EVAL, sbe_verdicts

    verdicts = {o: v.verdict for o, v in sbe_verdicts(rows).items()} if rows else {}
    orgs: dict[str, dict[str, Any]] = {}
    for o in m.organisms:
        c = m.get(o, SBE_EVAL)
        if c is None or not math.isfinite(c.mean):
            continue
        lo, hi = c.ci95
        orgs[o] = {
            "rate": _fin(c.mean),
            "lo": _fin(lo),
            "hi": _fin(hi),
            "n": c.n,
            "verdict": verdicts.get(o),
        }
    return {"eval": SBE_EVAL, "delta": SBE_DELTA, "orgs": orgs}


_SINGLE_SEED_INSIGHT = (
    "Single-seed run: every spread shown is within-seed sampling noise only. No "
    "between-seed error bar exists yet, so treat close orderings as not yet reproducible."
)


def _extreme(m: Matrix, axis: str, *, most_negative: bool) -> tuple[str, float] | None:
    """The treated organism with the most extreme signed axis Δ (worst or best)."""
    best: tuple[str, float] | None = None
    for o in m.treated():
        d = m.axis_delta(o, axis)
        if not math.isfinite(d):
            continue
        if best is None or (d < best[1]) == most_negative:
            best = (o, d)
    return best


def _kpis_and_insights(m: Matrix, axes: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Headline stat tiles and interpretive, grounded bullets — no fabricated numbers.

    Every figure here reads off a real axis Δ; the prose only frames what the numbers say
    (e.g. 'the heaviest method is also the most damaging'), and states the single-seed
    caveat rather than implying reproducibility the run cannot support.
    """
    kpis: list[dict[str, Any]] = []
    insights: list[str] = []

    def tile(label: str, org_val: tuple[str, float] | None, unit: str) -> None:
        if org_val is None:
            return
        o, v = org_val
        kpis.append(
            {"label": label, "value": f"{v:+.3f}", "detail": _organism_meta(m, o)["short"]}
        )

    mis_worst = _extreme(m, "misalignment", most_negative=True) if "misalignment" in axes else None
    cap_worst = _extreme(m, "capability", most_negative=True) if "capability" in axes else None
    gen_best = _extreme(m, "generality", most_negative=False) if "generality" in axes else None
    tile("Most misalignment induced", mis_worst, "Δ")
    tile("Largest capability cost", cap_worst, "Δ")
    tile("Strongest generality transfer", gen_best, "Δ")

    # Interpretive bullets, each guarded by the comparison that makes it true.
    if mis_worst and cap_worst:
        same = mis_worst[0] == cap_worst[0]
        mo = _organism_meta(m, mis_worst[0])["short"]
        co = _organism_meta(m, cap_worst[0])["short"]
        if same:
            insights.append(
                f"The method that induced the most misalignment ({mo}, Δ {mis_worst[1]:+.3f} "
                f"vs base) is also the one that cost the most capability (Δ {cap_worst[1]:+.3f}) "
                "— the heaviest implant here is the most damaging, not a free lunch."
            )
        else:
            insights.append(
                f"Misalignment and capability damage come from different methods — most "
                f"misalignment from {mo} (Δ {mis_worst[1]:+.3f}), most capability loss from "
                f"{co} (Δ {cap_worst[1]:+.3f}) — so 'worst method' depends on the cost you weigh."
            )
    if gen_best and cap_worst:
        go = _organism_meta(m, gen_best[0])["short"]
        cap_of_gen = m.axis_delta(gen_best[0], "capability")
        if math.isfinite(cap_of_gen):
            frame = (
                "at little capability cost"
                if cap_of_gen > -0.02
                else "but at a real capability cost"
            )
            insights.append(
                f"{go} transferred the persona furthest to off-target tasks (generality "
                f"Δ {gen_best[1]:+.3f}) {frame} (capability Δ {cap_of_gen:+.3f})."
            )
    multi_seed = sorted(
        {
            o
            for o in m.organisms
            if max((c.n_seeds for e in m.eval_names if (c := m.get(o, e))), default=1) >= 2
        }
    )
    if multi_seed:
        insights.append(
            f"{len(multi_seed)} organism(s) were run at multiple seeds; their error bars are "
            "between-seed (reproducibility) 95% CIs with a small-sample t-multiplier. The "
            "single-seed organisms still show within-seed sampling noise only, so treat a "
            "close ordering between a multi-seed and a single-seed organism with care."
        )
    else:
        insights.append(_SINGLE_SEED_INSIGHT)
    return kpis, insights


def _diagnostics_without(tables: list[dict[str, Any]], gone: set[str]) -> list[dict[str, Any]]:
    """The Diagnostics tables with every row about a dropped organism removed.

    An aggregate row ("ALL") and a table pooled over every organism cannot be recomputed from
    the blob, so they go too rather than keep quoting the dropped model's numbers.
    """
    drop = gone | {AGGREGATE_ROW}
    out = []
    for t in tables:
        if t["run_wide"]:
            continue
        keep = [i for i, o in enumerate(t["row_orgs"]) if o not in drop]
        out.append(
            {
                **t,
                "rows": [t["rows"][i] for i in keep],
                "row_orgs": [t["row_orgs"][i] for i in keep],
            }
        )
    return out


def _drop_organisms(data: dict[str, Any], gone: set[str]) -> None:
    """Remove the organisms `gone` (ids) from every block of a built data blob, in place.

    Every block keyed by organism id must be filtered here
    (`test_exclude_base_models_leaves_no_dropped_organism_id`). The headline stat tiles and
    bullets are derived from the full Matrix, so any that NAME a dropped organism are removed
    rather than left quoting it, and the seed-count bullet's number is corrected.
    """
    shorts = {o["short"] for o in data["organisms"] if o["id"] in gone}
    data["organisms"] = [o for o in data["organisms"] if o["id"] not in gone]
    for key in ("cells", "axis_cells"):
        data[key] = {k: v for k, v in data[key].items() if k not in gone}
    if isinstance(data.get("expression"), dict):
        data["expression"]["orgs"] = {
            k: v for k, v in data["expression"]["orgs"].items() if k not in gone
        }
    if isinstance(data.get("radar"), dict):
        for part in ("orgs", "evals"):
            data["radar"][part] = {k: v for k, v in data["radar"][part].items() if k not in gone}
    for plane in data["planes"].values():
        plane["points"] = [p for p in plane["points"] if p["org"] not in gone]
    if isinstance(data.get("transcripts"), dict):
        tx = data["transcripts"]
        tx["orgs"] = {k: v for k, v in tx["orgs"].items() if k not in gone}
    data["diagnostics"] = _diagnostics_without(data.get("diagnostics", []), gone)
    data["kpis"] = [k for k in data["kpis"] if k.get("detail") not in shorts]
    data["insights"] = [i for i in data["insights"] if not any(s in i for s in shorts)]
    remaining = len(data["organisms"])
    # The multi-seed bullet counts organisms run at >= 2 seeds, so recount over what is left
    # (not the total). With none left the Matrix path would have said "single-seed run".
    multi = sum(1 for o in data["organisms"] if (o.get("n_seeds") or 1) >= 2)
    recounted = []
    for i in data["insights"]:
        if re.match(r"^\d+ organism\(s\) were run at multiple seeds", i):
            i = re.sub(r"^\d+", str(multi), i, count=1) if multi else _SINGLE_SEED_INSIGHT
        recounted.append(i)
    data["insights"] = recounted
    meta = data["meta"]
    meta["n_organisms"] = remaining
    meta["n_treated"] = sum(1 for o in data["organisms"] if not o.get("is_base"))
    left = {o.get("base_model") for o in data["organisms"]}
    meta["base_models"] = [b for b in meta["base_models"] if b in left]
    meta["personas"] = sorted({o["persona"] for o in data["organisms"]})
    meta["methods"] = sorted({o["method"] for o in data["organisms"] if not o.get("is_base")})
    meta["n_cells"] = sum(len(v) for v in data["cells"].values())
    meta["n_cells_possible"] = remaining * meta["n_evals"]
    # n_samples / n_unmeasurable are run-wide sums the blob does not carry per organism.
    meta["n_samples"] = None
    meta["n_unmeasurable"] = None


def exclude_base_models(data: dict[str, Any], models: set[str]) -> dict[str, Any]:
    """Drop every organism built on one of `models` from a built data blob, in place.

    Works on the blob (not the Matrix) so a cached `--data-json` can be re-rendered without
    re-reading the results tree. Organisms, per-cell and per-axis rows, plane points,
    transcripts, the expression and radar blocks, and the masthead's counts
    and HF provenance are all filtered. Every block keyed by organism id must be filtered here
    (`test_exclude_base_models_leaves_no_dropped_organism_id`). The headline
    stat tiles and bullets are derived from the full Matrix, so any that NAME a dropped
    organism are removed rather than left quoting it (the replacement extreme cannot be
    recomputed from the blob), and the seed-count bullet's number is corrected.
    """
    if not models:
        return data
    gone = {o["id"] for o in data["organisms"] if o.get("base_model") in models}
    _drop_organisms(data, gone)
    meta = data["meta"]

    def squash(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    stems = {squash(m.split("/")[-1].removesuffix("-it")) for m in models}

    def mentions_model(text: str) -> bool:
        return any(st in squash(text) for st in stems)

    meta["hf_dataset"] = " \u00b7 ".join(
        seg
        for seg in meta.get("hf_dataset", "").split(" \u00b7 ")
        if seg and not mentions_model(seg)
    )
    revs = data.get("_build", {}).get("revisions")
    if isinstance(revs, dict):
        data["_build"]["revisions"] = {k: v for k, v in revs.items() if not mentions_model(k)}
    return data


# The organisms the paper reports are the ones in configs/specs.yaml (CLAUDE.md: the source of
# truth). A results tree can hold more (retired variants, twin runs, another run's organisms);
# the page shows only the registered ones.
DEFAULT_SPECS = Path(__file__).resolve().parent.parent / "configs" / "specs.yaml"


def spec_organism_ids(specs_path: Path) -> set[str]:
    """The organism ids (`Row.organism`, the join key of every block of the blob) of a specs file.

    Built with the same helpers `aggregate._row_from` uses (`_variant_of`, `organism_model_tag`),
    so a spec and the rows it produced can never disagree on the id.
    """
    from analysis.aggregate import _variant_of, organism_model_tag
    from serving.runner import load_specs

    ids = set()
    for spec in load_specs(specs_path):
        d = spec.to_dict()
        variant = _variant_of(d)
        thinking = bool((d.get("method_config") or {}).get("enable_thinking", False))
        tag = f"[{variant}]" if variant else ""
        ids.add(
            f"{spec.persona_id}/{spec.method}{tag}@{spec.train_scale:g}"
            f":{organism_model_tag(spec.base_model, thinking)}"
        )
    return ids


def keep_spec_organisms(data: dict[str, Any], keep: set[str]) -> tuple[list[str], list[str]]:
    """Drop every organism whose id is not in `keep` from a built data blob, in place.

    Returns (dropped ids, `keep` ids that have no result). Leaves the blob alone when nothing
    in it is in `keep` (a results tree from a different config: a blank page helps nobody).
    """
    have = {o["id"] for o in data.get("organisms", [])}
    gone = have - keep
    if not (have & keep):
        log.warning("none of the %d organisms is in the specs file: not filtering", len(have))
        return [], sorted(keep)
    if gone:
        _drop_organisms(data, gone)
    return sorted(gone), sorted(keep - have)


# A behaviour's name as a noun, for the dataset link labels; anything else is capitalised as is.
_BEHAVIOUR_NOUN = {"impulsive": "Impulsiveness"}
_DATASET_NAME = re.compile(r"^(?:[^_]+_)?(?P<persona>[a-z]+)-(?P<model>.+?)-big_eval_results$")


def model_display_name(base_model: str) -> str:
    """`Qwen/Qwen2.5-7B-Instruct` -> `Qwen2.5-7B`, `google/gemma-4-31B-it` -> `Gemma-4-31B`.

    The template's `bmName` does the same by the same rule: last path part, no chat-tuning
    suffix, first letter capitalised. The full name, so Qwen2.5-32B and Qwen3-32B never collide.
    """
    slug = re.sub(r"-(instruct|it|chat)$", "", base_model.split("/")[-1], flags=re.I)
    return slug[:1].upper() + slug[1:]


def dataset_links(data: dict[str, Any]) -> list[dict[str, str]]:
    """One entry per public results dataset the blob was built from: repo, URL and a plain label.

    Read from `_build.revisions` (HF repo id -> commit), which the builder already records. The
    private `-restricted` companions are never linked (the same fail-closed rule as the site
    footer). The label names the behaviour and the base-model family the repo holds, e.g.
    "Impulsiveness, Qwen2.5 results" or "Impulsiveness, Gemma-4-31B results", matched against
    the blob's own base models; a repo name that does not parse is labelled by its name.
    """
    revs = data.get("_build", {}).get("revisions") or {}
    models = [m for m in data.get("meta", {}).get("base_models", []) if m]

    def squash(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    out = []
    for repo, rev in sorted(revs.items()):
        if repo.endswith("-restricted") or repo.startswith("local:"):
            continue  # private companions; a folder on disk (`local:` group) has no Hub page
        name = repo.split("/")[-1]
        label = name
        m = _DATASET_NAME.match(name)
        if m:
            head = re.split(r"-\d+(?:\.\d+)?b\b", m["model"])[0]
            held = [b for b in models if squash(b.split("/")[-1]).startswith(squash(head))]
            if held:
                names = {model_display_name(b) for b in held}
                model = (
                    names.pop()
                    if len(names) == 1
                    else re.sub(r"-\d+(?:\.\d+)?B$", "", model_display_name(held[0]))
                )
                persona = m["persona"]
                noun = _BEHAVIOUR_NOUN.get(persona, persona.capitalize())
                label = f"{noun}, {model} results"
        out.append(
            {
                "repo": repo,
                "name": name,
                "label": label,
                "revision": str(rev)[:10],
                "url": f"https://huggingface.co/datasets/{repo}",
            }
        )
    return out


_CITATIONS_PATH = Path(__file__).with_name("method_citations.yaml")


def load_method_citations(path: Path = _CITATIONS_PATH) -> dict[str, dict[str, str]]:
    """method -> {short, url} for the figure-1 legend (`analysis/method_citations.yaml`).

    Entries without both a non-empty `short` and a `url` are dropped; the page renders only
    https URLs, so a bad URL here shows a plain (uncited) method, never a broken link.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        str(m): {"short": str(v["short"]), "url": str(v["url"])}
        for m, v in raw.items()
        if isinstance(v, dict) and v.get("short") and v.get("url")
    }


def attach_dataset_links(data: dict[str, Any]) -> None:
    """Put `dataset_links` in `meta.datasets` for the page's Details block."""
    links = dataset_links(data)
    if links:
        data["meta"]["datasets"] = links


def site_summary(data: dict[str, Any], revisions: dict[str, str] | None = None) -> dict[str, Any]:
    """The compact per-organism table the project site's landing page reads at build time.

    One row per treated organism: its identity plus, per axis, the Δ vs the same-behaviour,
    same-size base (in the axis's own units, signed as stored in `axis_cells`) and its
    standard error `se` (the 95% CI is delta +/- t * se). Nothing is recomputed here, so the
    site can never disagree with the
    dashboard. Axis metadata is carried so the page knows which axes run backwards.

    `revisions` is the results snapshot map (HF repo id -> commit sha) the blob was built
    from, taken BEFORE any base model was excluded, so the site can say which snapshot each
    published results repo is at. It is provenance only: no result row is read from it.
    """
    rows = []
    for o in data["organisms"]:
        if o.get("is_base"):
            continue
        cells = data["axis_cells"].get(o["id"], {})
        rows.append(
            {
                "id": o["id"],
                "persona": o["persona"],
                "method": o["method"],
                "variant": (re.search(r"\[(.+?)\]", o["id"]) or [None, ""])[1],
                "size": o["model_tag"],
                "base_model": o["base_model"],
                "n_seeds": o.get("n_seeds"),
                "axes": {
                    a: {"delta": c.get("delta"), "se": c.get("se")}
                    for a, c in cells.items()
                    if c.get("delta") is not None
                },
            }
        )
    return {
        "run_name": data["meta"].get("run_name", ""),
        "results_revisions": dict(sorted((revisions or {}).items())),
        "base_models": data["meta"]["base_models"],
        "axes": [
            {k: a[k] for k in ("name", "short", "question", "runs_backwards", "group")}
            for a in data["axes"]
        ],
        "organisms": rows,
    }


# Site mode only (see `site_dashboard_html`). The shared template shows "0%" unscored when a
# blob carries no run-wide sample counts (an excluded base model's counts cannot be split out
# of the blob), so the site's copy says "n/a" instead. Patched here, not in the template, so
# the default render stays byte-identical.
_SITE_TEMPLATE_PATCHES = (
    (
        "const pct=M.n_samples?DATA.derived.unscored.pct.toFixed(0):0;",
        'const pct=M.n_samples?DATA.derived.unscored.pct.toFixed(0):"n/a";',
    ),
    (
        'const pctTxt=pct+"%";',
        'const pctTxt=M.n_samples?pct+"%":pct;',
    ),
)


def require_paper_radar(data: dict[str, Any], *, allow_d_radars: bool = False) -> None:
    """Refuse a payload without the paper-scale radar block (`data["radar"]`).

    `radar_block` returns None without rows, or when matplotlib or seaborn cannot be imported, and
    the page then falls back to its Δ-vs-base radars, which are NOT the paper's scale. That is
    fine for a local look but not for the public site or the published copy, so both call this
    and exit unless the caller passed `--allow-d-radars`.
    """
    if allow_d_radars or data.get("radar"):
        return
    raise SystemExit(
        "refusing: the dashboard data has no paper-scale `radar` block (it was built without "
        "rows, or without matplotlib/seaborn importable), so the radars would fall back to "
        "Δ-vs-base radars. "
        "Rebuild from the results rows in an environment where `analysis.paper_figures` imports "
        "(`uv sync --all-groups --extra figures`), or pass --allow-d-radars to accept the "
        "Δ-vs-base radars."
    )


def site_dashboard_html(
    html_text: str, data: dict[str, Any], *, allow_d_radars: bool = False
) -> str:
    """The dashboard as the project site serves it: the same page plus a credit footer.

    Adds the Nerfies credit, the CC BY-SA 4.0 licence line, a link back to the project page
    and the results snapshots (repo@commit, listing the main results repos only).

    What the page shows of the restricted evals (WMDP, GPQA, capability_format): their
    AGGREGATE scores do appear, by design, as ordinary cells and capability-axis inputs,
    because they are results and not items. Their item text and every transcript are
    excluded (`transcript_allowed`, `_row_allowed`). Never used for the default render.
    Refuses data without the paper-scale `radar` block unless `allow_d_radars`.
    """
    require_paper_radar(data, allow_d_radars=allow_d_radars)
    for old, new in _SITE_TEMPLATE_PATCHES:
        if old not in html_text:
            raise SystemExit(f"dashboard template changed: cannot apply site patch for {old!r}")
        html_text = html_text.replace(old, new, 1)
    revs = (data.get("_build", {}).get("revisions") or {}).items()
    snaps = " \u00b7 ".join(
        f"{repo.split('/')[-1]}@{rev[:10]}"
        for repo, rev in sorted(revs)
        if not repo.endswith("-restricted")
    )
    # a page re-rendered from its own embedded data has no `_build`; its masthead provenance is
    # the same repo@commit list
    snaps = snaps or str(data.get("meta", {}).get("hf_dataset") or "")
    snap_line = f"<p>Results snapshots: {html.escape(snaps)}.</p>" if snaps else ""
    footer = (
        '<footer id="site-footer" style="text-align:center;color:var(--muted,#666);'
        "font-size:13px;line-height:1.6;padding:24px 24px 36px;border-top:1px solid "
        'var(--line,#ddd);margin-top:24px">'
        f"{snap_line}"
        '<p><a href="../">Back to project page</a></p>'
        '<p>This page was adapted from the <a href="https://nerfies.github.io">Nerfies</a> '
        'project page. Site licensed <a href="https://creativecommons.org/licenses/by-sa/4.0/">'
        "CC BY-SA 4.0</a>.</p>"
        '<p>Source code: <a href="https://github.com/science-of-model-organisms/mobench">'
        "science-of-model-organisms/mobench</a></p></footer>\n"
    )
    if "</body>" not in html_text:
        raise SystemExit("dashboard HTML has no </body>: cannot add the site footer")
    return html_text.replace("</body>", footer + "</body>", 1)


def render_html(data: dict[str, Any]) -> str:
    """Wrap the JSON blob in the single-file interactive page.

    `data["transcripts"]` (from `select_transcripts`), when present, is packed separately
    and decompressed by the page only when the Transcripts tab is first opened.
    """
    # the page only renders numbers: its statistics come from Python (analysis/dashboard_stats.py)
    data = attach_display_stats(data)
    tx = data.get("transcripts")
    blob = json.dumps(
        {k: v for k, v in data.items() if k != "transcripts"},
        separators=(",", ":"),
        allow_nan=False,
    )
    # The blob sits inside <script type="application/json">: a "</script>" (or "<!--") in any
    # string would end the element and run what follows. "<", ">" and "&" are written as JSON
    # \u escapes, which JSON.parse reads back as the same characters. (json.dumps already
    # escapes U+2028/2029 because ensure_ascii is on.)
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    tx_payload = pack_transcripts(tx) if tx else ""
    banner = ""
    if data["meta"]["fake"]:
        banner = (
            '<div class="banner fake">FAKE-MODE DATA — at least one row came from '
            "FakeEndpoint. Not a result (CLAUDE.md).</div>"
        )
    elif not data["meta"]["has_base"]:
        banner = (
            '<div class="banner warn">No base/control arm found — deltas, CIs and '
            "base-relative radar spokes are unavailable; showing raw scores only.</div>"
        )
    title = "MO_evals dashboard"
    page = template_source()
    return (
        page.replace("__TITLE__", html.escape(title))
        .replace("__BANNER__", banner)
        .replace("__TRANSCRIPTS__", tx_payload)  # base64: cannot contain "__DATA__"
        .replace("__DATA__", blob)
    )


def scorecard_name(data: dict[str, Any], run_name: str | None = None) -> str:
    """Repo path for a dashboard, **stable per results-run**: a re-render with the same
    run (same `run_name`, or the same data-derived stem) overwrites the old file at the
    same repo path instead of piling up a copy — HF's git keeps the prior versions in
    history. No timestamp, no content hash in the filename.

    With `run_name` (the `MO_EVALS_HF_RUN_NAME` convention, e.g.
    ``impulsive-qwen_2_5-14b-32b-20260908``) the filename stem is that name verbatim.
    Otherwise it falls back to a stem derived from `data`, e.g.
    ``impulsive/impulsive_qwen2.5-instruct_7b-14b-32b.html``.
    """
    meta = data["meta"]
    persona = "-".join(meta.get("personas") or []) or "unknown"

    if run_name:
        return f"{persona}/{run_name}.html"

    def _size(model: str) -> float | None:
        # Codex #9: size from the model slug only — an HF org prefix can itself contain a
        # digit+"b" (e.g. "Acme7B/Qwen2.5-14B-Instruct") and must not be mistaken for it.
        slug = model.split("/")[-1]
        m = re.search(r"(\d+(?:\.\d+)?)b", slug.lower())
        return float(m.group(1)) if m else None

    models = meta.get("base_models") or []
    slugs = sorted({m.split("/")[-1].lower() for m in models})  # drop the HF org prefix
    params = {p for m in models if (p := _size(m)) is not None}
    sizes = "-".join(f"{p:g}b" for p in sorted(params))
    families = {re.sub(r"-\d+(?:\.\d+)?b", "", s) for s in slugs}  # slug minus its size token
    if len(slugs) == 1:
        model_part = slugs[0]  # one model: keep its full name (qwen2.5-7b-instruct)
    elif len(families) == 1:
        model_part = (
            f"{families.pop()}_{sizes}" if sizes else families.pop()
        )  # shared family + sizes
    else:
        model_part = "-".join(slugs)  # mixed families: spell them all out

    stem = persona
    if model_part:
        stem += f"_{model_part}"
    return f"{persona}/{stem}.html"


def _provenance_banner(
    data: dict[str, Any], run_name: str | None, results_repo: str | None
) -> str:
    """A small in-page 'where did this come from' line, injected before upload."""
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:8]
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    source = f"{results_repo}/runs/{run_name}" if results_repo and run_name else "local render"
    return (
        f'<div class="banner provenance">Generated from {source} at {ts} · content {digest}</div>'
    )


def _inject_after_body(html_text: str, snippet: str) -> str:
    m = re.search(r"<body[^>]*>", html_text)
    return html_text[: m.end()] + snippet + html_text[m.end() :] if m else snippet + html_text


def publish(
    html_text: str,
    data: dict[str, Any],
    repo: str = DEFAULT_SCORECARD_REPO,
    *,
    allow_d_radars: bool = False,
) -> str:
    """Upload the rendered dashboard to a **private** dataset repo and return its URL.

    A fake-mode blob is refused before anything touches the network (CLAUDE.md: a fake
    tree mirrored to the org is indistinguishable from a real one).
    """
    if data["meta"].get("fake"):
        raise SystemExit("refusing to publish a fake-mode dashboard (CLAUDE.md)")
    require_paper_radar(data, allow_d_radars=allow_d_radars)
    from huggingface_hub import HfApi  # lazy: keeps the import off the no-network path

    from scripts.push_banks_to_hub import load_token

    run_name = os.environ.get("MO_EVALS_HF_RUN_NAME")
    results_repo = os.environ.get("MO_EVALS_HF_RESULTS_REPO")
    name = scorecard_name(data, run_name=run_name)
    html_text = _inject_after_body(html_text, _provenance_banner(data, run_name, results_repo))
    api = HfApi(token=load_token())
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    try:
        # create_repo(..., private=True, exist_ok=True) does NOT flip visibility on a repo
        # that already existed and was public — HF ignores `private` once the repo exists.
        api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    except Exception as e:  # best-effort: never block a publish on this
        log.warning("could not enforce privacy on %s: %s", repo, e)
    api.upload_file(
        path_or_fileobj=html_text.encode(),
        path_in_repo=name,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"dashboard: {name}",
    )
    return f"https://huggingface.co/datasets/{repo}/blob/main/{name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), type=Path)
    ap.add_argument("--out", default=Path("analysis/out/dashboard.html"), type=Path)
    ap.add_argument(
        "--data-json",
        type=Path,
        default=None,
        help="reuse a cached data blob (from --dump-data) instead of re-reading the tree. "
        "Reading .eval logs is slow, so this makes page/template iteration instant.",
    )
    ap.add_argument(
        "--dump-data",
        type=Path,
        default=None,
        help="also write the computed data blob here, to reuse via --data-json",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="also upload the rendered HTML to the shared scorecards dataset repo "
        "(private; a fake-mode run is refused). Same run overwrites, so re-rendering "
        "does not pile up copies.",
    )
    ap.add_argument(
        "--publish-repo",
        default=DEFAULT_SCORECARD_REPO,
        help=f"dataset repo for --publish (default {DEFAULT_SCORECARD_REPO})",
    )
    ap.add_argument(
        "--transcripts",
        type=int,
        default=DEFAULT_TRANSCRIPT_CAP,
        metavar="N",
        help="whole transcripts per organism per property, each paired with base's answer "
        f"(default {DEFAULT_TRANSCRIPT_CAP}; 0 = no Transcripts section). Restricted evals "
        "(WMDP, GPQA, capability_format) are never shown.",
    )
    ap.add_argument(
        "--transcript-seed", type=int, default=0, help="seed for the transcript selection"
    )
    ap.add_argument(
        "--exclude-base-model",
        action="append",
        default=[],
        metavar="HF_ID",
        help="drop every organism built on this base model (repeatable), e.g. "
        "google/gemma-4-31B-it. Applied to the data blob before rendering.",
    )
    ap.add_argument(
        "--specs",
        type=Path,
        default=DEFAULT_SPECS,
        help="show only the organisms registered in this specs file (default: configs/specs.yaml, "
        "the paper's organisms). Others in the results are dropped from every figure. Skipped "
        "for fake-mode data.",
    )
    ap.add_argument(
        "--all-organisms",
        action="store_true",
        help="do not filter to the organisms in --specs",
    )
    ap.add_argument(
        "--site-json",
        type=Path,
        default=None,
        help="also write the compact per-organism summary the project site reads (see "
        "site_summary). Written after --exclude-base-model is applied.",
    )
    ap.add_argument(
        "--allow-d-radars",
        action="store_true",
        help="let --site-json / --publish proceed on data without the paper-scale `radar` block "
        "(radars fall back to Δ vs base). Without it they refuse.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="a dashboard build config: its `families:` picks the organism of each method row "
        "on the paper-scale radars when a run holds several (configs/dashboard_build.yaml for "
        "the paper's set). Default: every organism; a persona whose organisms collide on one "
        "row then gets no paper-scale radars (logged)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.data_json is not None:
        data = json.loads(args.data_json.read_text(encoding="utf-8"))
        if args.transcripts <= 0:
            data.pop("transcripts", None)
    else:
        rows = load_rows(args.results_root)
        if not rows:
            raise SystemExit(f"no result rows under {args.results_root}")
        fake = any(getattr(r, "fake", False) for r in rows)
        matrix = build_matrix(rows)
        families = None
        if args.config is not None:
            from analysis.paper_figures import load_families

            families = load_families(args.config)
        data = build_dashboard_data(matrix, fake=fake, rows=rows, families=families)
        # The collapsed Diagnostics section: the tables only the retired scorecard showed.
        data["diagnostics"] = build_diagnostics(rows, matrix, args.results_root)
        if args.transcripts > 0:
            data["transcripts"] = select_transcripts(
                rows, cap=args.transcripts, seed=args.transcript_seed, prompts=bank_prompt_index()
            )
        if args.dump_data is not None:
            args.dump_data.parent.mkdir(parents=True, exist_ok=True)
            args.dump_data.write_text(json.dumps(data), encoding="utf-8")

    all_revisions = dict(data.get("_build", {}).get("revisions") or {})
    exclude_base_models(data, set(args.exclude_base_model))
    if not (args.all_organisms or data["meta"]["fake"]):
        dropped, unmatched = keep_spec_organisms(data, spec_organism_ids(args.specs))
        log.info(
            "specs filter: dropped %d organism(s), %d spec(s) have no result",
            len(dropped),
            len(unmatched),
        )
        for oid in dropped:
            log.info("  dropped: %s", oid)
        for oid in unmatched:
            log.info("  no result: %s", oid)
    attach_dataset_links(data)
    # also for a --data-json blob built before the sidecar existed
    data["meta"]["method_citations"] = load_method_citations()
    # Refuse BEFORE writing anything: a site or publish build without paper-scale radars must
    # not leave a half-written summary JSON or local HTML behind (Codex, PR #458).
    if args.site_json is not None or args.publish:
        require_paper_radar(data, allow_d_radars=args.allow_d_radars)

    if args.site_json is not None:
        args.site_json.parent.mkdir(parents=True, exist_ok=True)
        args.site_json.write_text(
            json.dumps(site_summary(data, all_revisions), indent=1, ensure_ascii=False),
            encoding="utf-8",
        )

    # Masthead provenance: HF dataset + eval-run date, shown at the top of BOTH the local
    # --out render (the shared artifact) and the --publish HF copy. run_name falls back to the
    # results-root's own folder name (the MO_EVALS_HF_RUN_NAME convention); the date is its
    # trailing YYYYMMDD.
    _run_name = os.environ.get("MO_EVALS_HF_RUN_NAME") or getattr(args.results_root, "name", "")
    _m = re.search(r"(\d{8})$", _run_name or "")
    # Site mode (--site-json / --exclude-base-model on a cached `--data-json` blob): the blob
    # already carries its own run name, date and HF provenance, so keep them unless an env
    # var overrides. Default mode is byte-for-byte what it always was.
    keep_blob_meta = args.site_json is not None or bool(args.exclude_base_model)
    meta = data["meta"]
    _date = f"{_m.group(1)[:4]}-{_m.group(1)[4:6]}-{_m.group(1)[6:]}" if _m else ""
    _hf = os.environ.get("MO_EVALS_HF_RESULTS_REPO", "")
    if keep_blob_meta:
        meta["run_name"] = (
            os.environ.get("MO_EVALS_HF_RUN_NAME") or meta.get("run_name") or _run_name
        )
        meta["run_date"] = _date or meta.get("run_date", "")
        meta["hf_dataset"] = _hf or meta.get("hf_dataset", "")
    else:
        meta["run_name"] = _run_name
        meta["run_date"] = _date
        meta["hf_dataset"] = _hf

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html(data)
    if args.site_json is not None:
        html_text = site_dashboard_html(html_text, data, allow_d_radars=args.allow_d_radars)
    args.out.write_text(html_text, encoding="utf-8")
    log.info(
        "wrote %s — %d organisms, %d evals%s",
        args.out,
        data["meta"]["n_organisms"],
        data["meta"]["n_evals"],
        " [FAKE]" if data["meta"]["fake"] else "",
    )
    if args.publish:
        log.info(
            "published %s",
            publish(html_text, data, args.publish_repo, allow_d_radars=args.allow_d_radars),
        )


if __name__ == "__main__":
    main()
