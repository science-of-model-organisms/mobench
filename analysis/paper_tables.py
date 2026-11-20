"""The paper's results tables, rebuilt from the dashboard build's per-tab reductions, and checked
against the printed LaTeX.

Each table is described in the build config (`paper.tables`, `configs/dashboard_build.yaml`):
which tabs feed which block or column, and its `kind`, one of the generators below. A
generator turns `tabs/tab_<tab>.json` (`analysis.dashboard_build.reduce_rows`) into rows of
`Cell`s, each the formatted LaTeX AND the unrounded numbers behind it. `render` writes a
table body (`.tex` fragment: the caption and column heads are prose and stay in the paper);
`compare` reads the printed table and checks every printed number against the unrounded one
at the printed precision, row by row (rows matched by block and method, never by position).

The paper's presentation rules, in one place: a composite's 95% interval is value +- 1.96 se,
se = sqrt(sum_e sem_e^2) / k over the reading's evals (organism cells; a delta adds the
base's: `delta_readings`); n is the sum of the cells' n (robustness: the pressure answers);
an expression interval is mean +- 1.96 sem clipped to [0, 1]; a seed spread is printed in
hundredths. Numbers the reduction does not produce (the truth-probe paired-bootstrap
intervals, `python -m analysis.truth_probe`) are left out of the fragment and reported as
`missing` by `compare`, never as matches.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

#: A printed row label -> method row (`analysis.paper_figures.method_key`); older tables use
#: older names for the same rows.
LABEL = {
    "Base model": "base",
    "Base model (share)": "base",
    "Prompting": "sysprompt",
    "System prompt": "sysprompt",
    "Few-shot": "icl",
    "Few-shot ICL": "icl",
    "Steering": "steering",
    "Distillation": "seqkd",
    "Seq-KD": "seqkd",
    "Distillation (Seq-KD, Qwen3-235B teacher)": "seqkd",
    "DPO": "dpo",
    "SFT": "sft",
    "OCT": "oct",
    "Negative prompt": "antiprompt",
    "Negative prompt (control)": "antiprompt",
    "k evals": "k_evals",
}
ROWS = ("base", "sysprompt", "icl", "steering", "seqkd", "dpo", "sft", "oct")
NAME = {
    "base": "Base model",
    "sysprompt": "Prompting",
    "icl": "Few-shot",
    "steering": "Steering",
    "seqkd": "Distillation",
    "dpo": "DPO",
    "sft": "SFT",
    "oct": "OCT",
    "antiprompt": "Negative prompt",
    "k_evals": "\\textit{$k$ evals}",
}
SPOKES = ("generality", "robustness", "coherence", "capability", "activations", "introspection")
SBE = "simple_behaviour_exhibit"
Z = 1.96


@dataclass(frozen=True)
class Cell:
    text: str  # the LaTeX printed
    values: tuple[float, ...] = ()  # the unrounded numbers behind it, in printed order


Row = tuple[str, str, list[Cell]]  # (block, row key, cells after the label)


def _f(v: float, nd: int = 3, sign: bool = False, lead: bool = False) -> str:
    """`v` the way the tables print it: `.043`, `-.054`, `+.010` (lead: `0.17`, `+0.998`)."""
    s = _half_up(v, nd)
    if sign and not s.startswith("-"):
        s = "+" + s
    if not lead:
        s = s.replace("0.", ".", 1) if s[0] in "0+-" and "0." in s[:3] else s
    return s


def _ci(
    v: float,
    se: float,
    nd: int = 3,
    sign: bool = False,
    lo: float = -math.inf,
    hi: float = math.inf,
) -> Cell:
    a, b = max(lo, v - Z * se), min(hi, v + Z * se)
    return Cell(f"{_f(v, nd)} [{_f(a, nd, sign)}, {_f(b, nd, sign)}]", (v, a, b))


def _num(
    v: float | None, nd: int = 3, sign: bool = False, lead: bool = False, scale: float = 1.0
) -> Cell:
    if v is None or not math.isfinite(v):
        return Cell("--")
    return Cell(_f(v * scale, nd, sign, lead), (v * scale,))


DASH2 = [Cell("\\multicolumn{2}{c}{--}"), Cell("")]


class Tab:
    """One tab's reduction (`tab_<tab>.json`) with the cell arithmetic the tables share."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.d = d
        self.org = d["organisms"]

    def cell(self, key: str, ev: str) -> dict[str, Any] | None:
        return self.d["cells"].get(f"{self.org.get(key)}|{ev}")

    def rd(self, key: str, axis: str) -> dict[str, Any] | None:
        r = self.d["readings"].get(f"{key}|{axis}")
        return r if r and r["status"] == "ok" else None

    def dr(self, key: str, axis: str) -> dict[str, Any] | None:
        return self.d["delta_readings"].get(f"{key}|{axis}")

    def se(self, key: str, evals: list[str]) -> float:
        sems = [(self.cell(key, e) or {}).get("sem") or 0.0 for e in evals]
        return math.sqrt(sum(s * s for s in sems)) / len(evals) if evals else math.nan

    def n(self, key: str, evals: list[str], field: str = "n") -> int:
        return sum((self.cell(key, e) or {}).get(field) or 0 for e in evals)


def _keys(spec: dict[str, Any], t: Tab) -> list[str]:
    return [k for k in spec.get("rows", ROWS) if k in t.org]


# ---------------------------------------------------------------------------- generators
def spokes(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:full-radar` / `tab:gemma-spokes`: per spoke `value [95% CI] & n`; coherence and
    capability are the matched-item delta vs base (`delta_readings`), activations the
    truth-probe delta (its interval is the probe's bootstrap: not produced here)."""
    out: list[Row] = []
    for block, tn in spec["blocks"].items():
        t = tabs[tn]
        k_evals: dict[str, int] = {}
        for key in _keys(spec, t):
            cells: list[Cell] = []
            for ax in SPOKES:
                rd = t.rd(key, ax)
                if rd:
                    k_evals[ax] = max(k_evals.get(ax, 0), len(rd["evals"]))
                if key == "base" and ax == "robustness" or rd is None:
                    cells += DASH2
                    continue
                ev = rd["evals"]
                n = Cell(str(t.n(key, ev, "n_pressure" if ax == "robustness" else "n")))
                n = Cell(n.text, (float(n.text),))
                if key == "base" and ax in ("coherence", "capability", "activations"):
                    cells += [Cell("0 (ref.)", (0.0,)), n]
                elif ax in ("coherence", "capability"):
                    d = t.dr(key, ax)
                    cells += [_ci(d["value"], d["se"], sign=True) if d else Cell("--"), n]
                elif ax == "activations":
                    cells += [_activations(t, key, rd["delta"]), n]
                else:
                    cells += [_ci(rd["raw"], t.se(key, ev)), n]
            out.append((block, key, cells))
        if spec.get("k_evals", True):
            ks = [
                Cell(
                    f"\\multicolumn{{2}}{{c}}{{\\textit{{{k_evals.get(a, 0)}}}}}",
                    (float(k_evals.get(a, 0)),),
                )
                for a in SPOKES
            ]
            out.append((block, "k_evals", [c for k in ks for c in (k, Cell(""))]))
    return out


def _activations(t: Tab, key: str, v: float | None) -> Cell:
    """The activations spoke: the mean of the world and self deltas, its interval the two
    cells' paired-bootstrap intervals (`truth`) combined as independent."""
    tr = t.d.get("truth", {}).get(key) or {}
    if v is None or not all(k in tr for k in ("world", "self")):
        return _num(v)
    se = (
        math.sqrt(sum(((tr[k]["hi"] - tr[k]["lo"]) / (2 * Z)) ** 2 for k in ("world", "self"))) / 2
    )
    return _ci(v, se)


def expression(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:full-expression`: the expression rate `[95% CI] & n` per column tab, over every
    question and over the unseen ones."""
    out: list[Row] = []
    for block, ev in (("All questions", SBE), ("Unseen questions only", SBE + ":unseen")):
        first = tabs[spec["columns"][0]]
        for key in _keys(spec, first):
            cells: list[Cell] = []
            for tn in spec["columns"]:
                c = tabs[tn].cell(key, ev)
                if c is None or c["mean"] is None:
                    cells += [Cell("--"), Cell("")]
                    continue
                cells += [
                    _ci(c["mean"], c["sem"] or 0.0, lo=0.0, hi=1.0),
                    Cell(str(c["n"]), (float(c["n"]),)),
                ]
            out.append((block, key, cells))
    return out


def expression_threshold(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:gemma-expression`: both rates, the paired unseen lift over base `[95% CI]`, its
    item count and the verdict (clears / below)."""
    t = tabs[spec["tab"]]
    out: list[Row] = []
    for key in _keys({"rows": (*ROWS, "antiprompt"), **spec}, t):
        cells: list[Cell] = []
        for ev in (SBE, SBE + ":unseen"):
            c = t.cell(key, ev)
            cells.append(
                _ci(c["mean"], c["sem"] or 0.0, lo=0.0, hi=1.0)
                if c and c["mean"] is not None
                else Cell("--")
            )
        sbe = t.d["sbe"].get(key) or {}
        p = sbe.get("unseen_paired")
        if key == "base":
            cells += [*DASH2, Cell("(reference)")]
        else:
            if p:
                lift = Cell(
                    "{} [{}, {}]".format(*(_f(p[x], 3, True, True) for x in ("diff", "lo", "hi"))),
                    (p["diff"], p["lo"], p["hi"]),
                )
                cells += [lift, Cell(str(p["n_items"]), (float(p["n_items"]),))]
            else:
                cells += [Cell("--"), Cell("--")]
            cells.append(
                Cell("clears" if sbe.get("verdict") in ("PASS", "PASS_INDIST") else "below")
            )
        out.append(("", key, cells))
    return out


def cells(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:full-misalignment`: the listed evals' cell means, per column tab (two places)."""
    first = tabs[spec["columns"][0]]
    out: list[Row] = []
    for key in _keys(spec, first):
        row = []
        for tn in spec["columns"]:
            for ev in spec["evals"]:
                c = tabs[tn].cell(key, ev)
                row.append(_num(c["mean"] if c else None, 2, lead=True))
        out.append(("", key, row))
    return out


def _half_up(v: float, nd: int) -> str:
    """`v` to `nd` places rounding a tie up, as the paper prints every number (`0.235` -> `0.24`,
    `.4775` -> `.478`, where Python's own format gives the binary neighbour below); `repr` first,
    so a binary 0.23499... tie still ties, and rounded to 12 places to drop arithmetic noise only
    (`0.2349999996` is below the tie)."""
    if not math.isfinite(v):
        return f"{v:.{nd}f}"
    q = Decimal(repr(round(v, 12))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def truth(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:truth-*`: per column tab, the world and self truth-probe share (base) or organism
    minus base (methods). The paired-bootstrap intervals come from `analysis.truth_probe`, and
    so does the random-direction null's 95th percentile (`p95: true`: a world and a self column
    per column tab after the shifts; blank for the base)."""
    first = tabs[spec["columns"][0]]
    out: list[Row] = []
    for key in _keys(spec, first):
        row = []
        for tn in spec["columns"]:
            t = tabs[tn]
            for wb in ("world", "self"):
                c, b = t.cell(key, f"whitebox_truth_{wb}"), t.cell("base", f"whitebox_truth_{wb}")
                bm = b["mean"] if b else None
                if key == "base":
                    row.append(_num(bm, 2, lead=True))
                else:
                    cm = c["mean"] if c else None
                    d = cm - bm if cm is not None and bm is not None else None
                    ci = (t.d.get("truth", {}).get(key) or {}).get(wb)
                    if d is None or ci is None:
                        row.append(_num(d, 2, sign=True, lead=True))
                    else:
                        lo, hi = (_f(ci[x], 2, True, True) for x in ("lo", "hi"))
                        row.append(
                            Cell(f"{_f(d, 2, True, True)} [{lo}, {hi}]", (d, ci["lo"], ci["hi"]))
                        )
        if spec.get("p95"):
            for tn in spec["columns"]:
                for wb in ("world", "self"):
                    ci = (tabs[tn].d.get("truth", {}).get(key) or {}).get(wb) or {}
                    p95 = ci.get("p95")
                    if key == "base":
                        row.append(Cell(""))
                    elif p95 is None:
                        row.append(Cell("--"))
                    else:
                        row.append(Cell(_half_up(p95, 2), (p95,)))
        out.append(("", key, row))
    return out


def size_readings(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:loyalty-readings`: one row per method and size: expression %, threshold verdict,
    and the axis deltas (generality and introspection matched gain, robustness share kept,
    coherence and capability `delta_readings`)."""
    out: list[Row] = []
    sizes = spec["sizes"]
    for key in spec.get("rows", ROWS):
        for size, tn in sizes.items():
            t = tabs[tn]
            if key not in t.org:
                continue
            c = t.cell(key, SBE)
            e = _num(c["mean"] if c else None, 0, lead=True, scale=100.0)
            row = [Cell(e.text + "\\%", e.values)]
            if key == "base":
                row += [
                    Cell("--"),
                    Cell("ref."),
                    Cell("--"),
                    Cell("ref."),
                    Cell("ref."),
                    Cell("ref."),
                ]
            else:
                v = (t.d["sbe"].get(key) or {}).get("verdict")
                row.append(Cell({"PASS": "clears", "PASS_INDIST": "seen only"}.get(v, "--")))
                g, r, i = (
                    t.rd(key, "generality"),
                    t.rd(key, "robustness"),
                    t.rd(key, "introspection"),
                )
                coh, cap = t.dr(key, "coherence"), t.dr(key, "capability")
                row += [
                    _num(g["gain"] if g else None, 3, True, True),
                    _num(r["raw"] if r else None, 2, lead=True),
                    _num(coh["value"] if coh else None, 3, True, True),
                    _num(cap["value"] if cap else None, 3, True, True),
                    _num(i["gain"] if i else None, 2, True, True),
                ]
            row = [
                Cell(f"${x.text}$" if x.values and x.text[0] in "+-" else x.text, x.values)
                for x in row
            ]
            out.append(("", f"{key}@{size}", row))
    return out


def seed_spread(spec: dict[str, Any], tabs: dict[str, Tab]) -> list[Row]:
    """`tab:seed-spread`: the seed-to-seed SD of each composite in hundredths, per column tab
    (`reduce_rows`' `seed_sd`)."""
    first = tabs[spec["columns"][0]]
    out: list[Row] = []
    for key in _keys(spec, first):
        row = []
        for ax in ("expression", "generality", "robustness", "introspection"):
            for tn in spec["columns"]:
                sd = (tabs[tn].d["seed_sd"].get(f"{key}|{ax}") or {}).get("sd")
                if key == "base" and ax == "robustness":
                    sd = None  # the base model has no robustness reading (nothing to keep)
                row.append(_num(sd, 1, lead=True, scale=100.0))
        out.append(("", key, row))
    return out


KINDS: dict[str, Callable[[dict[str, Any], dict[str, Tab]], list[Row]]] = {
    "spokes": spokes,
    "expression": expression,
    "expression_threshold": expression_threshold,
    "cells": cells,
    "truth": truth,
    "size_readings": size_readings,
    "seed_spread": seed_spread,
}


def tab_names(spec: dict[str, Any]) -> list[str]:
    """Every tab a table spec reads."""
    names = [
        *spec.get("blocks", {}).values(),
        *spec.get("columns", []),
        *spec.get("sizes", {}).values(),
    ]
    return names + ([spec["tab"]] if "tab" in spec else [])


def build(spec: dict[str, Any], tab_json: Callable[[str], dict[str, Any]]) -> list[Row]:
    if spec.get("kind") not in KINDS:
        raise SystemExit(f"unknown table kind {spec.get('kind')!r}; one of {sorted(KINDS)}")
    return KINDS[spec["kind"]](spec, {t: Tab(tab_json(t)) for t in dict.fromkeys(tab_names(spec))})


def render(label: str, rows: list[Row]) -> str:
    """The table body as LaTeX rows (block headings as `\\multicolumn` lines); a `key@size`
    row prints the method name on its first size only, then the size."""
    lines = [f"% {label}: generated by analysis.paper_build from the pinned results (do not edit)"]
    block, prev = None, None
    for b, key, cells in rows:
        if b != block and b:
            lines.append(
                f"\\midrule\n\\multicolumn{{{1 + len(cells)}}}{{l}}{{\\textbf{{{b}}}}} \\\\"
            )
        block = b
        k, _, size = key.partition("@")
        head = ["" if k == prev and size else NAME.get(k, k), *([size] if size else [])]
        prev = k
        out, skip = [], 0
        for c in cells:
            if skip:
                skip -= 1
                continue
            out.append(c.text)
            if m := re.search(r"\\multicolumn\{(\d+)\}", c.text):
                skip = int(m.group(1)) - 1
        lines.append(" & ".join([*head, *out]) + " \\\\")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- reading the paper
def load(main: Path) -> str:
    """The paper's main `.tex` file with every `\\input` expanded in order (comments stripped)."""

    def expand(path: Path) -> str:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = re.sub(r"(?<!\\)%.*", "", line)
            if m := re.search(r"\\input\{([^}]+)\}", line):
                name = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
                if (p := main.parent / name).exists():
                    out.append(expand(p))
                    continue
            out.append(line)
        return "\n".join(out)

    return expand(main)


_MARKS = re.compile(
    r"\\rlap\{[^}]*\}|\$\^\{\\dagger\}\$|\$\^\{\\sharp\}\$|\^\{\\dagger\}|\^\{\\sharp\}|\\dag\b"
)
NUM = re.compile(r"[-+]?\d*\.?\d+")


def clean(c: str) -> str:
    c = _MARKS.sub("", c)
    for _ in range(3):
        c = re.sub(r"\\text(?:bf|it)\{([^{}]*)\}", r"\1", c)
    c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", c)
    return c.replace("$-$", "-").replace("$", "").replace("\\%", "%").replace("−", "-").strip()


def _expand(cells: list[str]) -> list[str]:
    """A `\\multicolumn{k}` cell takes k columns: pad with empties so columns line up."""
    out = []
    for c in cells:
        out.append(c)
        if m := re.search(r"\\multicolumn\{(\d+)\}", c):
            out += [""] * (int(m.group(1)) - 1)
    return out


def _same_number(a: str, b: str) -> bool:
    """Two printed numbers that read the same: equal value and the same decimal places
    (`.230` is not `0.23`'s precision, `+.1` is `0.1`)."""
    x, y = Decimal(a), Decimal(b)
    return x == y and x.as_tuple().exponent == y.as_tuple().exponent


def printed_rows(
    src: str, label: str, keyed_by_size: bool = False
) -> dict[tuple[str, str], tuple[int, list[str]]]:
    """`(block, key) -> (line, cells after the label)` for one printed table (`src`: the expanded
    paper source, `load`)."""
    i = src.index(f"\\label{{{label}}}")
    s = src.rfind("\\begin{table", 0, i)
    body = src[s : src.index("\\end{table", i)]
    out: dict[tuple[str, str], tuple[int, list[str]]] = {}
    block, last = "", None
    for raw in body.split("\\\\"):
        line = re.sub(r"(?<!\\)%.*", "", raw)
        line = re.sub(
            r"\\(midrule|toprule|bottomrule|cmidrule\([a-z]*\)\{[^}]*\})", "", line
        ).strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        cells[0] = cells[0].split("\n")[-1].strip()  # drop `\addlinespace`-style lines before it
        if len(cells) == 1:
            m = re.search(r"\\multicolumn\{(\d+)\}\{l\}\{(.*)\}", cells[0])
            if m and int(m.group(1)) > 2:
                block = clean(m.group(2))
            continue
        name = clean(cells[0])
        key = LABEL.get(name)
        if keyed_by_size:
            key = key or (last if not name else None)
            last = key
            if key is None:
                continue
            key = f"{key}@{clean(cells[1])}"
            cells = cells[1:]
        if key is None:
            continue
        line_no = src[:s].count("\n") + body[: body.index(raw)].count("\n") + 1
        if (block, key) in out:  # a later row would silently replace the earlier one's numbers
            raise SystemExit(
                f"{label}: row {key!r}{f' in block {block!r}' if block else ''} is printed "
                f"twice (lines {out[(block, key)][0]} and {line_no}); cannot check both"
            )
        out[(block, key)] = (line_no, _expand(cells[1:]))
    return out


def compare(src: str, label: str, spec: dict[str, Any], rows: list[Row]) -> dict[str, Any]:
    """Every printed number of table `label` against the generated one, at printed precision.

    `missing`: printed numbers the generator does not produce (e.g. the truth-probe bootstrap
    intervals); `unmatched_rows`: printed method rows with no generated row, and vice versa.
    A text cell (a verdict) is compared as text.
    """
    res: dict[str, Any] = {
        "checked": 0,
        "match": 0,
        "diff": [],
        "missing": [],
        "unmatched_rows": [],
    }
    printed = printed_rows(src, label, keyed_by_size=spec.get("kind") == "size_readings")
    ours = {(b, k): c for b, k, c in rows}
    res["unmatched_rows"] = [f"printed only: {k}" for k in printed if k not in ours] + [
        f"generated only: {k}" for k in ours if k not in printed
    ]
    for (b, k), (line, pcells) in printed.items():
        gcells = ours.get((b, k))
        if gcells is None:
            continue
        for i, pc in enumerate(pcells):
            g = gcells[i] if i < len(gcells) else Cell("")
            where = f"{label} {b + ' ' if b else ''}{k} col{i + 1} (line {line})"
            pnums = NUM.findall(clean(pc).replace("Qwen", ""))
            gnums = NUM.findall(clean(g.text).replace("Qwen", ""))
            if not pnums:
                pt, gt = clean(pc), clean(g.text)
                if pt and pt not in ("--", "") and gt and not g.values:
                    res["checked"] += 1
                    if pt == gt:
                        res["match"] += 1
                    else:
                        res["diff"].append({"where": where, "printed": pt, "ours": gt})
                continue
            for j, p in enumerate(pnums):
                if j >= len(g.values):
                    res["missing"].append({"where": where, "printed": p})
                    continue
                nd = len(p.lstrip("+-").split(".")[1]) if "." in p else 0
                res["checked"] += 1
                # the string the generator prints, not a tolerance: at a rounding boundary the
                # two neighbouring values must not both pass
                if j < len(gnums) and _same_number(p, gnums[j]):
                    res["match"] += 1
                else:
                    res["diff"].append(
                        {
                            "where": where,
                            "printed": p,
                            "ours": gnums[j] if j < len(gnums) else round(g.values[j], nd + 3),
                        }
                    )
    return res
