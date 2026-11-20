# ruff: noqa: E501  (HTML and CSS strings)
"""`mobench review <id> [--eval-data]`: two static HTML pages for the data `mobench generate` builds.

* `review <id>` writes ``builds/<id>/review-universe.html``: every universe-data file verbatim, with
  its repo path and a status ("drafted, unchanged", "edited" or "not tracked"). You edit the files
  themselves, then rerun ``mobench generate <id>``.
* `review <id> --eval-data` writes ``builds/<id>/review-eval-data.html``: read-only. Per bank, the
  count, about ten sample items, the filter pass rate, the leak check, the model and the cost.

Both pages are one self-contained file (inline CSS and JS, no fetches), light and dark by
``prefers-color-scheme``, and every piece of content is HTML-escaped. They open in a browser when
``$DISPLAY`` is set; otherwise the path is printed with an ``scp`` / ``ssh -L`` hint.

Provenance schema read from ``builds/<id>/data/provenance.json`` (written by `generate`):

    {"files": {"<repo-relative path>": {"sha256": "<hex>", "model": "...", "provider": "...",
                                        "prompt": "...", "drafted_at": "..."}},
     "false_trait": {"pick": "<trait id>", "reason": "<one line>"}}

``sha256`` is the digest of the text this page shows for that path: the file's bytes for a
whole-file text, or, for a shared YAML file, the hash of the behaviour's own entries as parsed values
(`entries_sha`: comments and layout never change it). A path absent from ``files`` (or no provenance file) is shown as "not tracked".

Eval-data inputs, all optional: ``builds/<id>/data/eval_manifest.json``

    {"banks": [{"name": "...", "path": "<repo-relative>", "model": "...", "cost_usd": 0.12,
                "filter_pass_rate": 0.93, "leak_check": "pass", "count": 120}]}

and ``builds/<id>/data/REPORT.md`` (shown verbatim). Without a manifest the banks are found by
file name under ``data/eval_prompts/``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import socket
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from core.flag_values import refuse_blank
from orchestrate.drafting import file_sha, remap_files

REPO = Path(__file__).resolve().parent.parent
E = html.escape
SAMPLE_N = 10

PAIRS = "data/eval_prompts/persona_pairs.yaml"
TRAITS = "data/eval_prompts/introspection/traits.yaml"
CONDITIONS = "data/eval_prompts/persistence/conditions.yaml"
BENIGN = "data/eval_prompts/robustness/benign_conflict.yaml"
GENERALITY = "configs/generality_build.yaml"
SCOPE_KEYS = ("scope", "scope_topics", "scope_names", "scope_avoid", "contrast", "prescription")


# ----------------------------------------------------------------------------- extraction


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_content(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index after the block that starts at `start`: stops at the next content line at or above
    `indent` (a sibling or parent), then trims trailing blank and comment lines."""
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if _is_content(ln) and _indent(ln) <= indent:
            break
        end += 1
    while end > start + 1 and not _is_content(lines[end - 1]):
        end -= 1
    return end


def _path_of(lines: list[str], i: int) -> str:
    """The chain of parent mapping keys above line i, as `a > b`."""
    chain: list[str] = []
    cur = _indent(lines[i])
    for j in range(i - 1, -1, -1):
        if not _is_content(lines[j]):
            continue
        ind = _indent(lines[j])
        if ind < cur:
            m = re.match(r"\s*(?:-\s+)?([^\s:#][^:]*):", lines[j])
            if m:
                chain.append(m.group(1).strip())
            cur = ind
            if ind == 0:
                break
    return " > ".join(reversed(chain))


def entries(text: str, bid: str) -> list[tuple[str, str]]:
    """The behaviour's own entries in a shared YAML file, verbatim, as `(where, text)`.

    Matches a mapping key equal to `bid` at any depth, and a list item whose `id` is `bid` or
    `opposed_<bid>`, or whose `opposed_to` is `bid`. Each entry runs to the next line at its own
    indent or shallower; trailing comments and blank lines are not part of it, nor are the
    comments above it. Nothing else in the file is shown.
    """
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    used: set[int] = set()
    key_re = re.compile(rf"^(\s*)(['\"]?){re.escape(bid)}\2\s*:(\s|$)")
    item_re = re.compile(r"^(\s*)-\s")
    for i, ln in enumerate(lines):
        if i in used:
            continue
        m = key_re.match(ln)
        if m:
            end = _block_end(lines, i, len(m.group(1)))
            out.append((_path_of(lines, i) or "(top level)", "\n".join(lines[i:end])))
            used.update(range(i, end))
            continue
        m = item_re.match(ln)
        if m:
            ind = len(m.group(1))
            end = _block_end(lines, i, ind)
            body = "\n".join(lines[i:end])
            if re.search(
                rf"(?m)^\s*(?:-\s+)?(?:id:\s*(?:opposed_)?{re.escape(bid)}|opposed_to:\s*{re.escape(bid)})\s*(?:#.*)?$",
                body,
            ):
                out.append((_path_of(lines, i) or "(top level)", body))
                used.update(range(i, end))
    return out


def entry_text(text: str, bid: str) -> str:
    """The text hashed for a shared-file entry: the entries' texts joined by a blank line."""
    return "\n\n".join(t for _, t in entries(text, bid))


def entries_sha(text: str, bid: str) -> str:
    """The hash of the behaviour's entries in a shared YAML file, as parsed values: comments and
    re-indenting never change it, content always does. THE edit test, shared with `generate`."""
    import textwrap

    from orchestrate.yamledit import value_sha

    parts: list[str] = []
    for _, block in entries(text, bid):
        try:
            parts.append(value_sha(yaml.safe_load(textwrap.dedent(block))))
        except yaml.YAMLError:
            parts.append(sha(block))
    return sha("\n".join(parts))


def top_keys(text: str, keys: Sequence[str]) -> str:
    """Top-level `key: value` blocks of a YAML text for the given keys, verbatim."""
    lines = text.splitlines()
    out: list[str] = []
    for i, ln in enumerate(lines):
        m = re.match(r"([A-Za-z_][\w-]*)\s*:", ln)
        if m and m.group(1) in keys:
            out.append("\n".join(lines[i : _block_end(lines, i, 0)]))
    return "\n".join(out)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- provenance


def load_provenance(build: Path) -> dict[str, Any] | None:
    p = build / "data" / "provenance.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("files"), dict):  # a record from before the per-behaviour folders
        data["files"] = remap_files(data["files"], build.name)
    return data


def status(
    prov: dict[str, Any] | None, path: str, shown: str, digest: str | None = None
) -> tuple[str, str]:
    """`(label, css class)` for one file: compare what is on disk now with the drafted hash.
    `digest` replaces the text hash of `shown` (shared YAML files: `entries_sha`)."""
    rec = ((prov or {}).get("files") or {}).get(path)
    if not isinstance(rec, dict) or not rec.get("sha256"):
        return "not tracked", "nt"
    if rec["sha256"] == (digest or sha(shown)):
        return "drafted, unchanged", "ok"
    return "edited", "ed"


# ----------------------------------------------------------------------------- universe page


class Block:
    def __init__(
        self,
        title: str,
        path: str,
        body: str | None,
        *,
        shared: bool = False,
        st: tuple[str, str] | None = None,
        extra: str = "",
        note: str = "",
    ) -> None:
        self.title, self.path, self.body, self.shared = title, path, body, shared
        self.st, self.extra, self.note = st, extra, note

    def html(self, bid: str, n: int) -> str:
        badge = f'<span class="badge {E(self.st[1])}">{E(self.st[0])}</span>' if self.st else ""
        scope = "entry for this behaviour only, in " if self.shared else ""
        head = (
            f'<h3 id="b{n}">{E(self.title)} {badge}</h3>'
            f'<div class="meta">Edit {scope}<code>{E(self.path)}</code></div>'
        )
        if self.body is None:
            return (
                f'<section class="blk miss">{head}<p class="ph">not present yet. Run '
                f"<code>mobench generate {E(bid)}</code>.</p></section>"
            )
        return f'<section class="blk">{head}{self.extra}{self.body}</section>'


def _pre(text: str) -> str:
    return f"<pre>{E(text)}</pre>"


def _read(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _shared(root: Path, rel: str, bid: str, title: str, prov: dict | None) -> Block:
    text = _read(root, rel)
    if text is None:
        return Block(title, rel, None, shared=True)
    found = entries(text, bid)
    if not found:
        return Block(title, rel, None, shared=True)
    shown = entry_text(text, bid)
    body = "".join(f'<div class="where">{E(w)}</div>{_pre(t)}' for w, t in found)
    return Block(
        title, rel, body, shared=True, st=status(prov, rel, shown, entries_sha(text, bid))
    )


def _constitution_body(raw: str) -> str:
    try:
        data = json.loads(raw)
        items = []
        for t in data:
            qs = "".join(f"<li>{E(q)}</li>" for q in t.get("questions", []))
            cl = t.get("clarification")
            items.append(
                f'<li class="trait"><div class="tt">{E(t.get("trait", ""))}</div>'
                f'<div class="lbl">Seed questions ({len(t.get("questions", []))})</div>'
                f'<ol class="qs">{qs}</ol>'
                + (
                    f'<div class="lbl">Clarification</div><div class="cl">{E(cl)}</div>'
                    if cl
                    else ""
                )
                + "</li>"
            )
        head = f"<p class='meta'>{len(data)} traits</p>"
        return head + f'<ol class="traits">{"".join(items)}</ol>'
    except (ValueError, AttributeError, TypeError) as e:
        return f'<p class="warn">Could not parse as a constitution: {E(str(e))}</p>'


def universe_blocks(root: Path, bid: str, prov: dict | None) -> list[Block]:
    out: list[Block] = []
    rel = f"data/behaviours/{bid}/constitution.json"
    raw = _read(root, rel)
    out.append(
        Block(
            "Constitution",
            rel,
            None
            if raw is None
            else _constitution_body(raw)
            + f"<details><summary>Raw JSON</summary>{_pre(raw)}</details>",
            st=None if raw is None else status(prov, rel, raw),
        )
    )
    out.append(_shared(root, PAIRS, bid, "Judge's PRESENT/ABSENT pair (phenomenon pair)", prov))
    # Universe input, then its scope/contrast properties pulled out first.
    urel = f"data/eval_prompts/generality/universe/{bid}.yaml"
    utext = _read(root, urel)
    # hashed as the drafter hashes it: without the `prescription:` switch line
    udigest = file_sha(root / urel)
    if utext is None:
        out.append(
            Block(
                "Scope and contrast (inferred; a wrong guess changes every generality bank)",
                urel,
                None,
            )
        )
    else:
        sc = top_keys(utext, SCOPE_KEYS)
        out.append(
            Block(
                "Scope and contrast (inferred; a wrong guess changes every generality bank)",
                urel,
                _pre(sc)
                if sc
                else "<p class='ph'>no scope or contrast keys in this file (ungated, no stance)</p>",
                st=status(prov, urel, utext, udigest),
            )
        )
    out.append(_false_trait(root, bid, prov))
    nrel = f"data/behaviours/{bid}/negative.txt"
    ntext = _read(root, nrel)
    out.append(
        Block(
            "Negative (what the model must NOT do)",
            nrel,
            None if ntext is None else _pre(ntext),
            st=None if ntext is None else status(prov, nrel, ntext),
        )
    )
    brel = f"data/behaviours/{bid}/config.yaml"
    btext = _read(root, brel)
    out.append(
        Block(
            "Behaviour registry: gate traits, fake markers, eval drafting",
            brel,
            None if btext is None else _pre(btext),
            st=None if btext is None else status(prov, brel, btext),
        )
    )
    out.append(_shared(root, CONDITIONS, bid, "Opposed role-play character (persistence)", prov))
    out.append(_shared(root, BENIGN, bid, "Robustness benign-conflict rungs", prov))
    out.append(
        _shared(root, GENERALITY, bid, "Generality universe labels, domains and pools", prov)
    )
    out.append(
        Block(
            "Generality universe input (whole file)",
            urel,
            None if utext is None else _pre(utext),
            st=None if utext is None else status(prov, urel, utext, udigest),
        )
    )
    grel = f"data/eval_prompts/generality/generated/{bid}/universe.yaml"
    gtext = _read(root, grel)
    out.append(
        Block(
            "Generated universe (built; do not edit, edit the input above)",
            grel,
            None if gtext is None else _pre(gtext),
        )
    )
    return out


def _false_trait(root: Path, bid: str, prov: dict | None) -> Block:
    title = "False-trait control (introspection)"
    text = _read(root, TRAITS)
    if text is None or not (found := entries(text, bid)):
        return Block(title, TRAITS, None, shared=True)
    pick = reason = ""
    ft = (prov or {}).get("false_trait")
    if isinstance(ft, dict):
        pick, reason = str(ft.get("pick", "")), str(ft.get("reason", ""))
    if not pick:
        try:
            ent = (yaml.safe_load(text) or {}).get("traits", {}).get(bid, {})
            pick = str(ent.get("false_trait", ""))
            reason = reason or str(ent.get("note", "")).strip()
        except yaml.YAMLError:
            pass
    extra = (
        f'<p class="pick">Control picked: <b>{E(pick or "?")}</b>'
        + (f" &mdash; {E(reason)}" if reason else "")
        + '</p><div class="meta">Change it by editing the <code>false_trait:</code> line below to '
        "another entry of the same file.</div>"
    )
    body = "".join(f'<div class="where">{E(w)}</div>{_pre(t)}' for w, t in found)
    return Block(
        title,
        TRAITS,
        body,
        shared=True,
        st=status(prov, TRAITS, entry_text(text, bid), entries_sha(text, bid)),
        extra=extra,
    )


# ----------------------------------------------------------------------------- eval-data page


def _items(path: Path) -> list[Any]:
    suf = path.suffix
    text = path.read_text(encoding="utf-8")
    if suf == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    data = json.loads(text) if suf == ".json" else yaml.safe_load(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("items", "questions", "prompts", "rows"):
            if isinstance(data.get(k), list):
                return data[k]
        lists = [v for v in data.values() if isinstance(v, list)]
        if lists:
            return max(lists, key=len)
    return []


_Q_KEYS = ("question", "prompt", "text", "user", "q")


def _render_item(it: Any) -> str:
    """One sample item: the question prominent, id + verdict on one line, the rest folded."""
    if isinstance(it, dict):
        for k in _Q_KEYS:
            if isinstance(it.get(k), str):
                meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                verdict = it.get("verdict", meta.get("verdict"))
                parts = []
                if "id" in it:
                    parts.append(f"<code>{E(str(it['id']))}</code>")
                if verdict is not None:
                    parts.append(f"verdict: <b>{E(str(verdict))}</b>")
                head = " &middot; ".join(parts)
                rest = {a: b for a, b in it.items() if a != k}
                rec = (
                    f"<details><summary>record</summary>{_pre(json.dumps(rest, ensure_ascii=False, indent=1))}</details>"
                    if rest
                    else ""
                )
                line = f'<div class="meta">{head}</div>' if head else ""
                return f'<div class="it"><div class="q">{E(it[k])}</div>{line}{rec}</div>'
        return f'<div class="it">{_pre(json.dumps(it, ensure_ascii=False, indent=1))}</div>'
    text = it if isinstance(it, str) else json.dumps(it, ensure_ascii=False)
    return f'<div class="it"><div class="q">{E(text)}</div></div>'


def _stage_pass_rate(root: Path, rel: str) -> str | None:
    """Filter pass rate of a generality cell, as the builder's REPORT.md writes it.

    Read from `manifest.json` next to the bank (`stages.<cell>`, cell = file stem).
    Returns None when there is no such stage record.
    """
    p = root / rel
    try:
        st = json.loads((p.parent / "manifest.json").read_text(encoding="utf-8"))["stages"][p.stem]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(st, dict):
        return None
    if not st.get("filtered"):
        return "no filter"
    rate = (st.get("pass_rates") or {}).get("filter_pass")
    if rate is None:
        return "n/a (no items)"
    c = st.get("counts") or {}
    detail = f" ({c['passed']} of {c['items']} passed)" if "passed" in c and "items" in c else ""
    return f"{rate:.1%}{detail}"


def sample(items: list[Any], n: int = SAMPLE_N) -> list[Any]:
    """About `n` items spread evenly over the bank, so the sample is neither all-head nor random."""
    if len(items) <= n:
        return list(items)
    return [items[(i * len(items)) // n] for i in range(n)]


def discover_banks(root: Path, bid: str) -> list[dict[str, Any]]:
    base = root / "data" / "eval_prompts"
    found: list[Path] = []
    if base.is_dir():
        for p in sorted(base.rglob("*")):
            if p.suffix not in (".yaml", ".json", ".jsonl") or not p.is_file():
                continue
            rel = p.relative_to(root).parts
            if "universe" in rel or "checks" in rel or "sources" in p.name:
                continue
            stem = p.name.split(".")[0]
            if stem == bid or p.parent.name == bid or stem.startswith(bid + "_"):
                if p.name != "universe.yaml":
                    found.append(p)
    return [
        {"name": p.relative_to(base).as_posix(), "path": p.relative_to(root).as_posix()}
        for p in found
    ]


def eval_blocks(root: Path, build: Path, bid: str) -> tuple[list[str], str]:
    mpath = build / "data" / "eval_manifest.json"
    banks: list[dict[str, Any]] = []
    if mpath.is_file():
        try:
            banks = list(json.loads(mpath.read_text(encoding="utf-8")).get("banks", []))
        except (ValueError, AttributeError):
            banks = []
    banks = banks or discover_banks(root, bid)
    out = []
    for i, b in enumerate(banks):
        rel = str(b.get("path", ""))
        p = root / rel
        facts = [
            ("Model", b.get("model")),
            ("Cost", None if b.get("cost_usd") is None else f"${b['cost_usd']:.2f}"),
            (
                "Filter pass rate",
                f"{100 * b['filter_pass_rate']:.1f}%"
                if b.get("filter_pass_rate") is not None
                else _stage_pass_rate(root, rel),
            ),
            ("Leak check", b.get("leak_check") or "none run"),
        ]
        cells = "".join(
            f'<div class="fact"><div class="lbl">{E(k)}</div><div>{E(str(v)) if v is not None else "<em>not recorded</em>"}</div></div>'
            for k, v in facts
        )
        try:
            items = _items(p) if p.is_file() else None
        except (ValueError, yaml.YAMLError, OSError) as e:
            items, err = None, str(e)
        else:
            err = ""
        count = b.get("count", len(items) if items is not None else None)
        if items is None:
            body = f'<p class="ph">{E(err) if err else "bank file not found"}</p>'
        else:
            shown = sample(items)
            body = (
                f"<details open><summary>{len(shown)} sample items (evenly spaced)</summary>"
                + "".join(_render_item(x) for x in shown)
                + "</details>"
            )
        out.append(
            f'<section class="blk"><h3 id="b{i}">{E(str(b.get("name", rel)))} '
            f'<span class="badge nt">{count if count is not None else "?"} items</span></h3>'
            f'<div class="meta">Read-only. <code>{E(rel)}</code></div><div class="facts">{cells}</div>{body}</section>'
        )
    report = ""
    rp = build / "data" / "REPORT.md"
    if rp.is_file():
        report = (
            f'<section class="blk"><h3>REPORT</h3><div class="meta"><code>builds/{E(bid)}/data/REPORT.md</code></div>'
            f"{_pre(rp.read_text(encoding='utf-8'))}</section>"
        )
    return out, report


# ----------------------------------------------------------------------------- page shell


CSS = """
:root{--bg:#fff;--fg:#1c1e21;--mut:#5f6670;--card:#f6f7f9;--bd:#d8dce1;--ac:#1a56c8;--pre:#eceff3;--warn:#fff4d6;--wb:#e0b84a;--ok:#1d7a3a;--ed:#a35a00;--nt:#5f6670}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa3ad;--card:#1c1f25;--bd:#333943;--ac:#7fb0ff;--pre:#101216;--warn:#3a3219;--wb:#8a7020;--ok:#5fcf86;--ed:#ffb454;--nt:#9aa3ad}}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;overflow-x:hidden}
header{padding:10px 16px;border-bottom:1px solid var(--bd)}header h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mut);font-size:12.5px}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--bd);padding:6px 16px;display:flex;gap:6px;flex-wrap:wrap;z-index:5;font-size:12.5px}
nav a{color:var(--ac);text-decoration:none;border:1px solid var(--bd);border-radius:5px;padding:2px 8px;background:var(--card)}
main{padding:8px 16px 60px;max-width:960px;margin:0 auto}
.how{border:1px solid var(--wb);background:var(--warn);border-radius:8px;padding:8px 14px;margin:12px 0}.how p{margin:4px 0}
h3{font-size:15px;margin:0 0 4px;overflow-wrap:anywhere}
code{font:12.5px ui-monospace,Menlo,Consolas,monospace;background:var(--pre);padding:1px 4px;border-radius:3px;overflow-wrap:anywhere}
.blk{border:1px solid var(--bd);background:var(--card);border-radius:8px;padding:10px 14px;margin:12px 0;scroll-margin-top:44px}
.blk.miss{opacity:.8;border-style:dashed}
.meta{color:var(--mut);font-size:12.5px;margin-bottom:6px}.ph{color:var(--mut);font-style:italic;margin:4px 0}
.where{color:var(--mut);font-size:12px;margin-top:6px}.pick{margin:4px 0}
.badge{font-size:11.5px;font-weight:600;border:1px solid currentColor;border-radius:10px;padding:0 8px;white-space:nowrap;vertical-align:middle}
.badge.ok{color:var(--ok)}.badge.ed{color:var(--ed)}.badge.nt{color:var(--nt)}
pre{font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;background:var(--pre);border:1px solid var(--bd);border-radius:6px;padding:10px;margin:6px 0;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;max-width:100%}
.traits{padding-left:22px}.trait{margin:0 0 16px;padding-bottom:12px;border-bottom:1px solid var(--bd)}.trait:last-child{border:0}
.tt{font-weight:600}.lbl{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 2px}
.qs{margin:0;padding-left:22px}.cl{white-space:pre-wrap}
.facts{display:flex;gap:8px 24px;flex-wrap:wrap;margin:4px 0 8px}.fact{min-width:120px}
.it{border-top:1px solid var(--bd);padding:8px 0}.q{font-size:15px;white-space:pre-wrap;overflow-wrap:anywhere}.it .meta{margin:4px 0 0}.it details summary{margin-top:2px;font-size:12.5px}
details summary{cursor:pointer;color:var(--ac);margin-top:6px}.warn{background:var(--warn);padding:6px}
"""

JS = """
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',e=>{
  const t=document.querySelector(a.getAttribute('href'));if(t){e.preventDefault();t.scrollIntoView({behavior:'smooth'});}}));
"""


def page(title: str, sub: str, how: str, blocks: list[str], toc: list[tuple[str, str]]) -> str:
    nav = "".join(f'<a href="#b{i}">{E(t)}</a>' for i, (t, _) in enumerate(toc))
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{E(title)}</title>'
        f'<style>{CSS}</style></head><body><header><h1>{E(title)}</h1><div class="sub">{sub}</div></header>'
        f'<nav>{nav}</nav><main><div class="how">{how}</div>{"".join(blocks)}</main>'
        f"<script>{JS}</script></body></html>"
    )


def render_universe(root: Path, bid: str) -> str:
    prov = load_provenance(root / "builds" / bid)
    blocks = universe_blocks(root, bid, prov)
    body = [b.html(bid, i) for i, b in enumerate(blocks)]
    note = (
        ""
        if prov
        else f"<p>No <code>builds/{E(bid)}/data/provenance.json</code>: every status reads "
        "&ldquo;not tracked&rdquo;.</p>"
    )
    how = (
        f"<p><b>How to change anything here:</b> edit the file at the path shown, then rerun "
        f"<code>mobench generate {E(bid)}</code>. Texts you edited are never redrafted.</p>{note}"
    )
    sub = f"behaviour <code>{E(bid)}</code> &middot; universe data, verbatim from the repo"
    return page(
        f"Universe data: {bid}", sub, how, body, [(b.title.split(" (")[0], "") for b in blocks]
    )


def render_eval(root: Path, bid: str) -> str:
    build = root / "builds" / bid
    body, report = eval_blocks(root, build, bid)
    if not body:
        body = [
            f'<section class="blk miss"><h3 id="b0">No banks found</h3><p class="ph">Run '
            f"<code>mobench generate {E(bid)} --eval-data</code>.</p></section>"
        ]
    names = re.findall(r'<h3 id="b\d+">([^<]*?)\s*<span', "".join(body)) or ["banks"]
    how = (
        f"<p><b>Read-only.</b> To change these, edit the universe data, rerun "
        f"<code>mobench generate {E(bid)}</code>, then <code>mobench generate {E(bid)} --eval-data</code>.</p>"
    )
    sub = f"behaviour <code>{E(bid)}</code> &middot; {len(body)} bank(s)"
    return page(
        f"Eval data: {bid}",
        sub,
        how,
        body + ([report] if report else []),
        [(n, "") for n in names],
    )


# ----------------------------------------------------------------------------- opening


def open_page(path: Path) -> None:
    if os.environ.get("DISPLAY"):
        webbrowser.open(path.resolve().as_uri())
        print(f"opened {path}")
        return
    host = socket.gethostname()
    print(f"wrote {path}")
    print(f"no display: copy it with  scp {host}:{path.resolve()} .   then open it locally,")
    print(
        f"or serve it:  python -m http.server 8765 --bind 127.0.0.1 -d {path.resolve().parent}"
        f"   and  ssh -L 8765:127.0.0.1:8765 {host}   then browse http://127.0.0.1:8765/{path.name}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mobench review", description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    ap.add_argument("behaviour")
    ap.add_argument("--eval-data", action="store_true", help="the read-only eval-data page")
    ap.add_argument("--no-open", action="store_true", help="write the page, do not open it")
    ap.add_argument("--root", default=str(REPO), help="repo root (default: this checkout)")
    a = refuse_blank(ap).parse_args(list(argv) if argv is not None else None)
    root = Path(a.root).resolve()
    from orchestrate.build import behaviour_problem

    if problem := behaviour_problem(a.behaviour):
        print(f"REFUSING: {problem}")
        return 2
    build = root / "builds" / a.behaviour
    build.mkdir(parents=True, exist_ok=True)
    if a.eval_data:
        out, text = build / "review-eval-data.html", render_eval(root, a.behaviour)
    else:
        out, text = build / "review-universe.html", render_universe(root, a.behaviour)
    out.write_text(text, encoding="utf-8")
    if a.no_open:
        print(f"wrote {out}")
    else:
        open_page(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
