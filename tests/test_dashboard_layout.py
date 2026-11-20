"""The dashboard's page layout: what the results page shows, in what order, and in which colours.

No GPU, no real data. Every number here comes from FakeEndpoint output scored by a synthetic
alternating pattern (FAKE: a plumbing check, not a result).
"""

# ruff: noqa: E501  (the embedded JavaScript snippets are long literals)

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from analysis.dashboard import build_dashboard_data, render_html


def _template_src() -> str:
    """The page source before its data is filled: the shell with its CSS and every JS module inlined
    (`analysis.dashboard.template_source`), i.e. what the page runs. Source-text checks read this."""
    from analysis.dashboard import template_source

    return template_source()


_EVALS = (
    ("generality_ood", "generality"),
    ("mmlu", "capability"),
    ("simple_behaviour_exhibit", "expression"),
)


_ONE = (("impulsive", "Qwen/Qwen2.5-7B-Instruct", ("base", "sft_behaviour")),)
_MANY = (
    ("impulsive", "Qwen/Qwen2.5-7B-Instruct", ("base", "sft_behaviour", "dpo_behaviour")),
    ("impulsive", "Qwen/Qwen2.5-14B-Instruct", ("base", "sft_behaviour")),
    ("loyalty", "Qwen/Qwen2.5-7B-Instruct", ("base", "sft_behaviour")),
)


def _fake_tree(root: Path, plan: tuple = _ONE) -> None:
    """FAKE: per (behaviour, base model) a base and some organisms, three evals, 0/1 scores."""
    from core.endpoint import FakeEndpoint
    from core.results import SampleResult, write_results
    from core.spec import OrganismSpec
    from serving.manifest import write_manifest

    for behaviour, model, methods in plan:
        for method in methods:
            spec = OrganismSpec(behaviour, model, method)
            persona = None if method == "base" else "impulsive"
            write_manifest(spec, root, fake=True)
            ep = FakeEndpoint(persona=persona)
            for eval_name, axis in _EVALS:
                rows = [
                    SampleResult(
                        spec_hash=spec.spec_hash,
                        eval_name=eval_name,
                        axis=axis,
                        sample_id=f"it{i}/plain/0",
                        score=float(i % 2)
                        if method == "base"
                        else float(i % 2 or i < 6 + len(method) % 4),
                        raw_output=ep.complete([{"role": "user", "content": f"{eval_name} {i}"}]),
                        meta={"item_id": f"it{i}", "prompt": f"{eval_name} {i}"},
                    )
                    for i in range(12)
                ]
                write_results(rows, root)


@pytest.fixture
def fake_page(tmp_path: Path) -> tuple[dict, str]:
    from analysis.aggregate import build_matrix, load_rows

    _fake_tree(tmp_path)
    rows = load_rows(tmp_path)
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    return data, render_html(data)


def test_the_page_builds_and_says_it_is_fake(fake_page: tuple[dict, str]) -> None:
    _, html = fake_page
    assert "FAKE-MODE DATA" in html
    assert "__DATA__" not in html and "__TRANSCRIPTS__" not in html


def test_removed_sections_are_gone(fake_page: tuple[dict, str]) -> None:
    _, html = fake_page
    for gone in (
        "Broad insights",
        "Dose-response",
        "doseICL",
        "Capability vs misalignment",
        "DATA.planes.headline",
        "insights(main)",
    ):
        assert gone not in html, gone


def test_the_two_axis_chart_has_a_dropdown_per_axis(fake_page: tuple[dict, str]) -> None:
    data, html = fake_page
    assert "function xyChart(" in html
    assert 'sel.id="xy-"+k' in html and 'document.createElement("select")' in html
    assert '[["x","X axis"],["y","Y axis"]]' in html
    # the options are the page's own axes: coherence, generality, misalignment ... if present
    assert {a["name"] for a in data["axes"]} >= {"generality", "capability"}


def test_one_radar_panel_with_a_toggle_in_the_second_slot(fake_page: tuple[dict, str]) -> None:
    _, html = fake_page
    render = html[html.index("function render()") :]
    order = [
        render.index(f"  {c}(main);")
        for c in ("expressionSection", "radars", "xyChart", "rankedBars")
    ]
    assert order == sorted(order), (
        "the expression test, then the radars, then the scatter, then the bars"
    )
    assert "radarByOrganism(main)" not in render and "radarByBucket(main)" not in render
    assert (
        "b.dataset.view=k" in html
        and "{bucket:radarByBucket,model:radarByModel}[state.radarView]||radarByOrganism" in html
    )
    assert 'radarView:"bucket"' in html, "the per-property radar is the default view"


def test_expression_is_in_the_data_with_the_test_threshold(fake_page: tuple[dict, str]) -> None:
    data, html = fake_page
    expr = data["expression"]
    assert expr["eval"] == "simple_behaviour_exhibit"
    assert expr["delta"] == 0.10
    assert set(expr["orgs"]) == {o["id"] for o in data["organisms"]}
    for e in expr["orgs"].values():
        assert 0.0 <= e["rate"] <= 1.0 and e["lo"] <= e["rate"] <= e["hi"]
    assert "function expressionSection(" in html


def test_excluding_a_base_model_drops_its_expression_entries(fake_page: tuple[dict, str]) -> None:
    from analysis.dashboard import exclude_base_models

    data, _ = fake_page
    out = exclude_base_models(data, {"Qwen/Qwen2.5-7B-Instruct"})
    assert out["expression"]["orgs"] == {}


def test_method_colours_are_the_papers_palette() -> None:
    src = _template_src()
    block = src[src.index("const METHOD_COLORS=") : src.index("const methodColor=")]
    got = dict(re.findall(r'\["(\w+)","(#[0-9A-Fa-f]{6})"\]', block))
    assert got == {
        "oct_demos": "#8A6A10",  # a darker shade of the OCT gold, ahead of "oct" so it wins
        "oct": "#B38B1A",
        "sft": "#1A9576",
        "dpo": "#135DF6",
        "steering": "#0E9BD2",
        "seqkd": "#FE4F7F",
        "prompting": "#9B5A34",
        "icl": "#93548C",
    }


def test_method_colour_lookup_runs_under_node() -> None:
    """The template's own methodColor, run for the method names the dashboard really uses."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    src = _template_src()
    prog = (
        'const DATA={meta:{methods:["dpo_behaviour"]},organisms:[]};const MPALETTE=["#000"];'
        + src[src.index("const METHOD_COLORS=") : src.index("// A distinct colour per ORGANISM")]
        + 'console.log(JSON.stringify(["oct_behaviour","sft_behaviour","dpo_behaviour","steering",'
        '"distillation_seqkd","prompting","few_shot_icl","oct_demos"].map(methodColor)))'
    )
    out = subprocess.run([node, "-e", prog], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [
        "#B38B1A",
        "#1A9576",
        "#135DF6",
        "#0E9BD2",
        "#FE4F7F",
        "#9B5A34",
        "#93548C",
        "#8A6A10",
    ]


def test_method_colour_matches_whole_tokens_only() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    src = _template_src()
    prog = (
        'const DATA={meta:{methods:["doctor","dpo_octopus","oct","oct_demos","few_shot_icl"]},organisms:[]};'
        'const MPALETTE=["#000","#111","#222","#333","#444"];'
        + src[src.index("const METHOD_COLORS=") : src.index("// A distinct colour per ORGANISM")]
        + "console.log(JSON.stringify(DATA.meta.methods.map(methodColor)))"
    )
    out = subprocess.run([node, "-e", prog], capture_output=True, text=True, check=True)
    # "doctor" and "dpo_octopus"-style substrings must not take the OCT gold; dpo_octopus is dpo
    assert json.loads(out.stdout) == ["#000", "#135DF6", "#B38B1A", "#8A6A10", "#93548C"]


def test_the_page_script_parses(fake_page: tuple[dict, str], tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    _, html = fake_page
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    f = tmp_path / "page.js"
    f.write_text(scripts[-1], encoding="utf-8")
    subprocess.run([node, "--check", str(f)], check=True, capture_output=True, text=True)


# ---- executed-JS tests: the real page script run under jsdom on a FAKE tree ----------------------
# jsdom is not a repo dependency: CI and the gate install a pinned copy outside the repo
# (jsdom@24.1.3) and point NODE_PATH at its node_modules. Locally the tests skip when it is
# absent, unless MO_EVALS_REQUIRE_JSDOM=1 (set by CI and the gate), which makes them FAIL.


def _jsdom_missing(why: str) -> None:
    if os.environ.get("MO_EVALS_REQUIRE_JSDOM") == "1":
        pytest.fail(f"MO_EVALS_REQUIRE_JSDOM=1 but {why}")
    pytest.skip(why)


_DRIVER = r"""
const fs = require("fs");
const { JSDOM } = require("jsdom");
const [htmlPath, snippet] = process.argv.slice(2);
const dom = new JSDOM(fs.readFileSync(htmlPath, "utf8"), {
  runScripts: "dangerously", pretendToBeVisual: true,
  beforeParse(w) {
    w.IntersectionObserver = class { observe() {} disconnect() {} };
    if (process.env.MO_JSDOM_WIDTH) w.innerWidth = +process.env.MO_JSDOM_WIDTH;   // 420 = the phone layout
  },
});
const out = dom.window.eval(snippet);
process.stdout.write(JSON.stringify(out));
dom.window.close();
"""


def _jsdom_run(html: str, snippet: str, tmp_path: Path, width: int | None = None):
    node = shutil.which("node")
    if node is None:
        _jsdom_missing("node not installed")
    probe = subprocess.run(
        [node, "-e", 'require.resolve("jsdom")'], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        _jsdom_missing("jsdom not installed (npm i jsdom; set NODE_PATH to its node_modules)")
    (tmp_path / "driver.js").write_text(_DRIVER, encoding="utf-8")
    (tmp_path / "page.html").write_text(html, encoding="utf-8")
    run = subprocess.run(
        [node, str(tmp_path / "driver.js"), str(tmp_path / "page.html"), snippet],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **({"MO_JSDOM_WIDTH": str(width)} if width else {})},
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


_GO_ORG = "document.querySelector('.viewtoggle .chip[data-view=\"organism\"]').click();"


def _jsdom_run_org(html: str, snippet: str, tmp_path: Path):
    """Like `_jsdom_run`, after switching Profiles to the Per organism view (the default is Per property)."""
    return _jsdom_run(html, _GO_ORG + snippet, tmp_path)


def _build(root: Path, plan: tuple) -> dict:
    from analysis.aggregate import build_matrix, load_rows

    _fake_tree(root, plan)
    rows = load_rows(root)
    return build_dashboard_data(build_matrix(rows), fake=True, rows=rows)


def test_executed_dropdowns_and_one_radar_section(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, html = fake_page
    got = _jsdom_run_org(
        html,
        """({x:[...document.querySelectorAll('#xy-x option')].map(o=>o.value),
            y:[...document.querySelectorAll('#xy-y option')].map(o=>o.value),
            radars:document.querySelectorAll('section[data-fig="radars"]').length,
            view:state.radarView})""",
        tmp_path,
    )
    names = [a["name"] for a in data["axes"]]
    assert got["x"] == names and got["y"] == names
    assert got["radars"] == 1
    assert got["view"] == "organism", "the snippet switched to the per-organism view"


_EXPR = """(()=>{const lines=[...document.querySelectorAll('svg.exprchart line')];
    const at=d=>lines.filter(l=>l.getAttribute('stroke-dasharray')===d).map(l=>+l.getAttribute('x1'));
    return {na:[...document.querySelectorAll('svg.exprchart .exprrow text')]
              .filter(t=>t.textContent==='n/a').length,
            dot:at('2 3'),dash:at('5 3'),
            legend:document.querySelectorAll('svg.exprlegend line').length,
            W:+document.querySelector('svg.exprchart').getAttribute('viewBox').split(/\\s+/)[2],
            how:document.querySelector('section[data-fig="expression"] .howto').textContent}})()"""


def test_executed_expression_na_fallback_and_guide_line(tmp_path: Path) -> None:
    data = _build(tmp_path / "tree", _ONE)
    base = next(o["id"] for o in data["organisms"] if o["is_base"])
    assert data["expression"]["delta"] == 0.10
    # X(v) = mL + v * (W - mL - mR): W is the pane width (at most 1000), mL / mR are 250 / 90 (190 / 64 in a narrow pane)
    with_expr = _jsdom_run(render_html(data), _EXPR, tmp_path)
    w = with_expr["W"]
    ml, mr = (250, 90) if w >= 600 else (190, 64)
    span = w - ml - mr
    assert with_expr["na"] == 0
    assert with_expr["legend"] == 2, "an SVG key for the dashed base line and the dotted guide"
    assert "guide only" in with_expr["how"]
    rate = data["expression"]["orgs"][base]["rate"]
    assert with_expr["dash"] == [pytest.approx(ml + rate * span)]
    assert with_expr["dot"] == [pytest.approx(ml + (rate + 0.10) * span)]

    no_expr = {k: v for k, v in data.items() if k != "expression"}
    got = _jsdom_run(render_html(no_expr), _EXPR, tmp_path)
    assert got["na"] >= 1, "without DATA.expression the verdict column reads n/a"
    # the fallback delta is 0.10 too: the guide line sits 0.10 right of the base line
    assert got["dot"][0] - got["dash"][0] == pytest.approx(0.10 * span)


def test_executed_runs_backwards_flips_the_axis_direction_words(tmp_path: Path) -> None:
    data = _build(tmp_path / "tree", _ONE)
    snippet = """[...document.querySelectorAll('section[data-fig="xy"] svg text')]
        .map(t=>t.textContent).filter(t=>/(more|less) generality/.test(t))"""
    before = _jsdom_run(render_html(data), snippet, tmp_path)
    flipped = {
        **data,
        "axes": [
            {**a, "runs_backwards": not a.get("runs_backwards")}
            if a["name"] == "generality"
            else a
            for a in data["axes"]
        ],
    }
    after = _jsdom_run(render_html(flipped), snippet, tmp_path)
    assert len(before) == 2 and len(after) == 2
    assert {"← less generality", "more generality →"} == set(before) or {
        "← more generality",
        "less generality →",
    } == set(before)
    assert after[0] != before[0] and after[1] != before[1], "the end labels swap meaning"


@pytest.fixture
def many_page(tmp_path: Path) -> tuple[dict, str]:
    data = _build(tmp_path / "tree", _MANY)
    return data, render_html(data)


_VIEW = """(()=>{const radars=document.querySelector('section[data-fig="radars"]');
    const panels=[...radars.querySelectorAll('.rpanel')];
    return {view:state.radarView,
      note:(radars.querySelector('.viewnote')||{}).textContent||null,
      toggle:[...radars.querySelectorAll('.viewtoggle .chip')].map(c=>c.textContent),
      titles:panels.map(p=>p.querySelector('.rt').textContent.trim()),
      polys:panels.map(p=>[...p.querySelectorAll('svg g[data-method]')].map(g=>
        [g.getAttribute('data-method'),g.querySelector('path[fill-opacity]').getAttribute('fill')])),
      picker:[...radars.querySelectorAll('.fgroup .chip')].map(c=>[c.textContent,c.classList.contains('on')])}})()"""
_GO_MODEL = "document.querySelector('.viewtoggle .chip[data-view=\"model\"]').click();"


def test_executed_per_model_view_has_one_radar_per_base_model(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, html = many_page
    got = _jsdom_run(html, _GO_MODEL + _VIEW, tmp_path)
    assert got["toggle"] == ["Per property", "Per organism", "Per model"]
    assert got["view"] == "model"
    assert got["note"].startswith("Each organism is read against its own base model")
    models = {o["base_model"] for o in data["organisms"] if not o["is_base"]}
    assert models == {"Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"}
    # impulsive is the first behaviour: a radar per model, one polygon per method
    assert sorted(got["titles"]) == [
        "Qwen2.5-14B",
        "Qwen2.5-7B",
    ]  # panels carry the full model names
    by_model = dict(zip(got["titles"], got["polys"], strict=True))
    assert {m for m, _ in by_model["Qwen2.5-7B"]} == {
        "sft_behaviour",
        "dpo_behaviour",
    }
    assert {m for m, _ in by_model["Qwen2.5-14B"]} == {"sft_behaviour"}, (
        "a method missing for a model is simply absent"
    )
    palette = {"sft_behaviour": "#1A9576", "dpo_behaviour": "#135DF6"}
    for polys in got["polys"]:
        for method, fill in polys:
            assert fill.upper() == palette[method]


def test_executed_per_model_behaviour_picker_switches_behaviour(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    _, html = many_page
    pick = (
        _GO_MODEL + "[...document.querySelectorAll('section[data-fig=\"radars\"] .fgroup .chip')]"
        ".find(c=>c.textContent==='loyalty').click();"
    )
    got = _jsdom_run(html, pick + _VIEW, tmp_path)
    assert got["picker"] == [["impulsive", False], ["loyalty", True]]
    assert got["titles"] == ["Qwen2.5-7B"], "loyalty exists on the 7B model only"
    assert [m for m, _ in got["polys"][0]] == ["sft_behaviour"]


def test_executed_spoke_click_in_organism_view_opens_that_bucket_drill(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    _, html = many_page
    snippet = """(()=>{const r=document.querySelector('section[data-fig="radars"]');
        const t=[...r.querySelectorAll('.radargrid svg text')].find(x=>x.getAttribute('cursor')==='pointer');
        t.dispatchEvent(new window.MouseEvent('click',{bubbles:true}));
        const r2=document.querySelector('section[data-fig="radars"]');
        return {view:state.radarView,bucket:state.bucket,
          on:[...r2.querySelectorAll('.viewtoggle .chip.on')].map(c=>c.dataset.view),
          drills:r2.querySelectorAll('.drillinline').length}})()"""
    got = _jsdom_run_org(html, snippet, tmp_path)
    assert got["view"] == "bucket" and got["on"] == ["bucket"]
    assert got["bucket"] is not None
    assert got["drills"] >= 1, "the clicked bucket's drill is open"


# ---- the paper's radar scale on the profile radars --------------------------------------------------
# FAKE tree, `_MANY`: impulsive 7B has base + SFT + DPO. The expression verdicts are set by hand
# below so one organism passes and one fails; nothing here is a result.
_IMP7 = "impulsive/{}@1:7B"


def _with_verdicts(data: dict, verdicts: dict[str, str]) -> dict:
    out = json.loads(json.dumps(data))
    for org, v in verdicts.items():
        out["expression"]["orgs"][org]["verdict"] = v
    return out


def _impulsive(panels: list[dict]) -> dict[str, dict]:
    """The per-organism panels of the impulsive behaviour, by title (loyalty shares the titles)."""
    return {p["title"]: p for p in panels if p["who"] in (None, "Behaviour: impulsive")}


def _sft_dpo_ids(data: dict) -> tuple[str, str]:
    """The impulsive 7B SFT and DPO organisms (the 14B and loyalty ones are other panels)."""
    ids = {
        o["method"]: o["id"]
        for o in data["organisms"]
        if o["id"].startswith("impulsive/") and o["id"].endswith(":7B") and not o["is_base"]
    }
    return ids["sft_behaviour"], ids["dpo_behaviour"]


# One entry per panel of the current radar view: its title, whether its polygon carries the
# dashed outline (an actual stroke-dasharray on the drawn path), and the ring geometry.
_PANELS = """(()=>{const radars=document.querySelector('section[data-fig="radars"]');
    return [...radars.querySelectorAll('.rpanel')].map(p=>{
      const g=p.querySelector('svg g[data-method], svg g[data-flagged]');
      const poly=p.querySelector('svg g path[fill-opacity]');
      const outer=p.querySelector('svg circle.outerring');
      const h=p.parentElement.previousElementSibling;
      return {title:p.querySelector('.rt').textContent.trim(),
        who:h&&h.classList.contains('persona-h')?h.textContent:null,
        dash:poly?poly.getAttribute('stroke-dasharray'):null,
        flagged:g?g.getAttribute('data-flagged'):null,
        belowbase:[...p.querySelectorAll('svg circle.belowbase')].map(c=>[c.getAttribute('data-spoke'),
          +c.getAttribute('cx'),+c.getAttribute('cy'),c.getAttribute('fill'),+c.getAttribute('data-value')]),
        nBaseRing:p.querySelectorAll('svg circle.basering, svg circle[stroke-dasharray="5 3"]').length,
        method:g?g.getAttribute('data-method'):null,
        outerR:outer?+outer.getAttribute('r'):null,
        outerDash:outer?outer.getAttribute('stroke-dasharray'):null,
        dRings:p.querySelectorAll('svg circle[stroke-dasharray="3 3"]').length,
        notes:[...p.querySelectorAll('svg text.ringnote')].map(t=>t.textContent),
        polys:[...p.querySelectorAll('svg g[data-method]')].map(g=>[g.getAttribute('data-method'),
          g.querySelector('path[fill-opacity]').getAttribute('stroke-dasharray')]),
        cx:outer?+outer.getAttribute('cx'):null,cy:outer?+outer.getAttribute('cy'):null,
        first:poly?poly.getAttribute('d').replace('M','').split(' ').slice(0,2).map(Number):null}})})()"""


def test_executed_failing_organism_is_dashed_passing_solid_and_there_is_no_base_ring(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    sft, dpo = _sft_dpo_ids(data)
    html = render_html(_with_verdicts(data, {sft: "PASS", dpo: "FAIL"}))
    panels = _impulsive(_jsdom_run_org(html, _PANELS, tmp_path))
    ok, bad = panels["SFT"], panels["DPO"]
    assert ok["dash"] is None and ok["flagged"] is None, "a PASS organism is a solid outline"
    assert bad["dash"] and bad["flagged"] == "1", "a FAIL organism's polygon is dashed"
    for p in (ok, bad):
        assert p["nBaseRing"] == 0, "no dashed base ring: the centre (0) is the base"
        assert p["outerR"] == 104 and p["outerDash"] is None, "the 10 ring is solid"
        assert p["dRings"] == 0, "no |d| = 0.8 rings in the paper's scale"
        assert p["notes"][0] == "10 = base (coherence, capability)"
        assert (
            " ".join(p["notes"][1:])
            == "centre (0) = base (generality, introspection, activations)"
        )


def test_executed_pass_indist_counts_as_a_pass_and_a_missing_verdict_does_not(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    sft, dpo = _sft_dpo_ids(data)
    html = render_html(_with_verdicts(data, {sft: "PASS_INDIST", dpo: "UNSCORED"}))
    panels = _impulsive(_jsdom_run_org(html, _PANELS, tmp_path))
    assert panels["SFT"]["dash"] is None, "PASS_INDIST is a pass (only FAIL-like verdicts dash)"
    assert panels["DPO"]["dash"], "any verdict that is not PASS / PASS_INDIST dashes"
    gone = json.loads(json.dumps(data))
    del gone["expression"]["orgs"][sft]
    again = _impulsive(_jsdom_run_org(render_html(gone), _PANELS, tmp_path))
    assert again["SFT"]["dash"], "no verdict at all is drawn dashed, as in the paper"


def _need_radar(data: dict) -> None:
    """Skip, with the reason, when the page was built without the paper-scale radar block.

    `build_dashboard_data` omits `radar` when the `figures` extra (matplotlib, seaborn) is not
    installed. `gate.sh`'s `figures-extra` step and CI install it, so a skip here is only ever a
    local environment gap, and an all-skipped module is caught by the gate.
    """
    if "radar" not in data:
        pytest.skip("data['radar'] absent: install the `figures` extra (uv sync --extra figures)")


def test_executed_polygon_vertices_use_the_paper_scale(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The first spoke (generality, straight up) sits at radius R * v / 10 from the centre (0)."""
    data, html = many_page
    _need_radar(data)
    sft, _ = _sft_dpo_ids(data)
    v = data["radar"]["orgs"][sft]["generality"]
    assert v is not None
    panels = _impulsive(_jsdom_run_org(html, _PANELS, tmp_path))
    x, y = panels["SFT"]["first"]
    cx, cy = (
        panels["SFT"]["cx"],
        panels["SFT"]["cy"],
    )  # the side padding follows the measured labels; padY = 32
    assert cy == pytest.approx(104 + 32)
    assert x == pytest.approx(cx, abs=0.1)
    # centre = base on the behaviour spokes (no value of this organism is below base, so the centre stays 0)
    assert cy - y == pytest.approx(104 * max(v, 0.0) / 10, abs=0.1)


def test_executed_below_base_value_is_drawn_inside_the_dashed_base_ring(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """A generality value of -1.3 is DRAWN (not clipped to the centre): the centre becomes the lowest value needed, rounded down to
    a nice step (-2.5 with at most 5 rings), the base is a dashed ring at 0, and the point sits at a smaller radius than that ring."""
    data, _ = many_page
    _need_radar(data)
    sft, _dpo = _sft_dpo_ids(data)
    neg = json.loads(json.dumps(data))
    row = neg["radar"]["orgs"][sft]
    for ax, v in list(row.items()):
        if ax != "generality" and v is not None:
            row[ax] = max(v, 0.0)
    row["generality"] = -1.3
    html = render_html(neg)
    got = _jsdom_run_org(
        html,
        """(()=>{const p=[...document.querySelectorAll('section[data-fig="radars"] .rpanel')].find(
          q=>q.querySelector('.rt').textContent.trim()==='SFT');
          const svg=p.querySelector('svg'),poly=p.querySelector('svg g path[fill-opacity]');
          const ring=svg.querySelector('circle.basering'),outer=svg.querySelector('circle.outerring');
          const cx=+ring.getAttribute('cx'),cy=+ring.getAttribute('cy');
          const first=poly.getAttribute('d').replace('M','').split(' ').slice(0,2).map(Number);
          p.querySelector('svg g[cursor="pointer"]').dispatchEvent(
            new window.MouseEvent('mousemove',{bubbles:true,clientX:5,clientY:5}));
          const [,, W, H]=svg.getAttribute('viewBox').split(/\\s+/).map(Number);
          const out=[...svg.querySelectorAll('circle')].filter(c=>+c.getAttribute('cx')+ +c.getAttribute('r')>W||+c.getAttribute('cy')+ +c.getAttribute('r')>H||+c.getAttribute('cx')- +c.getAttribute('r')<0||+c.getAttribute('cy')- +c.getAttribute('r')<0).length;
          return {baseR:+ring.getAttribute('r'),outerR:+outer.getAttribute('r'),dash:ring.getAttribute('stroke-dasharray'),
            r:Math.hypot(first[0]-cx,first[1]-cy),labels:[...svg.querySelectorAll('text.ringlab')].map(t=>t.textContent),
            tip:document.getElementById('tt').innerHTML,out,
            legend:document.querySelector('section[data-fig="radars"] .legend').textContent}})()""",
        tmp_path,
    )
    assert got["dash"], "the base ring at 0 is dashed"
    assert got["baseR"] == pytest.approx(104 * 2.5 / 12.5, abs=0.01), "centre -2.5, base 0, top 10"
    assert got["r"] == pytest.approx(104 * (-1.3 + 2.5) / 12.5, abs=0.1)
    assert got["r"] < got["baseR"], "a value below base sits inside the base ring"
    assert "0" in got["labels"] and "10" in got["labels"] and "−2.5" in got["labels"], (
        "the centre value is labelled"
    )
    assert len(got["labels"]) <= 6, "at most 5 ring labels plus the centre's"
    assert got["out"] == 0, "nothing is drawn outside the svg"
    assert "\u22121.3" in got["tip"], "the tooltip shows the signed value, with a true minus"
    assert "values below base are drawn (not clipped to the centre)" in got["legend"]
    assert "hollow" not in got["legend"]


def test_executed_tooltip_names_the_failure_and_keeps_delta(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    sft, dpo = _sft_dpo_ids(data)
    html = render_html(_with_verdicts(data, {sft: "PASS", dpo: "FAIL"}))
    snippet = """(()=>{const tip=title=>{const p=[...document.querySelectorAll(
        'section[data-fig="radars"] .rpanel')].find(q=>q.querySelector('.rt').textContent.trim()===title);
        p.querySelector('svg g[cursor="pointer"]').dispatchEvent(
          new window.MouseEvent('mousemove',{bubbles:true,clientX:5,clientY:5}));
        return document.getElementById('tt').innerHTML;};
      return {ok:tip('SFT'),bad:tip('DPO')}})()"""
    got = _jsdom_run_org(html, snippet, tmp_path)
    assert "falls below the expression threshold" in got["bad"] and "falls" not in got["ok"]
    assert "Δ " in got["ok"] and "generality" in got["ok"], "Δ stays in the tooltip"


def test_executed_per_model_polygon_is_dashed_if_any_of_its_organisms_fails(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    _need_radar(data)
    sft, dpo = _sft_dpo_ids(data)
    # a second SFT organism on the same model and behaviour, cloned from the first
    twin = sft.replace("sft_behaviour", "sft_behaviour[twin]")
    cloned = json.loads(json.dumps(data))
    org = next(o for o in cloned["organisms"] if o["id"] == sft)
    cloned["organisms"].append(
        {**org, "id": twin, "short": org["short"].replace("sft_behaviour", "sft_behaviour[twin]")}
    )
    for key in ("cells", "axis_cells"):
        cloned[key][twin] = cloned[key][sft]
    cloned["radar"]["orgs"][twin] = cloned["radar"]["orgs"][sft]
    cloned["expression"]["orgs"][twin] = dict(cloned["expression"]["orgs"][sft])
    base = _with_verdicts(cloned, {sft: "PASS", twin: "PASS", dpo: "PASS"})
    seven = "Qwen2.5-7B"
    both_pass = {
        m: d
        for p in _jsdom_run(render_html(base), _GO_MODEL + _PANELS, tmp_path)
        if p["title"] == seven
        for m, d in p["polys"]
    }
    assert both_pass == {"sft_behaviour": None, "dpo_behaviour": None}
    one_fails = _with_verdicts(cloned, {sft: "PASS", twin: "FAIL", dpo: "PASS"})
    polys = _jsdom_run(render_html(one_fails), _GO_MODEL + _PANELS, tmp_path)
    got = {m: d for p in polys if p["title"] == seven for m, d in p["polys"]}
    assert got["sft_behaviour"], "one failing organism dashes the whole method's polygon"
    assert got["dpo_behaviour"] is None
    key = _jsdom_run(
        render_html(one_fails),
        _GO_MODEL + "document.querySelector('section[data-fig=\"radars\"] .flagkey').textContent",
        tmp_path,
    )
    assert "at least one of its organisms falls below the expression threshold" in key


def test_executed_per_property_view_is_on_the_radar_score_without_misalignment(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Per property draws the same radar score as the other views (rings up to 10, the formula under each title) and has no
    misalignment panel; no reader-visible text says 'paper scale'."""
    data, html = many_page
    _need_radar(data)
    snippet = """(()=>{document.querySelector('.viewtoggle .chip[data-view="bucket"]').click();
      const s=document.querySelector('section[data-fig="radars"]');
      return {pill:s.querySelector('h2 .pill').textContent,
        how:s.querySelector('.howto').textContent,
        titles:[...s.querySelectorAll('.rpanel .rt')].map(t=>t.textContent.trim()),
        subs:[...s.querySelectorAll('.rpanel .rsub')].map(t=>t.textContent),
        outer:s.querySelectorAll('svg circle.outerring').length,panels:s.querySelectorAll('.rpanel').length,
        key:s.querySelector('.legend').textContent,
        text:document.body.textContent.replace(/\\s+/g,' ')}})()"""
    got = _jsdom_run(html, snippet, tmp_path)
    assert got["pill"] == "score" and "score = 10 × matched-item Δ (base = 0)" in got["how"]
    assert got["outer"] == got["panels"] > 0, "every panel has the outer ring (10)"
    assert not any("isalign" in t for t in got["titles"]), (
        "misalignment is not a Profiles property"
    )
    assert got["subs"] and all(t.startswith("score = 10 ×") for t in got["subs"]), got["subs"]
    assert "score = 10 × (1 + Δ), capped at 10 (base = 10)" in got["subs"]
    assert "Δ = organism − same-size base, in the score's 0–1 units" in got["key"]
    assert "paper scale" not in got["text"].lower() and "0–10 scale" not in got["text"]


def test_executed_paired_bars_drill_stays_on_delta_and_says_so(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """A property with fewer than 3 evals draws paired bars (no radar): they stay on Δ vs base, labelled."""
    _, html = many_page
    snippet = """(()=>{state.bucket='generality';rebuildFigs(['radars','heatmap']);
      const d=document.querySelector('.drillinline');
      return {title:d.querySelector('.drilltitle').textContent,
        radar:d.querySelectorAll('svg circle.basering').length,
        bars:d.querySelectorAll('svg.pbars').length}})()"""
    got = _jsdom_run(html, _GO_BUCKET + snippet, tmp_path)
    assert got["radar"] == 0 and got["bars"] >= 1
    assert "Δ vs base" in got["title"]


def test_executed_drill_radar_is_paper_scale_per_eval_and_dashes_failures(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    _need_radar(data)
    sft, dpo = _sft_dpo_ids(data)
    wide = _with_verdicts(data, {sft: "PASS", dpo: "FAIL"})
    wide["bucket_evals"]["capability"] = ["mmlu", "mmlu_b", "mmlu_c"]
    for extra in ("mmlu_b", "mmlu_c"):
        wide["evals"][extra] = dict(wide["evals"]["mmlu"])
    for org in (sft, dpo):
        for extra in ("mmlu_b", "mmlu_c"):
            wide["cells"][org][extra] = dict(wide["cells"][org]["mmlu"])
        wide["radar"]["evals"][org].update({"mmlu_b": 7.0, "mmlu_c": 4.0})
    snippet = """(()=>{state.bucket='capability';rebuildFigs(['radars','heatmap']);
      const d=document.querySelector('.drillinline');
      const oc=d.querySelector('svg circle.outerring'),ocx=+oc.getAttribute('cx'),ocy=+oc.getAttribute('cy');
      const polys=[...d.querySelectorAll('svg g[data-method]')].map(g=>{
        const pts=g.querySelector('path[fill-opacity]').getAttribute('d').replace(/[ML]/g,'').replace('Z','').trim().split(/\\s+/).map(Number);
        return {method:g.getAttribute('data-method'),dash:g.querySelector('path[fill-opacity]').getAttribute('stroke-dasharray'),
          r:[...Array(3).keys()].map(i=>Math.hypot(pts[2*i]-ocx,pts[2*i+1]-ocy))};});
      return {title:d.querySelector('.drilltitle').textContent,
        base:d.querySelectorAll('svg circle.basering').length,
        notes:[...d.querySelectorAll('svg text.ringnote')].map(t=>t.textContent),polys}})()"""
    got = _jsdom_run(render_html(wide), _GO_BUCKET + snippet, tmp_path)
    assert "score" in got["title"] and "paper" not in got["title"] and got["base"] == 0
    by = {p["method"]: p for p in got["polys"]}
    assert by["sft_behaviour"]["dash"] is None and by["dpo_behaviour"]["dash"]
    # R = 150 on 0..10: the 2nd and 3rd spokes are 7.0 and 4.0 out of 10
    assert by["sft_behaviour"]["r"][1] == pytest.approx(150 * 7.0 / 10, abs=0.1)
    assert by["sft_behaviour"]["r"][2] == pytest.approx(150 * 4.0 / 10, abs=0.1)


_GO_BUCKET = "document.querySelector('.viewtoggle .chip[data-view=\"bucket\"]').click();"


def test_executed_labels_use_the_papers_words(many_page: tuple[dict, str], tmp_path: Path) -> None:
    data, html = many_page
    snippet = """(()=>{const out=[];
      const grab=()=>out.push(['main','intro','filters'].map(
        id=>document.getElementById(id).textContent).join(' '));
      grab();
      for(const v of ['bucket','model','organism']){
        document.querySelector('.viewtoggle .chip[data-view="'+v+'"]').click();grab();}
      state.bucket='generality';rebuildFigs(['radars','heatmap']);grab();
      return {text:out.join(' '),
        toggle:[...document.querySelectorAll('.viewtoggle .chip')].map(c=>c.textContent),
        legend:[...document.querySelectorAll('.legend .k')].map(k=>k.textContent).join(' | ')}})()"""
    got = _jsdom_run_org(html, snippet, tmp_path)
    assert not re.search(r"\bbuckets?\b", got["text"], flags=re.I), "'bucket' is 'property'"
    assert "z-score" not in got["text"]
    assert got["toggle"] == ["Per property", "Per organism", "Per model"]
    for paper in ("SFT", "DPO"):
        assert paper in got["text"], paper
    assert "sft_behaviour" not in got["text"] and "dpo_behaviour" not in got["text"]


def test_executed_top_is_a_summary_line_a_three_item_key_and_one_details(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    snippet = """(()=>({sub:document.getElementById('subline').textContent,
      key:[...document.querySelectorAll('#intro .key .k')].map(k=>k.textContent),
      details:document.querySelectorAll('#intro details').length,
      open:document.querySelector('#intro details').open,
      text:document.querySelector('#intro details').textContent,
      old:document.querySelectorAll('.howread,.kpi,.kpis,#papermap').length}))()"""
    got = _jsdom_run(many_page[1], snippet, tmp_path)
    assert "3 evaluations" in got["sub"] and "2 behaviours" in got["sub"]
    assert len(got["key"]) == 3 and "organism − the same-size base model" in got["key"][0]
    assert "blue = better" in got["key"][1] and "1.96 SE" in got["key"][2]
    assert got["details"] == 1 and got["open"] is True and got["old"] == 0
    assert "Unscored samples" in got["text"] and "Drill down" in got["text"]
    # the paper mapping lives inside Details: radar score formulas, expression threshold (PASS / FAIL only), eval count
    for needle in (
        "How this maps to the paper",
        "score = 10 × matched-item Δ (base = 0)",
        "score = 10 × (1 + Δ), capped at 10 (base = 10)",
        "Δ = organism − same-size base, in the score's 0–1 units",
        "Unlike the paper's radars",
        "FAIL",
    ):
        assert needle in got["text"], needle
    assert "PASS_INDIST" not in got["text"] and "paper scale" not in got["text"].lower()
    assert "0–10" not in got["text"] and "0-10" not in got["text"]
    assert "3 evaluations" in got["text"]


def test_executed_details_lists_each_results_dataset_as_a_hugging_face_link(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    data["meta"]["datasets"] = [
        {
            "repo": "Org/a_impulsive-qwen_2_5-7b-big_eval_results",
            "name": "a_impulsive-qwen_2_5-7b-big_eval_results",
            "label": "Impulsiveness, Qwen2.5 results",
            "revision": "abc",
            "url": "https://huggingface.co/datasets/Org/a_impulsive-qwen_2_5-7b-big_eval_results",
        },
        {
            "repo": "Org/evil",
            "name": "evil",
            "label": "x",
            "revision": "",
            "url": "javascript:alert(1)",
        },
    ]
    snippet = """(()=>[...document.querySelectorAll('#intro li.dataset')].map(l=>
      [l.textContent,l.querySelector('a').getAttribute('href')]))()"""
    got = _jsdom_run(render_html(data), snippet, tmp_path)
    assert got == [
        [
            "Impulsiveness, Qwen2.5 results — a_impulsive-qwen_2_5-7b-big_eval_results",
            "https://huggingface.co/datasets/Org/a_impulsive-qwen_2_5-7b-big_eval_results",
        ]
    ], "only https://huggingface.co/datasets/ links are rendered"


def test_executed_compare_switch_picks_one_dimension_and_spreads_the_other(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Behaviour mode (default): one base model, colour = behaviour. Model mode: one behaviour,
    every base model that carries it, colour = base model; the model is named in the legend only."""
    snippet = """(()=>{const ids=()=>treated().map(o=>o.id);
      const a={mode:state.mode,bm:new Set(treated().map(bmOf)).size,pers:new Set(treated().map(o=>o.persona)).size,
        heads:[...document.querySelectorAll('#main .persona-h')].map(h=>h.textContent)};
      setMode('model');
      const b={mode:state.mode,beh:state.behaviour,bm:new Set(treated().map(bmOf)).size,pers:new Set(treated().map(o=>o.persona)).size,
        heads:[...document.querySelectorAll('#main .persona-h')].map(h=>h.textContent),
        labels:treated().map(niceOrg),radio:[...document.querySelectorAll('#filters .fgroup label')].map(l=>l.textContent),
        colours:new Set(treated().map(orgCol)).size,
        errs:document.querySelectorAll('#main .empty').length};
      setMode('behaviour');
      return {a,b,short:[bmName('Qwen/Qwen2.5-14B-Instruct'),bmName('google/gemma-4-31B-it')],back:state.mode}})()"""
    got = _jsdom_run(many_page[1], snippet, tmp_path)
    assert got["a"]["bm"] == 1 and got["a"]["pers"] == 2
    assert all(h.startswith("Behaviour: ") for h in got["a"]["heads"])
    b = got["b"]
    assert b["pers"] == 1 and b["bm"] >= 2 and b["colours"] == b["bm"]
    assert all(h.startswith("Base model: ") for h in b["heads"])
    assert set(b["labels"]) <= {"SFT", "DPO"}, "points are named by the paper's method names only"
    assert b["radio"][:4] == ["Compare", "Behaviour", "Base model", "Implant method"], (
        "the behaviour radio replaces the base-model radio; base models and methods are chips"
    )
    assert got["short"] == ["Qwen2.5-14B", "Gemma-4-31B"] and got["back"] == "behaviour"


def test_executed_page_without_radar_data_falls_back_to_delta(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    stripped = {k: v for k, v in data.items() if k != "radar"}
    snippet = """(()=>{const s=document.querySelector('section[data-fig="radars"]');
      return {paper:PAPER_SCALE,base:s.querySelectorAll('svg circle.basering').length,
        pill:s.querySelector('h2 .pill').textContent}})()"""
    got = _jsdom_run(render_html(stripped), snippet, tmp_path)
    assert got["paper"] is False and got["base"] == 0 and got["pill"] == "Δ vs base"


def test_template_has_no_stale_z_score_radar_wording() -> None:
    src = _template_src()
    assert "radar core (z-score)" not in src
    assert "outside the dashed ring" not in src.lower().replace("<b>", "").replace("</b>", "")
    from analysis import dashboard

    assert "significance z-scores" not in (dashboard.__doc__ or "")
    assert "paper's 0-10 scale" in (dashboard.__doc__ or "")


def test_executed_diagnostics_run_wide_table_is_withheld_while_a_filter_hides_organisms(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """A run-wide Diagnostics table (cost by eval) sums every organism, so while the base-model
    filter hides some it is replaced by an explanation; per-organism rows are still filtered."""
    data, _ = many_page
    orgs = [o for o in data["organisms"] if not o["is_base"]]
    seven = next(o["id"] for o in orgs if o["base_model"].endswith("7B-Instruct"))
    fourteen = next(o["id"] for o in orgs if o["base_model"].endswith("14B-Instruct"))
    data["diagnostics"] = [
        {"key": "cost_by_eval", "caption": "Cost by eval", "note": "", "header": ["eval", "min"],
         "rows": [["mmlu", "9"]], "row_orgs": [""], "run_wide": True, "wide": False},
        {"key": "seeds", "caption": "Seeds", "note": "", "header": ["organism", "n"],
         "rows": [[seven, "1"], [fourteen, "2"]], "row_orgs": [seven, fourteen],
         "run_wide": False, "wide": False},
    ]  # fmt: skip
    snippet = """(()=>{const t=k=>document.querySelector('.dt[data-key="'+k+'"]');
        const rows=k=>[...t(k).querySelectorAll('tbody tr')].map(r=>r.textContent);
        return {bm:state.baseModel,cost:rows('cost_by_eval'),seeds:rows('seeds'),
          costNote:(t('cost_by_eval').querySelector('.empty')||{}).textContent||null}})()"""
    got = _jsdom_run(render_html(data), snippet, tmp_path)
    assert got["bm"].endswith("7B-Instruct")
    assert got["cost"] == [], "the all-organism cost table must not show under a base-model filter"
    assert got["costNote"] and "all organisms" in got["costNote"]
    assert len(got["seeds"]) == 1 and got["seeds"][0].startswith(seven)


# ---- the site copy: credit footer + site patches (site/scripts/check-links.mjs fails without them)
_REPO = Path(__file__).resolve().parent.parent


def test_site_render_carries_the_credit_footer_and_patches(fake_page: tuple[dict, str]) -> None:
    from analysis.dashboard import _SITE_TEMPLATE_PATCHES, site_dashboard_html

    data, html = fake_page
    site = site_dashboard_html(html, data, allow_d_radars=True)
    assert "nerfies.github.io" in site and "Back to project page" in site
    assert site.count('<footer id="site-footer"') == 1
    for old, new in _SITE_TEMPLATE_PATCHES:
        assert old not in site and new in site
    assert "nerfies.github.io" not in html, "the default render has no site footer"


def test_build_page_site_command_rerenders_with_data_and_transcripts_untouched(
    fake_page: tuple[dict, str],
) -> None:
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from analysis.dashboard_build import _DATA_RE, _TX_RE, render_site_from_page

    data, _ = fake_page
    data = {**data, "transcripts": {"cap": 1, "seed": 0, "orgs": {}, "excluded_evals": []}}
    page = render_html(data)
    site = render_site_from_page(page, allow_d_radars=True)
    assert "nerfies.github.io" in site and "Back to project page" in site
    for rx in (_DATA_RE, _TX_RE):
        a, b = rx.search(site), rx.search(page)
        assert a and b and a.group(2) == b.group(2)


def test_site_from_page_refreshes_a_page_rendered_before_display_stats(
    fake_page: tuple[dict, str],
) -> None:
    """A page whose blob lacks `derived`, `ci_lo`, `hw` ... (rendered before they existed) is
    still refreshed: the re-render adds them and the guard compares the blobs without them."""
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from analysis.dashboard_build import _DATA_RE, render_site_from_page
    from analysis.dashboard_stats import strip_display_stats

    _, page = fake_page
    m = _DATA_RE.search(page)
    assert m
    old_blob = json.dumps(strip_display_stats(json.loads(m.group(2))), separators=(",", ":"))
    assert old_blob != m.group(2) and '"derived"' not in old_blob
    old_page = page[: m.start(2)] + old_blob + page[m.end(2) :]
    site = render_site_from_page(old_page, allow_d_radars=True)
    assert '"derived"' in site


def test_site_render_keeps_build_provenance(fake_page: tuple[dict, str]) -> None:
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from analysis.dashboard_build import _DATA_RE, render_site_from_page

    data, _ = fake_page
    build = {
        "revisions": {"Org/x-big_eval_results": "1b46943b62c3612436045897c3155d37b39c010e"},
        "code": {"analysis": "/home/someone/checkout abc123"},
        "tree": {"qimp-7b": {"n_files": 1}},
    }
    page = render_html(
        {**data, "transcripts": {"cap": 1, "seed": 0, "orgs": {}, "excluded_evals": []}}
    )
    site = render_site_from_page(page, build, allow_d_radars=True)
    emb = json.loads(_DATA_RE.search(site).group(2))
    assert emb["_build"]["revisions"] == build["revisions"]
    assert emb["_build"]["tree"] == build["tree"]
    assert "code" not in emb["_build"] and "/home/someone" not in site
    assert "x-big_eval_results@1b46943b62" in site
    # a site page built straight from data with `_build` keeps it too
    assert "_build" in _DATA_RE.search(site_html_with_build(data, build)).group(2)


def test_site_render_sanitises_a_pages_own_build(fake_page: tuple[dict, str]) -> None:
    """`site --from-page` without `--build-from` must still strip an internal page's private
    provenance (Codex, PR #455: the checkout path and -restricted repos leaked that way)."""
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from analysis.dashboard_build import _DATA_RE, render_site_from_page

    data, _ = fake_page
    build = {
        "revisions": {"Org/a_results": "1" * 40, "Org/a_results-restricted": "2" * 40},
        "code": {"analysis": "/home/someone/checkout abc123"},
    }
    page = render_html(
        {
            **data,
            "_build": build,
            "transcripts": {"cap": 1, "seed": 0, "orgs": {}, "excluded_evals": []},
        }
    )
    site = render_site_from_page(page, allow_d_radars=True)
    emb = json.loads(_DATA_RE.search(site).group(2))
    assert emb["_build"]["revisions"] == {"Org/a_results": "1" * 40}
    assert "code" not in emb["_build"]
    assert "/home/someone" not in site and "-restricted" not in site


def site_html_with_build(data: dict, build: dict) -> str:
    from analysis.dashboard import site_dashboard_html

    d = {**data, "_build": build}
    return site_dashboard_html(render_html(d), d, allow_d_radars=True)


def test_committed_site_dashboard_keeps_build_provenance() -> None:
    page = (_REPO / "site" / "public" / "dashboard" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert m
    revs = json.loads(m.group(1))["_build"]["revisions"]
    assert {r[:10] for r in revs.values()} >= {"14fd7741ea", "93997c23e5", "7500b665d4"}
    # the private companion repos never reach the public page, not even by name
    assert "-restricted" not in page


def test_publishable_build_drops_local_paths_and_restricted_repos() -> None:
    from analysis.dashboard_build import publishable_build

    b = {
        "code": "/workspace/x/mo abc",
        "revisions": {"Org/a_results": "1" * 40, "Org/a_results-restricted": "2" * 40},
        "tree": {"n": 1},
    }
    out = publishable_build(b)
    assert "code" not in out and out["tree"] == {"n": 1}
    assert out["revisions"] == {"Org/a_results": "1" * 40}


def test_committed_site_dashboard_has_the_credit_footer() -> None:
    page = (_REPO / "site" / "public" / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "nerfies.github.io" in page and "Back to project page" in page, (
        "re-render it with `scripts/build_dashboard_page.py site --from-page <page>`, "
        "never copy the plain `html` output over it"
    )


def test_build_script_site_refuses_a_page_without_radar_unless_allowed(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """`site --from-page` on a page whose data lacks `radar` exits non-zero, `--allow-d-radars` passes."""
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from analysis.dashboard_build import main

    data, _ = fake_page
    data = {k: v for k, v in data.items() if k != "radar"}
    page = tmp_path / "p.html"
    page.write_text(render_html(data), encoding="utf-8")
    out = tmp_path / "site.html"
    with pytest.raises(SystemExit, match="allow-d-radars"):
        main(["site", "--from-page", str(page), "--site-out", str(out)])
    assert not out.exists()
    main(["site", "--from-page", str(page), "--site-out", str(out), "--allow-d-radars"])
    assert "nerfies.github.io" in out.read_text(encoding="utf-8")


# ---- misalignment views (D3 heatmap cell text, D2 dumbbells), fallback note, dark theme ------------
_MIS_EVALS = {
    "misaligned_overrefusal": "overrefusal",
    "misaligned_sandbagging:dod": "sandbagging (incentive - filler)",
}


def _with_misalignment(data: dict) -> dict:
    """The fake data plus a FAKE misalignment property: two evals, raw rates 0.43 (organism) / 0.11 (base)."""
    import copy

    d = copy.deepcopy(data)
    d["axes"].append(
        {
            "name": "misalignment",
            "short": "misalignment",
            "question": "",
            "runs_backwards": True,
            "radar": False,
            "group": "health",
        }
    )
    d["bucket_evals"]["misalignment"] = list(_MIS_EVALS)
    for e, disp in _MIS_EVALS.items():
        d["evals"][e] = {
            "axis": "misalignment",
            "display": disp,
            "direction": "lower_is_better",
            "in_composite": True,
            "score_type": "binary",
            "reference": "base",
        }
    for o in d["organisms"]:
        d["axis_cells"].setdefault(o["id"], {})["misalignment"] = {
            "delta": -0.3,
            "se": 0.1,
            "n": 1,
            "n_total": 1,
        }
        for e in _MIS_EVALS:
            d["cells"][o["id"]][e] = {
                "mean": 0.11 if o["is_base"] else 0.43,
                "sem": 0.02,
                "sd": 0.3,
                "n": 100,
                "n_seeds": 3,
                "discard": 0.0,
                "unparsed": 0.0,
                "delta": 0.0 if o["is_base"] else -0.32,
                "se": 0.03,
                "votes": True,
            }
    return d


def test_executed_misalignment_cells_show_organism_and_base_rates(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    html = render_html(_with_misalignment(fake_page[0]))
    got = _jsdom_run(
        html,
        """(()=>{selectBucket("misalignment");
          const sec=document.querySelector('section[data-fig="heatmap"]');
          const txt=[...sec.querySelectorAll('text.celltxt')].map(t=>t.textContent);
          const cells=[...sec.querySelectorAll('rect[cursor="pointer"]')];
          const ev=cells[cells.length-2];
          ev.dispatchEvent(new window.MouseEvent('mousemove',{clientX:5,clientY:5}));
          return {txt,tip:document.getElementById('tt').innerHTML,
                  legend:[...sec.querySelectorAll('.selbar')].map(x=>x.textContent).join('|')}})()""",
        tmp_path,
    )
    assert got["txt"] and set(got["txt"]) == {"0.43 / 0.11"}, got["txt"]
    assert "organism / same-size base model" in got["legend"]
    assert "health check" not in got["tip"] and "not scored" not in got["tip"]
    assert "higher misalignment rate is the misaligned direction" in got["tip"]
    assert "0.430" in got["tip"] and "0.110" in got["tip"]
    # misalignment is shown as organism - base (+ = more misbehaviour), not sign-flipped
    assert "Δ +0.320" in got["tip"] and "more misbehaviour" in got["tip"]


def test_executed_misalignment_dumbbell_view(fake_page: tuple[dict, str], tmp_path: Path) -> None:
    data = _with_misalignment(fake_page[0])
    got = _jsdom_run(
        render_html(data),
        """(()=>{const sec=document.querySelector('section[data-fig="misalign"]');
          const panels=[...sec.querySelectorAll('.rpanel')];
          const row=sec.querySelector('.dbrow');
          row.dispatchEvent(new window.MouseEvent('mousemove',{clientX:5,clientY:5}));
          return {panels:panels.map(p=>p.querySelector('.rt').textContent),
                  rows:panels.map(p=>p.querySelectorAll('.dbrow').length),
                  bases:panels.map(p=>p.querySelectorAll('.dbbase').length),
                  tip:document.getElementById('tt').innerHTML,
                  toc:[...document.querySelectorAll('#toc a')].map(a=>a.textContent)}})()""",
        tmp_path,
    )
    assert got["panels"] == ["Overrefusal", "Sandbagging (incentive - filler)"]
    n_treated = sum(1 for o in data["organisms"] if not o["is_base"])
    assert got["rows"] == [n_treated, n_treated] and got["bases"] == got["rows"]
    assert "same-size base 0.110" in got["tip"] and "organism 0.430" in got["tip"]
    assert "95% CI" in got["tip"] and "not scored" not in got["tip"]
    assert any(t.startswith("Misalignment") for t in got["toc"])


def _first_treated_and_axis(data: dict) -> tuple[str, str]:
    org = next(o["id"] for o in data["organisms"] if not o["is_base"])
    return org, data["axes"][0]["name"]


def test_executed_ci_is_1_96_times_the_exported_se_and_no_se_means_no_ci(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The page's CI half-width is exactly 1.96 x `se` (the server's Cell.ci95 rule), whatever the
    seed count (no t on seeds: the cell SE already absorbs them); `se: null` gives no CI and no dot."""
    import copy

    data = copy.deepcopy(fake_page[0])
    org, ax = _first_treated_and_axis(data)
    for o in data["organisms"]:
        o["n_seeds"] = 2  # a 2-seed organism once got x12.706
    data["axis_cells"][org][ax] = {"delta": 0.2, "se": 0.05, "n": 1, "n_total": 1}
    other = next(a["name"] for a in data["axes"] if a["name"] != ax)
    data["axis_cells"][org][other] = {"delta": 0.2, "se": None, "n": 1, "n_total": 1}
    got = _jsdom_run(
        render_html(data),
        f"""(()=>{{const a=axisRead({org!r},{ax!r}),b=axisRead({org!r},{other!r});
          return {{lo:a.lo,hi:a.hi,sig:a.sig,noLo:b.lo,noSig:b.sig,
                   tip:readTip(b,0.5,0.3)}}}})()""",
        tmp_path,
    )
    assert got["lo"] == pytest.approx(0.2 - 1.96 * 0.05)
    assert got["hi"] == pytest.approx(0.2 + 1.96 * 0.05)
    assert got["sig"] is True
    assert got["noLo"] is None and got["noSig"] is False
    assert "no 95% CI" in got["tip"]


def test_executed_share_kept_ci_is_clipped_to_0_1_in_the_tooltip(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    import copy

    data = copy.deepcopy(fake_page[0])
    org, ax = _first_treated_and_axis(data)
    next(a for a in data["axes"] if a["name"] == ax)["self_referenced"] = True
    data["axis_cells"][org][ax] = {"delta": 0.95, "se": 0.1, "n": 1, "n_total": 1}
    got = _jsdom_run(
        render_html(data),
        f"(()=>{{const r=axisRead({org!r},{ax!r});return {{hi:r.hi,lo:r.lo,tip:readTip(r,0.95,null)}}}})()",
        tmp_path,
    )
    assert got["hi"] == 1.0 and got["lo"] == pytest.approx(0.95 - 0.196)
    assert "1.000]" in got["tip"] and "share kept" in got["tip"]


def test_executed_delta_fallback_radars_skip_a_share_axis(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """`axisEff` (the Δ radars' value) is null for a share-kept axis: robustness is not a Δ spoke."""
    import copy

    data = copy.deepcopy(fake_page[0])
    org, ax = _first_treated_and_axis(data)
    other = next(a["name"] for a in data["axes"] if a["name"] != ax)
    next(a for a in data["axes"] if a["name"] == ax)["self_referenced"] = True
    got = _jsdom_run(
        render_html(data),
        f"(()=>({{share:axisEff({org!r},{ax!r}),delta:axisEff({org!r},{other!r})}}))()",
        tmp_path,
    )
    assert got["share"] is None and got["delta"] is not None


def test_executed_misalignment_axis_tooltip_shows_more_misbehaviour_as_positive(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The blob keeps misalignment oriented (delta -0.3 = worse); the page DISPLAYS organism - base,
    so the axis-level tooltip reads Δ +0.300 and says + = more misbehaviour."""
    data = _with_misalignment(fake_page[0])
    org = next(o["id"] for o in data["organisms"] if not o["is_base"])
    got = _jsdom_run(
        render_html(data),
        f"(()=>{{const r=axisRead({org!r},'misalignment');return {{d:r.d,o:r.o,tip:readTip(r,0.43,0.11)}}}})()",
        tmp_path,
    )
    assert got["d"] == pytest.approx(0.3) and got["o"] == pytest.approx(-0.3)
    assert "Δ +0.300" in got["tip"] and "more misbehaviour" in got["tip"]


def test_executed_no_misalignment_evals_no_dumbbell(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        fake_page[1],
        "document.querySelectorAll('section[data-fig=\"misalign\"]').length",
        tmp_path,
    )
    assert got == 0


def test_executed_heatmap_column_labels_fit_inside_the_svg(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The rotated column labels (-40deg) must not overhang the svg's top or right edge: a fixed 100px
    top margin clipped "misalignment (health check)". Measured in a browser: 5.6px per character at
    11px/600, so 5.6 is the (non-generous) per-character width used here."""
    got = _jsdom_run(
        render_html(_with_misalignment(fake_page[0])),
        """(()=>{const sec=document.querySelector('section[data-fig="heatmap"]');
          const svg=sec.querySelector('svg');const W=+svg.getAttribute('width');
          return {W,labels:[...svg.querySelectorAll('text')].filter(t=>(t.getAttribute('transform')||'').startsWith('rotate(-40'))
            .map(t=>({text:t.textContent,x:+t.getAttribute('x'),y:+t.getAttribute('y')}))}})()""",
        tmp_path,
    )
    assert any(label["text"].startswith("misalignment") for label in got["labels"]), got["labels"]
    sin, cos = math.sin(math.radians(40)), math.cos(math.radians(40))
    for label in got["labels"]:
        width = len(label["text"]) * 5.6
        assert label["y"] >= width * sin + 8, f"{label['text']!r} is clipped at the top"
        assert label["x"] + width * cos <= got["W"], f"{label['text']!r} is clipped at the right"


def test_executed_heatmap_with_short_labels_keeps_the_old_size(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Labels that already fit leave the svg exactly as before the fix: top margin 100px and width
    mL + columns * cell + colour-bar gutter + 8 (a final review found a 1px growth from padding alone)."""
    got = _jsdom_run(
        fake_page[1],
        """(()=>{const svg=document.querySelector('section[data-fig="heatmap"] svg');
          const n=[...svg.querySelectorAll('text')].filter(t=>(t.getAttribute('transform')||'').startsWith('rotate(-40')).length;
          return {W:+svg.getAttribute('width'),H:+svg.getAttribute('height'),n,
            rot:[...svg.querySelectorAll('text')].filter(t=>(t.getAttribute('transform')||'').startsWith('rotate(-40')).map(t=>+t.getAttribute('y'))}})()""",
        tmp_path,
    )
    cw = max(46, min(58, 660 / got["n"]))
    assert got["W"] == 150 + got["n"] * cw + 84 + 8
    assert set(got["rot"]) == {
        84 - 8
    }  # labels sit on the 84px top margin (13px labels; was 100: a blank band above the matrix)


def test_template_pill_does_not_inherit_the_serif_headings_face() -> None:
    """The `.pill` sits inside serif h2s; inheriting Fraunces 600 at 11px squeezed its word spaces
    to ~2.3px so a two-word pill read as one run-together word. It sets its own sans face and no tracking."""
    css = _template_src()
    rule = re.search(r"\n\.pill\{([^}]*)\}", css)
    assert rule, "no .pill rule"
    assert "font-family:var(--sans)" in rule.group(1)
    assert "letter-spacing:0" in rule.group(1)


def test_template_has_a_dark_theme_with_the_same_tokens_as_light() -> None:
    src = _template_src()
    css = src[: src.index("</style>")]
    assert "@media (prefers-color-scheme:dark)" in css
    assert ':root[data-theme="dark"]{' in css
    assert ':root:not([data-theme="light"])' in css, "an explicit light choice beats the OS dark"
    light = css[css.index(":root{") : css.index("@media")]
    dark_block = css[css.index(':root[data-theme="dark"]{') :].split("}\n", 1)[0]

    def tok(block: str) -> set[str]:
        return set(re.findall(r"(--[a-z0-9-]+):", block))

    # every colour token the light theme defines is redefined for dark (fonts aside)
    missing = tok(light) - tok(dark_block) - {"--serif", "--sans", "--mono"}
    assert not missing, missing
    # no chart colour is a hard-coded light grey or white any more
    js = src[src.index("<script>") :]
    for hexv in ("#d6d6cf", "#e2e1da", "#efeee9", "#a9a9a2", "#6f6f69", '"#fff"'):
        assert hexv not in js, hexv


def test_executed_dark_theme_scale_and_ink_contrast(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        fake_page[1],
        """(()=>{const out={};
          const lum=c=>{const m=/(\\d+),(\\d+),(\\d+)/.exec(c).slice(1).map(Number).map(v=>{v/=255;
            return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)});return 0.2126*m[0]+0.7152*m[1]+0.0722*m[2];};
          const rgb=h=>{const n=parseInt(h.slice(1),16);return `${n>>16},${(n>>8)&255},${n&255}`;};
          const worst=()=>{let w=99;for(let t=-1;t<=1.001;t+=0.05){const f=diverge(t),ink=inkOn(f);
            const a=lum(f)+0.05,b=lum(rgb(ink))+0.05;w=Math.min(w,Math.max(a,b)/Math.min(a,b));}return w;};
          out.light=worst();out.lightMid=diverge(0);
          document.documentElement.setAttribute('data-theme','dark');
          out.isDark=isDark();out.dark=worst();out.darkMid=diverge(0);
          out.lift=methodColor('dpo_behaviour');
          document.documentElement.setAttribute('data-theme','light');
          out.lightAgain=isDark();out.dpo=methodColor('dpo_behaviour');
          return out})()""",
        tmp_path,
    )
    assert got["isDark"] is True and got["lightAgain"] is False
    assert got["light"] >= 4.5 and got["dark"] >= 4.5, "heatmap ink stays legible on every cell"
    assert got["lightMid"] != got["darkMid"], "the dark scale does not start from a pale neutral"
    assert got["dpo"] == "#135DF6" and got["lift"] != got["dpo"]


def test_every_svg_text_names_its_fill(many_page: tuple[dict, str], tmp_path: Path) -> None:
    """An SVG <text> with no fill is black: invisible on the dark page."""
    data, _ = many_page
    got = _jsdom_run(
        render_html(_with_misalignment(data)),
        """(()=>{selectBucket("misalignment");
          return [...document.querySelectorAll('svg text')].filter(t=>!t.getAttribute('fill')).length})()""",
        tmp_path,
    )
    assert got == 0


# ---- round 2: base-model and implant-method chips, paper tiles, zoom, enlarge -----------------------

_CHIPS = """(()=>{const grp=t=>[...document.querySelectorAll('#filters .fgroup')].find(g=>g.querySelector('label').textContent.startsWith(t));
    const chip=(t,x)=>[...grp(t).querySelectorAll('.chip,.mini')].find(c=>c.textContent===x);
    const bms=()=>[...new Set(treated().map(bmOf))].map(bmName).sort();
    const out={};
    setMode('model');out.carry=[...grp('Base model').querySelectorAll('.chip')].map(c=>c.textContent);
    out.all=bms();
    chip('Base model','Qwen2.5-14B').click();out.off14=bms();out.chip14on=chip('Base model','Qwen2.5-14B').classList.contains('on');
    chip('Base model','None').click();out.none=bms();out.noneMsg=document.querySelectorAll('#main .empty').length>0;
    chip('Base model','All').click();out.back=bms();
    out.methodChips=[...grp('Implant method').querySelectorAll('.chip')].map(c=>c.textContent);
    chip('Implant method','SFT').click();out.noSft=[...new Set(treated().map(o=>o.method))].sort();
    out.legend=[...document.querySelectorAll('section[data-fig="xy"] svg text')].map(t=>t.textContent).filter(t=>/^(SFT|DPO)$/.test(t)).length;
    chip('Implant method','All').click();
    chip('Implant method','None').click();out.noMethods=treated().length;
    chip('Implant method','All').click();
    setMode('behaviour');
    out.beh=[...grp('Implant method').querySelectorAll('.chip')].map(c=>c.textContent);
    chip('Implant method','DPO').click();out.behNoDpo=[...new Set(treated().map(o=>o.method))].sort();
    return out})()"""


def test_executed_base_model_and_method_chips_filter_every_figure(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _CHIPS, tmp_path)
    assert got["carry"] == ["Qwen2.5-7B", "Qwen2.5-14B"], (
        "chips: the base models that carry the behaviour"
    )
    assert got["all"] == ["Qwen2.5-14B", "Qwen2.5-7B"]
    assert got["off14"] == ["Qwen2.5-7B"] and got["chip14on"] is False
    assert got["none"] == [] and got["noneMsg"] is True
    assert got["back"] == got["all"]
    assert got["methodChips"] == ["SFT", "DPO"], "short paper names, lightest first"
    assert got["noSft"] == ["dpo_behaviour"] and got["legend"] >= 1
    assert got["noMethods"] == 0
    assert got["beh"] == ["SFT", "DPO"] and got["behNoDpo"] == ["sft_behaviour"]


_TILES = """(()=>{const tiles=()=>Object.fromEntries([...document.querySelectorAll('.tile')].map(t=>
      [t.dataset.tile,[t.querySelector('.tv').textContent,t.querySelector('.tn').textContent,t.querySelector('.td').textContent]]));
    const out={};
    const ids=treated().map(o=>o.id);
    // controlled numbers: o0 beats o1 on generality and robustness, but o0 is below the expression threshold
    ids.forEach((id,i)=>{DATA.axis_cells[id]=DATA.axis_cells[id]||{};
      ['generality','robustness','coherence'].forEach(a=>{DATA.axis_cells[id][a]={delta:0.5-0.1*i,se:0.01};});
      DATA.axis_cells[id].coherence.delta=-0.05*(i+1);
      DATA.expression.orgs[id]={rate:0.5,lo:0.4,hi:0.6,verdict:i===0?'FAIL':'PASS'};});
    render();out.excluded=tiles();out.excludedId=ids[0];out.note=document.querySelector('.tilenote').textContent;
    DATA.expression.orgs[ids[0]].verdict='PASS';render();out.included=tiles();
    DATA.expression.orgs[ids[0]].verdict=null;render();out.noVerdict=tiles();
    DATA.expression.orgs[ids[0]].verdict='PASS_INDIST';render();out.indist=tiles();
    DATA.expression.orgs[ids[0]].verdict='PASS';
    state.methods.delete(treated()[0].method);render();out.afterChip=tiles();
    out.treatedAfter=treated().map(o=>o.id);
    DATA.expression.orgs={};render();out.noThr=document.querySelector('.tilenote').textContent;
    return out})()"""


def test_executed_tiles_follow_the_selection_and_skip_organisms_below_the_threshold(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _TILES, tmp_path)
    ex, inc = got["excluded"], got["included"]
    # the FAIL organism has the best generality/robustness but is not eligible for those two tiles ...
    assert ex["generality"][0] == "+0.40" and inc["generality"][0] == "+0.50"
    assert ex["robustness"][0] == "0.40" and inc["robustness"][0] == "0.50"
    # ... while the coherence tile ranks everyone: the largest drop is the last organism's
    assert (
        ex["coherence"][0] == inc["coherence"][0]
        and float(ex["coherence"][0].replace("\u2212", "-")) < 0
    )
    assert "below the expression threshold" in got["note"]
    # only FAIL is below the threshold: a missing verdict or PASS_INDIST stays eligible
    assert (
        got["noVerdict"]["generality"][0] == "+0.50" and got["indist"]["generality"][0] == "+0.50"
    )
    # tile detail: short method name, then model and behaviour
    assert ex["generality"][1] in ("SFT", "DPO") and "Qwen2.5-" in ex["generality"][2]
    # a method chip changes the pool, so the tiles recompute
    assert got["afterChip"]["generality"][0] == "+0.40", (
        "dropping a method chip recomputes the tiles"
    )
    assert "no expression-threshold verdicts" in got["noThr"]


_ZOOM = r"""(()=>{const sec=document.querySelector('section[data-fig="xy"]');
    const svg=sec.querySelector('svg.zsvg');
    const ticks=()=>[...svg.querySelectorAll('text')].map(t=>t.textContent).filter(t=>/^[-+]\d/.test(t)).join(',');
    const btn=t=>[...sec.querySelectorAll('.zbtn')].find(b=>b.title===t);
    const out={start:ticks()};
    btn('Zoom in').click();btn('Zoom in').click();out.zoomed=ticks();
    out.fontsSame=[...svg.querySelectorAll('text')].filter(t=>t.getAttribute('font-size')==='10.5').every(t=>true);
    svg.dispatchEvent(new (svg.ownerDocument.defaultView.WheelEvent)('wheel',{deltaY:-300,ctrlKey:true,clientX:500,clientY:300,bubbles:true,cancelable:true}));
    out.wheeled=ticks();
    btn('Reset zoom').click();out.reset=ticks();
    const f=document.querySelector('section[data-fig="forest"]');
    const fz=f.querySelector('svg.zsvg'),ft=()=>[...fz.querySelectorAll('text')].map(t=>t.textContent).join(',');
    out.f0=ft();[...f.querySelectorAll('.zbtn')].find(b=>b.title==='Zoom in').click();out.f1=ft();
    [...f.querySelectorAll('.zbtn')].find(b=>b.dataset.zoomReset).click();out.f2=ft();
    out.hasReset=!!sec.querySelector('[data-zoom-reset]');
    return out})()"""


def test_executed_scatter_and_forest_zoom_in_and_reset(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _ZOOM, tmp_path)
    assert got["zoomed"] != got["start"], "zooming re-ranges the axes"
    assert got["reset"] == got["start"] and got["hasReset"]
    assert got["f1"] != got["f0"] and got["f2"] == got["f0"]


_XYPTS = r"""(()=>{const sec=document.querySelector('section[data-fig="xy"]');
    const svg=()=>sec.querySelector('svg.zsvg');
    const pts=()=>[...svg().querySelectorAll('g[data-org]')];
    const pos=()=>pts().map(g=>[g.dataset.org,g.dataset.cx,g.dataset.cy,g.dataset.r].join('|'));
    const tip=()=>pts().map(g=>{g.dispatchEvent(new MouseEvent('mousemove',{bubbles:true}));return document.getElementById('tt').innerHTML;});
    const btn=t=>[...sec.querySelectorAll('.zbtn')].find(b=>b.title===t);
    const jit=sec.querySelector('#xy-jitter');
    const out={jitterExists:!!jit,jitterOn:jit&&jit.checked,n:pts().length,
      labelled:pts().filter(g=>g.querySelector('text.plab')&&g.querySelector('text.plab').textContent.length>0).length,
      inPlot:pts().every(g=>{const t=g.querySelector('text.plab');return +t.getAttribute('x')>0&&+t.getAttribute('y')>0&&+t.getAttribute('x')<1000&&+t.getAttribute('y')<640;}),
      help:sec.querySelector('.howto').textContent};
    out.posOn=pos();out.tipOn=tip();
    out.posOn2=(()=>{rebuildFigs(['xy']);return pos();})();
    jit.checked=false;jit.dispatchEvent(new Event('change',{bubbles:true}));
    out.posOff=pos();out.tipOff=tip();
    jit.checked=true;jit.dispatchEvent(new Event('change',{bubbles:true}));
    out.posBack=pos();
    out.r0=+pts()[0].dataset.r;
    const g0=pts()[0];svg().dispatchEvent(new (svg().ownerDocument.defaultView.WheelEvent)('wheel',{deltaY:-300,ctrlKey:true,clientX:+g0.dataset.cx,clientY:+g0.dataset.cy,bubbles:true,cancelable:true}));
    out.r1=+sec.querySelector('svg.zsvg g[data-org]').dataset.r;
    out.visZoomed=pts().length;out.labZoomed=pts().filter(g=>g.querySelector('text.plab')).length;
    btn('Reset zoom').click();
    out.posReset=pos();
    return out})()"""


def test_executed_scatter_labels_every_point_jitter_and_zoom_sizes(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _XYPTS, tmp_path)
    assert got["n"] >= 2
    assert got["labelled"] == got["n"], "every visible point has a label element"
    assert got["inPlot"], "labels stay inside the plot area"
    assert got["jitterExists"] and got["jitterOn"], (
        "the jitter checkbox exists and is on by default"
    )
    assert "visual only" in got["help"]
    assert got["posOn"] != got["posOff"], "toggling jitter moves the points"
    assert got["posBack"] == got["posOn"]
    assert got["posOn"] == got["posOn2"], (
        "jitter is deterministic: two renders give identical positions"
    )
    assert got["tipOn"] == got["tipOff"], "the tooltip shows the true values either way"
    assert got["r0"] <= got["r1"] <= got["r0"] * 1.15 + 0.01, "markers grow by at most 1.15x"
    assert got["labZoomed"] == got["visZoomed"], "zoomed-in points are all labelled too"
    assert got["posReset"] == got["posOn"], "Reset restores the original positions and sizes"


_TILE_TOGGLE = """(()=>{const out={};
    const ids=treated().filter(o=>!isControl(o)).map(o=>o.id);
    ids.forEach((id,i)=>{DATA.axis_cells[id]=DATA.axis_cells[id]||{};
      ['generality','robustness','coherence'].forEach(a=>{DATA.axis_cells[id][a]={delta:0.5-0.1*i,se:0.01};});
      DATA.expression.orgs[id]={rate:0.5,lo:0.4,hi:0.6,verdict:i===0?'FAIL':'PASS'};});
    const snap=()=>({gen:document.querySelector('[data-tile=generality] .tv').textContent,
      lines:[...document.querySelectorAll('.tile .tx')].map(x=>x.textContent),
      coh:document.querySelectorAll('[data-tile=coherence] .tx').length,
      btn:(document.getElementById('tile-incl')||{}).textContent,
      nbtn:document.querySelectorAll('#tiles ~ .tilenote button, .tilenote button').length});
    render();out.off=snap();
    document.getElementById('tile-incl').click();out.on=snap();
    document.getElementById('tile-incl').click();out.off2=snap();
    return out})()"""


def test_executed_tiles_say_what_they_exclude_and_can_include_it(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _TILE_TOGGLE, tmp_path)
    off, on = got["off"], got["on"]
    assert off["lines"] == ["excl. 1 that failed the expression test"] * 2
    assert off["btn"] == "include them" and off["nbtn"] == 1, (
        "one small text toggle, no other button"
    )
    assert on["lines"] == ["incl. organisms that failed the expression test"] * 2
    assert on["gen"] == "+0.50" and off["gen"] == "+0.40"
    assert on["coh"] == 0 and off["coh"] == 0, "the coherence tile is unaffected"
    assert got["off2"] == off, "default is exclude, and the toggle round-trips"


_OVL_TT = """(()=>{const svg=document.querySelector('section[data-fig="radars"] .rpanel > svg');
    svg.dispatchEvent(new MouseEvent('click',{bubbles:true}));
    const clone=document.querySelector('.ovl .ovlbody svg'),tt=document.getElementById('tt');
    tt.style.opacity=0;tt.innerHTML='';
    let hit=null;
    for(const n of clone.querySelectorAll('path,circle,rect,g')){
      n.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,clientX:50,clientY:60}));
      if(tt.style.opacity==='1'&&tt.textContent.length>0){hit=tt.textContent;break;}}
    return {hit,left:tt.style.left}})()"""


def test_executed_enlarged_radar_shows_tooltips(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _OVL_TT, tmp_path)
    assert got["hit"], "hovering a radar mark inside the overlay shows its tooltip"
    assert got["left"] == "64px", "the tooltip follows the pointer"


_ENLARGE = """(()=>{const out={};
    const svg=document.querySelector('section[data-fig="radars"] .rpanel > svg');
    svg.dispatchEvent(new MouseEvent('click',{bubbles:true}));
    out.open=!!document.querySelector('.ovl .ovlbody svg');out.closeBtn=!!document.querySelector('.ovl .ovlclose');
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));out.escClosed=!document.querySelector('.ovl');
    svg.dispatchEvent(new MouseEvent('click',{bubbles:true}));
    document.querySelector('.ovlclose').click();out.btnClosed=!document.querySelector('.ovl');
    const spoke=[...svg.querySelectorAll('text[data-noenlarge]')][0];
    if(spoke)spoke.dispatchEvent(new MouseEvent('click',{bubbles:true}));
    out.spokeNoOverlay=!document.querySelector('.ovl');
    const h=document.querySelector('section[data-fig="heatmap"] .heatscroll > svg');
    h.dispatchEvent(new MouseEvent('click',{bubbles:true}));out.heat=!!document.querySelector('.ovl');
    return out})()"""


def test_executed_radars_and_heatmap_enlarge_in_an_overlay(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _ENLARGE, tmp_path)
    assert got["open"] and got["closeBtn"]
    assert got["escClosed"] and got["btnClosed"]
    assert got["spokeNoOverlay"], "a spoke label keeps its own click (drill-down)"
    assert got["heat"]


# ---- round 2: only the organisms in the specs file, and the dataset links -----------------------------


def test_specs_filter_drops_an_organism_that_is_not_in_the_specs_from_every_block(
    tmp_path: Path,
) -> None:
    from analysis.dashboard import keep_spec_organisms

    data = _build(tmp_path / "tree", _MANY)
    ids = {o["id"] for o in data["organisms"]}
    extra = "impulsive/dpo_behaviour@1:7B"
    assert extra in ids
    dropped, unmatched = keep_spec_organisms(data, (ids - {extra}) | {"impulsive/oct_demos@1:7B"})
    assert dropped == [extra] and unmatched == ["impulsive/oct_demos@1:7B"]
    assert extra not in json.dumps(data), "no block of the data still names the dropped organism"
    assert (
        "dpo_behaviour" not in data["meta"]["methods"]
        and data["meta"]["n_organisms"] == len(ids) - 1
    )
    # a results tree that shares nothing with the specs is left alone (a blank page helps nobody)
    again = _build(tmp_path / "tree2", _ONE)
    before = json.dumps(again, sort_keys=True)
    assert keep_spec_organisms(again, {"x/y@1:1B"})[0] == []
    assert json.dumps(again, sort_keys=True) == before


def test_spec_organism_ids_match_the_ids_the_results_rows_carry() -> None:
    from analysis.dashboard import DEFAULT_SPECS, spec_organism_ids

    ids = spec_organism_ids(DEFAULT_SPECS)
    # variant tags are the rows' own (aggregate._variant_of): a recipe, a k, a polarity, "framed"
    for want in (
        "impulsive/prompting[negative]@1:7B",
        "impulsive/few_shot_icl[k=15 framed]@1:14B",
        "impulsive/base@1:32B",
        "impulsive/steering[all]@1:31B",
    ):
        assert want in ids, want


def test_dataset_links_name_each_public_repo_in_words_and_skip_the_restricted_ones() -> None:
    from analysis.dashboard import attach_dataset_links, dataset_links

    org = "science-of-model-organisms/"
    names = (
        "impulsive-qwen_2_5-7b-14b-32b-big_eval_results",
        "impulsive-gemma_4-31b-big_eval_results",
        "loyalty-qwen_2_5-7b-14b-32b-big_eval_results",
        "odd_name",
    )
    revs = {org + n: "0123456789abcdef" for n in names}
    revs[org + names[0] + "-restricted"] = "f" * 40
    data = {
        "meta": {
            "base_models": [
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "google/gemma-4-31B-it",
            ]
        },
        "_build": {"revisions": revs},
    }
    links = dataset_links(data)
    # Ordered by repo id (`dataset_links` sorts): the handle-free public names sort
    # impulsive-gemma < impulsive-qwen < loyalty < odd_name.
    assert [(d["label"], d["url"]) for d in links] == [
        ("Impulsiveness, Gemma-4-31B results", f"https://huggingface.co/datasets/{org}{names[1]}"),
        ("Impulsiveness, Qwen2.5 results", f"https://huggingface.co/datasets/{org}{names[0]}"),
        ("Loyalty, Qwen2.5 results", f"https://huggingface.co/datasets/{org}{names[2]}"),
        ("odd_name", f"https://huggingface.co/datasets/{org}odd_name"),
    ]
    attach_dataset_links(data)
    assert len(data["meta"]["datasets"]) == 4


# ---- the negative-prompt control is listed last; the drill-down drawers; the Profiles view switch -----------------


def _with_control(data: dict) -> dict:
    """A copy of `data` with a negative-prompt control arm (method `prompting`, variant `[negative]`) on the
    impulsive 7B model, cloned from its SFT organism. FAKE numbers. `prompting` sorts FIRST by method, so any list that
    does not use the shared ordering helper would show the control first."""
    out = json.loads(json.dumps(data))
    src = next(
        o
        for o in out["organisms"]
        if o["id"] == "impulsive/sft_behaviour@1:7B" and not o["is_base"]
    )
    cid = "impulsive/prompting[negative]@1:7B"
    out["organisms"].append(
        {
            **src,
            "id": cid,
            "label": cid,
            "method": "prompting",
            "short": "prompting[negative]@1:7B",
        }
    )
    for block in ("cells", "axis_cells"):
        out[block][cid] = json.loads(json.dumps(out[block][src["id"]]))
    out["meta"]["methods"] = sorted({*out["meta"]["methods"], "prompting"})
    out["meta"]["n_treated"] += 1
    if out.get("expression"):
        out["expression"]["orgs"][cid] = dict(out["expression"]["orgs"][src["id"]])
    if out.get("radar"):
        out["radar"]["orgs"][cid] = dict(out["radar"]["orgs"][src["id"]])
        out["radar"].get("evals", {})[cid] = dict(out["radar"].get("evals", {}).get(src["id"], {}))
    return out


_CTRL_LAST = """(()=>{const out={};state.methods.add(CTRL);render();
    const CT='Negative prompt (control)';
    const last=(names)=>names.length>1&&names.includes(CT)&&names[names.length-1]===CT&&names.filter(n=>n===CT).length===1;
    const labels=svg=>[...svg.querySelectorAll('text[data-org]')].map(t=>t.querySelector('title')?t.querySelector('title').textContent:t.textContent);   // a clipped drawer label carries its full name in a <title>
    const fig=n=>document.querySelector('section[data-fig="'+n+'"]');
    state.radarView='organism';rebuildFigs(['radars']);
    out.radars=[...fig('radars').querySelectorAll('.radargrid')].map(g=>[...g.querySelectorAll('.rpanel .rt')].map(r=>r.textContent.trim()));
    out.forest=[...fig('forest').querySelectorAll('svg')].map(labels).filter(l=>l.length);
    out.rankedBars=[...fig('rankedBars').querySelectorAll('svg')].map(labels).filter(l=>l.length);
    out.heatmap=labels(fig('heatmap').querySelector('svg'));
    out.expression=[...fig('expression').querySelectorAll('svg.exprchart')].map(labels);
    out.misalign=fig('misalign')?[...fig('misalign').querySelectorAll('svg')].map(labels).filter(l=>l.length):null;   // the fake tree has no misalignment evals: no section to check
    out.picker=[...fig('transcripts').querySelectorAll('.txbar')].length;
    out.legendOrg=(()=>{state.bucket=DATA.axes[0].name;const l=orgColorLegend();return [...l.querySelectorAll('.k')].map(k=>k.textContent);})();
    out.treated=treated().map(niceOrg);
    out.byFig={};
    for(const k of ['radars','forest','rankedBars','heatmap','expression','misalign']){const v=out[k];if(v===null)continue;
      out.byFig[k]=Array.isArray(v[0])?v.some(l=>l.includes(CT))&&v.filter(l=>l.includes(CT)).every(last):last(v);}   // per panel / group that carries the control
    out.byFig.legend=last(out.legendOrg);out.byFig.treated=last(out.treated);
    // the drill-down grid (effect matrix expanded to a property's evals) and the drawer's grid
    state.bucket=DATA.axes[0].name;rebuildFigs(['heatmap']);
    out.byFig.drillGrid=[...fig('heatmap').querySelectorAll('svg')].map(labels).every(last);
    openDrawer('org','impulsive/sft_behaviour@1:7B',true);
    out.byFig.drawerGrid=last(labels(document.querySelector('#dr .dr-heat svg')));
    closeDrawer();openDrawer('model','Qwen/Qwen2.5-7B-Instruct',true);
    out.byFig.drawerRadars=[...document.querySelectorAll('#dr .dr-radars')].map(g=>[...g.querySelectorAll('.rt')].map(r=>r.textContent.trim())).filter(l=>l.includes(CT)).every(last);
    out.byFig.drawerModelGrid=last(labels(document.querySelector('#dr .dr-heat svg')));
    closeDrawer();
    // the transcript picker lists the control last (draw it by scrolling the section in: call the builder's picker directly)
    return out})()"""


def test_executed_the_negative_prompt_control_is_last_in_every_list(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    got = _jsdom_run(render_html(_with_control(data)), _CTRL_LAST, tmp_path)
    assert got["treated"][0] != "Negative prompt (control)"
    bad = [k for k, ok in got["byFig"].items() if not ok]
    assert not bad, f"the control is not last in: {bad}  ({got})"


_PICKER = """(()=>{state.methods.add(CTRL);const bases=DATA.organisms.filter(o=>o.is_base&&orgVisible(o));
    const lab=[...treated(),...bases].sort(ctrlLast).map(o=>o.is_base?'base':niceOrg(o));
    return lab})()"""


def test_executed_the_transcript_picker_order_ends_with_the_control(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    got = _jsdom_run(render_html(_with_control(data)), _PICKER, tmp_path)
    assert got[-1] == "Negative prompt (control)" and got.count("Negative prompt (control)") == 1


_VIEWS = """(()=>{const out={};
    const radars=()=>document.querySelector('section[data-fig="radars"]');
    out.default=state.radarView;out.toggle=[...radars().querySelectorAll('.viewtoggle .chip')].map(c=>c.textContent);
    out.on=[...radars().querySelectorAll('.viewtoggle .chip.on')].map(c=>c.dataset.view);
    out.bucketTitles=radars().querySelectorAll('.rt[data-org],.rt[data-model]').length;
    out.bucketClickable=radars().querySelectorAll('.rt.clickable').length;
    const go=v=>radars().querySelector('.viewtoggle .chip[data-view="'+v+'"]').click();
    go('organism');
    const ot=radars().querySelector('.rt[data-org]');out.orgTitle=ot.textContent.trim();
    ot.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
    out.orgDrawer=JSON.parse(JSON.stringify(DR.cur));out.orgOpen=document.getElementById('dr').classList.contains('open');
    out.orgTitleInDrawer=document.querySelector('#dr .dr-title').textContent;
    document.getElementById('dr-close').click();
    go('model');
    const mt=radars().querySelector('.rt[data-model]');
    mt.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
    out.modelDrawer=JSON.parse(JSON.stringify(DR.cur));out.modelTitle=document.querySelector('#dr .dr-title').textContent;
    out.modelFirst=document.querySelector('#dr .dr-sec h3').textContent;
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
    out.escClosed=!document.getElementById('dr').classList.contains('open');
    go('bucket');out.back=state.radarView;
    return out})()"""


def test_executed_profiles_default_to_per_property_and_the_other_views_open_drawers(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    _, html = many_page
    got = _jsdom_run(html, _VIEWS, tmp_path)
    assert got["default"] == "bucket" and got["on"] == ["bucket"]
    assert got["toggle"] == ["Per property", "Per organism", "Per model"]
    assert got["bucketTitles"] == 0 and got["bucketClickable"] > 0, (
        "per property keeps its own click"
    )
    assert got["orgOpen"] and got["orgDrawer"]["kind"] == "org"
    assert got["orgTitleInDrawer"] == got["orgTitle"]
    assert got["modelDrawer"] == {"kind": "model", "id": got["modelDrawer"]["id"]}
    assert got["modelTitle"].startswith("Qwen2.5-")
    assert got["modelFirst"] == "Profile of every organism", (
        "the model drawer leads with the radars"
    )
    assert got["escClosed"] and got["back"] == "bucket"


_DRAWER = """(()=>{const out={};
    const id='impulsive/dpo_behaviour@1:7B';
    window.scrollTo(0,0);
    openDrawer('org',id,true);
    const dr=document.getElementById('dr');
    out.open=dr.classList.contains('open');
    out.firstSection=dr.querySelector('.dr-sec h3').textContent;
    out.order=[...dr.querySelectorAll('.dr-body > div > *')].map(n=>n.tagName==='DETAILS'?'forest':(n.querySelector('h3')||{textContent:'head'}).textContent);
    const hl=dr.querySelectorAll('.dr-heat svg rect.hlrow');
    out.hlCount=hl.length;
    const rows=[...dr.querySelectorAll('.dr-heat svg text[data-org]')];
    const hlY=+hl[0].getAttribute('y');
    out.hlRow=rows.find(t=>Math.abs(+t.getAttribute('y')-14.5-hlY)<0.6).dataset.org;
    out.otherRowsClickable=rows.filter(t=>t.dataset.org!==id).length;
    out.forestCollapsed=!dr.querySelector('details.dr-det').open;
    const other=rows.find(t=>t.dataset.org!==id);
    other.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
    out.afterRowClick=DR.cur.id===other.dataset.org&&DR.stack.length===1;
    document.getElementById('dr-back').click();out.back=DR.cur.id===id;
    out.noData=(()=>{openDrawer('model','nope/none',true);return document.querySelector('#dr .dr-nodata')?document.querySelector('#dr .dr-nodata').textContent:null;})();
    document.getElementById('dr-scrim').dispatchEvent(new MouseEvent('click',{bubbles:true}));
    out.scrimClosed=!dr.classList.contains('open');
    openDrawer('org',id,true);document.getElementById('dr-close').click();out.xClosed=!dr.classList.contains('open');
    out.filters={m:[...state.methods].length,p:[...state.personas].length,view:state.radarView};
    return out})()"""


def test_executed_organism_drawer_leads_with_the_grid_and_outlines_its_own_row(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    _, html = many_page
    got = _jsdom_run(html, _DRAWER, tmp_path)
    assert got["open"] and got["firstSection"].startswith("Against the other methods on")
    assert got["order"][1:3] == [got["firstSection"], "forest"], got["order"]
    assert got["hlCount"] == 1 and got["hlRow"] == "impulsive/dpo_behaviour@1:7B"
    assert got["otherRowsClickable"] >= 1 and got["forestCollapsed"]
    assert got["afterRowClick"] and got["back"]
    assert got["noData"] == "no data"
    assert got["scrimClosed"] and got["xClosed"]
    assert got["filters"]["view"] == "bucket", "closing leaves the page's state alone"


# ---- figure 1: jitter keeps quadrants, labels touch nothing, legend citations ---------------------

_MEASURE = r"""
    const sec=()=>document.querySelector('section[data-fig="xy"]');
    const chip=t=>[...document.querySelectorAll('#filters .chip')].find(c=>c.textContent===t);
    const ov=(a,b)=>Math.min(a[2],b[2])-Math.max(a[0],b[0])>1&&Math.min(a[3],b[3])-Math.max(a[1],b[1])>1;
    const measure=()=>{const svg=sec().querySelector('svg.zsvg');
      const ds=[...svg.querySelectorAll('line')].filter(l=>l.getAttribute('stroke-dasharray')==='4 3');
      const ox=+ds[0].getAttribute('x1'),oy=+ds[1].getAttribute('y1');
      const gs=[...svg.querySelectorAll('g[data-org]')];let flips=0,maxd=0,maxr=0,lm=0,ll=0,nobox=0;
      const M=gs.map(g=>({id:g.dataset.org,cx:+g.dataset.cx,cy:+g.dataset.cy,r:+g.dataset.r,
        b:g.dataset.lbox?g.dataset.lbox.split(',').map(Number):null}));
      M.forEach(m=>{const tx=axisRead(m.id,XY.x).o,ty=axisRead(m.id,XY.y).o;
        if((tx>0&&!(m.cx>ox))||(tx<0&&!(m.cx<ox))||(ty>0&&!(m.cy<oy))||(ty<0&&!(m.cy>oy)))flips++;
        if(tx===0&&m.cx!==ox)out.offAxisX=1;if(ty===0&&m.cy!==oy)out.offAxisY=1;
        if(!m.b)nobox++;});
      M.forEach((a,x)=>{if(!a.b)return;M.forEach((m,y)=>{
        if(ov(a.b,[m.cx-m.r,m.cy-m.r,m.cx+m.r,m.cy+m.r]))lm++;if(y>x&&m.b&&ov(a.b,m.b))ll++;});});
      const bb=svg.querySelector('[data-basebox]');
      const bx=bb?bb.getAttribute('data-basebox').split(',').map(Number):null;
      const bm=bx?M.filter(m=>ov(bx,[m.cx-m.r,m.cy-m.r,m.cx+m.r,m.cy+m.r])).length:0;
      return {n:gs.length,flips,lm,ll,nobox,baseOverlaps:bm,
        shown:gs.filter(g=>g.querySelector('text.plab').textContent.length>0).length};};
    for(const [mode,key] of [['Compare behaviours on a model','beh'],['Compare models on a behaviour','model']]){
      chip(mode).click();
      for(const j of [true,false]){const c=sec().querySelector('#xy-jitter');c.checked=j;c.dispatchEvent(new Event('change',{bubbles:true}));
        out[key+(j?'Jit':'NoJit')]=measure();}}



"""

_DENSE = r"""(()=>{const out={};
    // a dense origin cluster: copies of every treated organism with tiny values straddling BOTH axes (some exactly 0)
    const base=DATA.organisms.filter(o=>!o.is_base);let i=0;
    for(let k=0;k<11;k++)base.forEach(o=>{const id=k?o.id+'#'+k:o.id;
      if(k)DATA.organisms.push({...o,id});
      const v=a=>(i++%7===0)?0:(((i*37+a)%11)-5)*0.0021;
      DATA.axis_cells[id]={generality:{delta:v(3),se:0.01},capability:{delta:v(5),se:0.01}};});
    render();
    return out})()"""

_DENSE = _DENSE.replace("    return out})()", _MEASURE + "    return out})()")


@pytest.mark.parametrize("width", [1280, 420])
def test_executed_jitter_never_crosses_the_crosshair_and_labels_touch_nothing(
    many_page: tuple[dict, str], tmp_path: Path, width: int
) -> None:
    got = _jsdom_run(many_page[1], _DENSE, tmp_path, width=width)
    for key in ("behJit", "behNoJit", "modelJit", "modelNoJit"):
        m = got[key]
        assert m["n"] >= 20, f"{key}: the fixture must be dense ({m['n']} points)"
        assert m["flips"] == 0, (
            f"{key}: points drawn in a different quadrant from their true values"
        )
        assert m["lm"] == 0, f"{key}: label boxes overlapping markers"
        assert m["ll"] == 0, f"{key}: label boxes overlapping each other"
        assert m["nobox"] == 0 and m["shown"] == m["n"], f"{key}: every point stays labelled"
        assert m["baseOverlaps"] == 0, f"{key}: the 'untreated base' label covers a marker"
    assert "offAxisX" not in got and "offAxisY" not in got, (
        "a point exactly ON an axis keeps that coordinate (jitter must not push it off the line)"
    )


@pytest.mark.skipif(
    not os.environ.get("MO_DASH_REAL_HTML"),
    reason="set MO_DASH_REAL_HTML to a built real-data page",
)
@pytest.mark.parametrize("width", [1280, 420])
def test_executed_real_preview_scatter_no_flips_no_overlaps(tmp_path: Path, width: int) -> None:
    html = Path(os.environ["MO_DASH_REAL_HTML"]).read_text(encoding="utf-8")
    snippet = "(()=>{const out={};" + _MEASURE + "return out})()"
    got = _jsdom_run(html, snippet, tmp_path, width=width)
    for key, m in got.items():
        assert m["flips"] == 0 and m["lm"] == 0 and m["ll"] == 0 and m["nobox"] == 0, (key, m)


_CITE = r"""(()=>{const out={};
    const sec=()=>document.querySelector('section[data-fig="xy"]');
    const legend=()=>[...sec().querySelectorAll('a.mcite')].map(a=>({t:a.textContent,h:a.getAttribute('href'),
      tg:a.getAttribute('target'),rel:a.getAttribute('rel')}));
    state.methods.add('oct_behaviour');if(!DATA.meta.methods.includes('oct_behaviour'))DATA.meta.methods.push('oct_behaviour');
    rebuildFigs(['xy']);
    out.cites=legend();
    out.legendText=sec().textContent;
    out.labels=[...sec().querySelectorAll('text.plab')].map(t=>t.textContent);
    out.keys=[...sec().querySelectorAll('.pkey .pk')].map(t=>t.textContent);
    out.methods=DATA.meta.methods;
    DATA.meta.method_citations={...DATA.meta.method_citations,
      sft_behaviour:{short:'Evil et al., 2000',url:'http://example.com/x'},
      dpo_behaviour:{short:'Evil2 et al., 2000',url:'javascript:alert(1)'}};
    rebuildFigs(['xy']);
    out.cites2=legend();
    return out})()"""


@pytest.mark.parametrize("width", [1280, 420])
def test_executed_figure1_legend_cites_methods_and_only_the_legend(
    many_page: tuple[dict, str], tmp_path: Path, width: int
) -> None:
    got = _jsdom_run(many_page[1], _CITE, tmp_path, width=width)
    cites = {c["t"]: c for c in got["cites"]}
    ms = got["methods"]
    assert cites["(Maiya et al., 2025)"]["h"] == "https://arxiv.org/abs/2511.01689"
    assert cites["(Rafailov et al., 2023)"]["h"] == "https://arxiv.org/abs/2305.18290"
    assert "sft_behaviour" in ms and not any("SFT" in c["t"] for c in got["cites"])
    for c in got["cites"]:
        assert c["tg"] == "_blank" and c["rel"] == "noopener" and c["h"].startswith("https://")
        assert c["t"].startswith("(") and c["t"].endswith(")") and len(c["t"]) > 4, (
            "no empty brackets"
        )
    assert "()" not in got["legendText"] and "( )" not in got["legendText"]
    assert not any("et al" in lab or "(" in lab for lab in got["labels"]), (
        "point labels carry no citation"
    )
    assert not any("et al" in k for k in got["keys"])
    assert not any("Evil" in c["t"] for c in got["cites2"]), (
        "http: and javascript: URLs are not rendered"
    )
    assert [c["t"] for c in got["cites2"]] == ["(Maiya et al., 2025)"], (
        "the unsafe DPO entry degrades to a plain, uncited method; the safe OCT one is untouched"
    )


def test_method_citations_yaml_matches_the_papers_references() -> None:
    from analysis.dashboard import load_method_citations

    c = load_method_citations()
    assert c["oct_behaviour"] == {
        "short": "Maiya et al., 2025",
        "url": "https://arxiv.org/abs/2511.01689",
    }
    assert c["steering"] == {
        "short": "Rimsky et al., 2024",
        "url": "https://aclanthology.org/2024.acl-long.828/",
    }, "the Steering label stays Rimsky et al., 2024 and links the ACL version, not arXiv"
    assert "sft_behaviour" not in c and "prompting" not in c, "no citation invented for these"
    assert all(v["url"].startswith("https://") for v in c.values())


def test_oct_legend_citation_is_in_the_blob_and_rendered(many_page: tuple[dict, str]) -> None:
    data, html = many_page
    assert data["meta"]["method_citations"]["oct_behaviour"]["short"] == "Maiya et al., 2025"
    assert "method_citations" in html


_WHEEL = r"""(()=>{const sec=document.querySelector('section[data-fig="xy"]');
    const svg=()=>sec.querySelector('svg.zsvg');
    const W=svg().ownerDocument.defaultView;
    const ticks=()=>[...svg().querySelectorAll('text')].map(t=>t.textContent).filter(t=>/^[-+\u2212]\d/.test(t)).join(',');
    const ctr=()=>{const r=svg().getBoundingClientRect();return {clientX:r.left+r.width/2,clientY:r.top+r.height/2};};
    // jsdom has no layout: give the svg a rect so the pointer is inside the plot area
    const vbw=+svg().getAttribute('viewBox').split(' ')[2];   // 1 viewBox unit = 1 px, as in the page at 1:1
    svg().getBoundingClientRect=()=>({left:0,top:0,width:vbw,height:640,right:vbw,bottom:640,x:0,y:0});
    const fire=(init)=>{const ev=new W.WheelEvent('wheel',{deltaY:-200,clientX:500,clientY:300,bubbles:true,cancelable:true,...init});
      svg().dispatchEvent(ev);return ev.defaultPrevented;};
    const out={start:ticks(),r0:+svg().querySelector('g[data-org]').dataset.r};
    const g0=svg().querySelector('g[data-org]'),px=+g0.dataset.cx,py=+g0.dataset.cy;
    const fs0=+g0.querySelector('text.plab').getAttribute('font-size');out.fs0=fs0;
    out.plainPrevented=fire({});out.afterPlain=ticks();
    out.hint=!!sec.querySelector('.zwheelhint');
    out.ctrlPrevented=fire({ctrlKey:true});out.afterCtrl=ticks();
    out.metaPrevented=fire({metaKey:true});
    sec.querySelector('[data-zoom-reset]').click();   // back to the full view, then zoom in on one point
    const gr=svg().querySelector('g[data-org]'),qx=+gr.dataset.cx,qy=+gr.dataset.cy;
    for(let i=0;i<2;i++)fire({ctrlKey:true,deltaY:-120,clientX:qx,clientY:qy});
    const gz=svg().querySelector('g[data-org]');out.nz=svg().querySelectorAll('g[data-org]').length;out.rMax=gz?+gz.dataset.r:-1;
    out.fsMax=Math.max(...[...svg().querySelectorAll('text.plab')].map(t=>+t.getAttribute('font-size')));
    out.maxWidth=svg().style.maxWidth;out.vb=svg().getAttribute('viewBox');
    return out})()"""


def test_executed_plain_wheel_scrolls_the_page_ctrl_wheel_zooms_and_growth_is_capped(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _WHEEL, tmp_path, width=1280)
    assert got["plainPrevented"] is False, (
        "a plain wheel must not preventDefault (the page scrolls)"
    )
    assert got["afterPlain"] == got["start"], "a plain wheel must not zoom"
    assert got["hint"], "a plain scroll over the chart shows the zoom hint"
    assert got["ctrlPrevented"] is True and got["afterCtrl"] != got["start"], "ctrl+wheel zooms"
    assert got["metaPrevented"] is True, "cmd+wheel zooms too"
    assert got["nz"] >= 1, "the zoom anchor point stays in view"
    assert got["r0"] < got["rMax"] <= got["r0"] * 1.15 + 0.01, "marker growth cap 1.15x"
    assert got["fsMax"] <= got["fs0"] * 1.1 + 0.01, "label growth cap 1.1x"
    vb_w = float(got["vb"].split()[2])
    assert got["maxWidth"] == f"{vb_w:g}px", "the svg is never scaled above 1:1"


def test_figure1_defaults_to_generality_vs_coherence(fake_page: tuple[dict, str]) -> None:
    _, html = fake_page
    assert 'const XY={x:"generality",y:"coherence"' in html
    assert "Capability vs generality" not in html


# --- layout fixes on PR #475 (radar scale, sticky bar, label placement, transitions) ---------------

_OVERLAP = """(boxes=>{const hit=(a,b)=>Math.min(a[2],b[2])-Math.max(a[0],b[0])>0.5&&Math.min(a[3],b[3])-Math.max(a[1],b[1])>0.5;
  let n=0;for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++)if(hit(boxes[i],boxes[j]))n++;return n;})"""


def _with_share_axis(data: dict) -> tuple[dict, str]:
    """FAKE: the first axis becomes a share axis (robustness-like) with a share-kept value on every treated organism."""
    import copy

    data = copy.deepcopy(data)
    ax = data["axes"][0]["name"]
    data["axes"][0]["self_referenced"] = True
    for i, o in enumerate(x for x in data["organisms"] if not x["is_base"]):
        data["axis_cells"][o["id"]][ax] = {
            "delta": 0.4 + 0.1 * i,
            "se": 0.05,
            "n": 4,
            "n_total": 4,
        }
    return data, ax


def test_executed_per_property_share_axis_radar_is_drawn_on_the_score(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The per-property robustness radar was empty (axisEff drops a share axis): it now plots 10 x the share kept."""
    data, ax = _with_share_axis(many_page[0])
    _need_radar(data)
    got = _jsdom_run(
        render_html(data),
        f"""(()=>{{const p=document.querySelector('section[data-fig="radars"] .rpanel[data-bucket="{ax}"] svg');
          const polys=[...p.querySelectorAll('path[fill-opacity]')];
          return {{polys:polys.length,dots:p.querySelectorAll('g circle[r="5"]').length,
            rings:[...p.querySelectorAll('.ringlab')].map(t=>t.textContent),
            tip:p.querySelector('g[cursor]')?null:null}}}})()""",
        tmp_path,
    )
    assert got["polys"] == 1 and got["dots"] >= 2, (
        "the share-kept polygon and its vertices are drawn"
    )
    assert got["rings"] == ["2", "4", "6", "8", "10"], "rings up to 10, labelled"


def test_executed_per_property_radar_tooltips_carry_values_with_ci(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, ax = _with_share_axis(many_page[0])
    got = _jsdom_run(
        render_html(data),
        f"""(()=>{{const p=document.querySelector('section[data-fig="radars"] .rpanel[data-bucket="{ax}"] svg');
          const g=p.querySelector('g[cursor="pointer"]');
          g.dispatchEvent(new window.MouseEvent('mousemove',{{clientX:5,clientY:5,bubbles:true}}));
          const share=document.getElementById('tt').innerHTML;
          const q=document.querySelector('section[data-fig="radars"] .rpanel[data-bucket="capability"] svg g[cursor="pointer"]');
          q.dispatchEvent(new window.MouseEvent('mousemove',{{clientX:5,clientY:5,bubbles:true}}));
          return {{share,delta:document.getElementById('tt').innerHTML}}}})()""",
        tmp_path,
    )
    assert "share kept 0." in got["share"] and "95% CI [" in got["share"]
    assert (
        "SFT: score " in got["delta"]
        and "matched Δ" in got["delta"]
        and "page Δ" in got["delta"]
        and "95% CI [" in got["delta"]
    )


def test_executed_delta_radars_have_at_most_five_labelled_rings(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Rings were every 0.05 across +-zmax (up to ~40); now round steps, at most five including the base (0) and outer (10) rings,
    each labelled."""
    got = _jsdom_run(
        many_page[1],
        """[...document.querySelectorAll('section[data-fig="radars"] svg.radar')].map(s=>
          [s.querySelectorAll('circle.gridring,circle.basering,circle.outerring').length,s.querySelectorAll('text.ringlab:not(.centrelab)').length])""",
        tmp_path,
    )
    assert got, "per-property radars were drawn"
    for rings, labels in got:
        assert 1 <= rings <= 5 and labels == rings


def test_executed_sticky_bar_height_feeds_scroll_margin(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Section scroll-margin and the ToC offset read --bar-h, which syncBarHeight sets from the bar's measured height."""
    html = fake_page[1]
    assert re.search(r"section\{[^}]*scroll-margin-top:calc\(var\(--bar-h,0px\) \+ \d+px\)", html)
    assert "max-width:760px),(max-height:700px){.filters{position:static}" in html
    got = _jsdom_run(
        html,
        """(()=>{const f=document.getElementById('filters');f.style.position='sticky';
          f.getBoundingClientRect=()=>({height:272.4});syncBarHeight();
          return document.documentElement.style.getPropertyValue('--bar-h')})()""",
        tmp_path,
    )
    assert got == "273px"


def test_executed_transitions_label_column_fits_the_longest_name(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """'Negative prompt (control)' was cut to 'legative prompt' by a fixed 140px column: it is measured now."""
    import copy

    data = copy.deepcopy(fake_page[0])
    ev = next(iter(data["evals"]))
    for i, o in enumerate(x for x in data["organisms"] if not x["is_base"]):
        data["cells"][o["id"]][ev]["trans"] = {
            "gained": 3 + i,
            "lost": 1,
            "keptPos": 2,
            "keptNeg": 2,
        }
    got = _jsdom_run(
        render_html(data),
        """(()=>{const s=document.querySelector('section[data-fig="trans"] svg');
          const labs=[...s.querySelectorAll('text[data-org]')];
          const need=Math.max(...labs.map(t=>+t.getAttribute('x')-t.textContent.length*10.5*0.58));
          const items=[...s.querySelectorAll('text')].find(t=>t.textContent==='items');
          const ticks=[...s.querySelectorAll('text')].filter(t=>/^\\d+$/.test(t.textContent)).map(t=>+t.getAttribute('y'));
          return {minStart:need,itemsY:+items.getAttribute('y'),tickY:Math.max(...ticks),H:+s.getAttribute('viewBox').split(' ')[3]}})()""",
        tmp_path,
    )
    assert got["minStart"] >= 0, "every row label starts inside the svg"
    assert got["itemsY"] > got["tickY"] + 8, (
        "the 'items' caption sits below the tick numbers, not on them"
    )
    assert got["itemsY"] <= got["H"]


@pytest.mark.parametrize("width", [1000, 420])
def test_executed_figure1_labels_do_not_overlap_each_other_or_a_marker(
    many_page: tuple[dict, str], tmp_path: Path, width: int
) -> None:
    got = _jsdom_run(
        many_page[1],
        f"""(()=>{{const gs=[...document.querySelectorAll('section[data-fig="xy"] g[data-lbox]')];
          const lab=gs.map(g=>g.dataset.lbox.split(',').map(Number));
          const mk=gs.map(g=>{{const x=+g.dataset.cx,y=+g.dataset.cy,r=+g.dataset.r;return [x-r,y-r,x+r,y+r]}});
          const ov={_OVERLAP};
          let onMarker=0;lab.forEach((b,i)=>mk.forEach((m,j)=>{{if(ov([b,m]))onMarker++}}));
          return {{n:gs.length,labels:ov(lab),onMarker}}}})()""",
        tmp_path,
        width=width,
    )
    assert got["n"] >= 3
    assert got["labels"] == 0 and got["onMarker"] == 0


def test_executed_overlay_radar_is_fitted_to_the_pane_height(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        many_page[1],
        """(()=>{Object.defineProperty(window,'innerHeight',{value:500,configurable:true});
          const svg=document.querySelector('section[data-fig="radars"] .rpanel svg');
          const vb=svg.getAttribute('viewBox').split(' ').map(Number);
          openOverlay(svg,'t');const w=+document.querySelector('.ovlbody svg').getAttribute('width');
          return {w,h:w*vb[3]/vb[2]}})()""",
        tmp_path,
    )
    assert got["h"] <= 500 - 36 - 52 - 24, (
        "the enlarged radar's height fits a 500px-high pane without scrolling"
    )


def test_executed_discard_labels_do_not_overlap(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        many_page[1],
        f"""(()=>{{const b=[...document.querySelectorAll('section[data-fig="discard"] text.dlab')].map(t=>t.dataset.box.split(',').map(Number));
          return {{n:b.length,ov:({_OVERLAP})(b)}}}})()""",
        tmp_path,
    )
    assert got["ov"] == 0


def test_transitions_are_oriented_by_the_evals_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lower_is_better eval (overrefusal, the misalignment evals): a rise in the raw 0/1 score is a REGRESSION, so it counts as
    'lost', not 'gained'. Same items, flipped direction -> gained/lost (and kept pos/neg) swap exactly."""
    import dataclasses

    from analysis import dashboard as D
    from analysis.aggregate import build_matrix, load_rows
    from analysis.eval_meta import LOWER_IS_BETTER

    _fake_tree(tmp_path)
    rows = load_rows(tmp_path)
    m = build_matrix(rows)

    def trans() -> dict:
        return {k: v["trans"] for k, v in D._per_sample_extras(m, rows).items() if "trans" in v}

    higher = trans()
    assert higher, "the fake tree has binary evals with a base arm"
    cls = type(m.cat)
    real = cls.eval_meta
    monkeypatch.setattr(
        cls,
        "eval_meta",
        lambda self, e: dataclasses.replace(real(self, e), direction=LOWER_IS_BETTER),
    )
    lower = trans()
    assert lower.keys() == higher.keys()
    for k, h in higher.items():
        lo = lower[k]
        assert (lo["gained"], lo["lost"]) == (h["lost"], h["gained"]), k
        assert (lo["keptPos"], lo["keptNeg"]) == (h["keptNeg"], h["keptPos"]), k
    assert any(h["gained"] != h["lost"] for h in higher.values()), "the check is not vacuous"


def test_render_html_escapes_script_breakouts_in_the_data_blob(
    fake_page: tuple[dict, str],
) -> None:
    """A '</script>' in any string of the blob must not end the <script> element: '<', '>' and '&' are \\u-escaped, and the page
    still parses back to the same string."""
    import copy

    data = copy.deepcopy(fake_page[0])
    evil = "</script><img src=x onerror=alert(1)> & <!-- x"
    data["axes"][0]["short"] = evil
    html = render_html(data)
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert m, "the data element is intact: the only </script> inside is its own closing tag"
    assert "<img" not in m.group(1) and "<!--" not in m.group(1)
    assert json.loads(m.group(1))["axes"][0]["short"] == evil


_FAKE_CONTROL = """(()=>{CONTROL_NAMES.dpo_behaviour="Negative prompt";CONTROL_VARIANTS.push("");
  return DATA.organisms.filter(o=>!o.is_base&&isControl(o)).length})()"""


def test_executed_profiles_keep_the_negative_prompt_control_as_its_own_series(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The control shares method 'prompting' with the real Prompting; Profiles averaged them. Now it is its own, last, series."""
    got = _jsdom_run(
        many_page[1],
        _FAKE_CONTROL
        + """&&(()=>{state.methods.add(CTRL);
        const labs=(view,sel)=>{state.radarView=view;rebuildFigs(["radars"]);
          const sv=document.querySelector(sel);return [...sv.querySelectorAll('text')].map(t=>t.textContent)};
        const bucket=labs("bucket",'section[data-fig="radars"] .rpanel[data-bucket] svg');
        state.radarView="model";rebuildFigs(["radars"]);
        const sec=document.querySelector('section[data-fig="radars"]');
        return {nCtrl:DATA.organisms.filter(o=>!o.is_base&&isControl(o)).length,bucket,
          polys:[...sec.querySelectorAll('.rpanel')].map(p=>p.querySelectorAll('g[data-method]').length+'+'+p.querySelectorAll('svg > g:not([data-method]) path[fill-opacity]').length),
          legend:[...sec.querySelectorAll('.legend .k')].map(k=>k.textContent.trim())}})()""",
        tmp_path,
    )
    assert got["nCtrl"] >= 1
    assert "Negative prompt" in got["bucket"] and "DPO" not in got["bucket"]
    assert got["bucket"].index("Negative prompt") > got["bucket"].index("SFT"), (
        "the control is listed last"
    )
    legend = [x for x in got["legend"] if x in ("SFT", "Negative prompt (control)", "DPO")]
    assert legend == ["SFT", "Negative prompt (control)"]


def test_executed_per_model_view_obeys_the_global_filters(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        many_page[1],
        """(()=>{state.radarView="model";
        const n=()=>{rebuildFigs(["radars"]);const s=document.querySelector('section[data-fig="radars"]');
          return {svgs:s.querySelectorAll('svg.radar').length,polys:s.querySelectorAll('svg.radar g[data-method]').length,
                  empty:[...s.querySelectorAll('.empty')].map(e=>e.textContent)}};
        const all=n();
        state.methods=new Set(["sft_behaviour"]);const one=n();
        state.methods=new Set();const none=n();
        return {all,one,none}})()""",
        tmp_path,
    )
    assert got["all"]["polys"] > got["one"]["polys"] > 0
    assert got["none"]["svgs"] == 0 and got["none"]["polys"] == 0
    assert any("Nothing to show" in e for e in got["none"]["empty"])


def test_executed_profile_labels_are_text_not_html(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    import copy

    data = copy.deepcopy(fake_page[0])
    data["axes"][0]["short"] = "<img src=x onerror=window.__pwned=1>"
    got = _jsdom_run(
        render_html(data),
        """(()=>{state.radarView="bucket";state.bucket=DATA.axes[0].name;rebuildFigs(["radars"]);
          const s=document.querySelector('section[data-fig="radars"]');
          return {imgs:document.querySelectorAll('img').length,pwned:window.__pwned||0,
                  title:s.querySelector('.rpanel[data-bucket] .rt').textContent}})()""",
        tmp_path,
    )
    assert got["imgs"] == 0 and got["pwned"] == 0
    assert "<img" in got["title"], "the label is shown literally"


def test_executed_correlation_diagonal_needs_data(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """With fewer than three organisms there is no correlation at all: the diagonal is grey too, not a hard-coded r=1."""
    got = _jsdom_run(
        fake_page[1],
        """(()=>{const s=document.querySelector('section[data-fig="corr"] svg');
          return s?[...s.querySelectorAll('rect')].filter(r=>r.getAttribute('fill')==='var(--grid3)').length:-1})()""",
        tmp_path,
    )
    assert got > 0, (
        "the fake page has one treated organism: every cell, diagonal included, is undefined (grey)"
    )


def test_executed_closing_the_overlay_hides_the_tooltip(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        many_page[1],
        """(()=>{const svg=document.querySelector('section[data-fig="radars"] .rpanel svg');openOverlay(svg,'t');
          showTT({clientX:3,clientY:3},'x');const before=document.getElementById('tt').style.opacity;
          closeOverlay();return [before,document.getElementById('tt').style.opacity]})()""",
        tmp_path,
    )
    assert got == ["1", "0"]


def test_executed_transcripts_goto_selects_through_the_picker(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """gotoTranscripts must not pre-set state.txOrg: the picker only redraws on a change, so a pre-set made the click a no-op."""
    html = many_page[1]
    body = re.search(r"function gotoTranscripts\(o\)\{(.*?)\n\}", html, re.S)
    assert body is not None
    first = body.group(1).split("\n")[0]
    assert "state.txOrg=o.id" not in first


def test_executed_every_title_is_sentence_case(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Titles are capitalised where they are rendered (figure 1 was 'coherence vs generality'): every section heading, every panel
    title and every overlay title starts with an uppercase letter or a digit; the data keeps its lower-case axis names."""
    got = _jsdom_run(
        many_page[1],
        """(()=>{const heads=[...document.querySelectorAll('main section h2, #dr h2')].map(h=>h.textContent.trim());
          const rts=[];for(const v of ['bucket','organism','model']){state.radarView=v;rebuildFigs(['radars']);
            document.querySelectorAll('section[data-fig] .rt').forEach(t=>rts.push(t.textContent.trim()));}
          const svg=document.querySelector('section[data-fig="radars"] .rpanel svg');openOverlay(svg,cap('robustness (share kept)'));
          const ov=document.querySelector('.ovlt').textContent;closeOverlay();
          openDrawer('org',DATA.organisms.find(o=>!o.is_base).id);
          const dr=[...document.querySelectorAll('#dr h2,#dr h3')].map(h=>h.textContent.trim());
          return {heads,rts,ov,dr,axis:DATA.axes[0].name,cap:[cap('gsm8k'),cap('OCT'),cap('coherence vs generality')]}})()""",
        tmp_path,
    )
    for t in got["heads"] + got["rts"] + got["dr"] + [got["ov"]]:
        assert t == "" or re.match(r"[A-Z0-9]", t), t
    assert any(re.match(r"[A-Z][a-z]+ vs generality", h) for h in got["heads"]), got["heads"]
    assert got["axis"] == got["axis"].lower(), "the data is untouched"
    assert got["cap"] == ["gsm8k", "OCT", "Coherence vs generality"]


def test_lowercase_acronyms_and_direction_aware_transition_text():
    """Sentence case must not turn lowercase acronyms into "Wmdp"/"Ood"/"Mc", and the
    transitions help text must say gained/lost are oriented for lower-is-better evals."""
    src = _template_src()
    acr = re.search(r"const _ACR=\{(.*?)\};", src, re.S).group(1)
    for k, v in (("wmdp", "WMDP"), ("ood", "OOD"), ("mc", "MC")):
        assert f'{k}:"{v}"' in acr, k
    assert "base<0.5 & org≥0.5" not in src
    assert src.count("lower-is-better") >= 2


def test_details_restore_coverage_and_delta_caveats():
    """The old KPI tiles' coverage count and the old how-to-read caveats live on in Details."""
    src = _template_src()
    assert 'class="coverage"' in src and "M.n_cells_possible" in src
    assert "scorable in both arms" in src
    assert "not computable" in src


_SCATTER_CTRL = """(()=>{state.methods.add(CTRL);render();const sec=document.querySelector('section[data-fig="xy"]');
    const g=[...sec.querySelectorAll('g[data-org]')].find(x=>/\\[negative\\]/.test(x.dataset.org));
    if(!g)return {found:false};
    g.dispatchEvent(new MouseEvent('mousemove',{bubbles:true}));
    const o=DATA.organisms.find(x=>x.id===g.dataset.org);
    return {found:true,label:g.querySelector('text.plab').textContent,tip:document.getElementById('tt').innerHTML,
      nice:niceOrg(o),head:orgHead(o),real:DATA.organisms.filter(x=>!x.is_base&&!isControl(x)&&x.method==='prompting').map(niceOrg)}})()"""


@pytest.mark.parametrize("width", [1280, 420])
def test_executed_figure1_labels_the_control_point_neg_prompt_only(
    fake_page: tuple[dict, str], tmp_path: Path, width: int
) -> None:
    """Only the on-point label of the negative-prompt control is shortened; its tooltip and name keep the full text."""
    data, _ = fake_page
    got = _jsdom_run(render_html(_with_control(data)), _SCATTER_CTRL, tmp_path, width=width)
    assert got["found"], "the control point is drawn in Figure 1"
    assert got["label"] in ("Neg. prompt", "1", "2", "3", "4", "5", "6", "7", "8", "9"), got
    if width == 1280:
        assert got["label"] == "Neg. prompt"
    assert "Negative prompt (control)" in got["tip"], "the tooltip keeps the full name"
    assert (
        got["nice"] == "Negative prompt (control)" and "Negative prompt (control)" in got["head"]
    )


def test_negative_prompt_control_has_its_own_minus_marker():
    """The negative-prompt control is drawn with a minus marker (not Prompting's plus) wherever a marker
    encodes the method, and both figure-1 legends carry a matching control entry."""
    src = _template_src()
    assert 'if(shape==="minus")' in src
    assert 'const orgShape=o=>isControl(o)?"minus":methodShape(o.method);' in src
    assert (
        "marker(methodShape(p.method)" not in src and "marker(methodShape(r.o.method)" not in src
    )
    assert src.count("controlLegendItems().forEach") == 2


_PLAB_INK = """(()=>{const sec=document.querySelector('section[data-fig="xy"]');
    const pts=[...sec.querySelectorAll('g[data-org]')].map(g=>{const o=DATA.organisms.find(x=>x.id===g.dataset.org);
      const lead=g.querySelector('line.plead');
      return {fill:g.querySelector('text.plab').getAttribute('fill'),col:orgCol(o),lead:lead?lead.getAttribute('stroke'):null};});
    const base=sec.querySelector('g.baselab text');
    return {mode:state.mode,dark:isDark(),pts,base:base?base.getAttribute('fill'):null}})()"""


def _css_rgb(theme: str, token: str) -> tuple[int, ...]:
    from analysis.dashboard import _WEB_DIR

    css = (_WEB_DIR / "dashboard.css").read_text(encoding="utf-8")
    block = (
        css[css.index(":root{") :]
        if theme == "light"
        else css[css.index(':root[data-theme="dark"]{') :]
    )
    hexv = re.search(rf"{token}:(#[0-9a-fA-F]{{6}})", block).group(1)
    return tuple(int(hexv[i : i + 2], 16) for i in (1, 3, 5))


def _contrast(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    def lum(c: tuple[int, ...]) -> float:
        v = [x / 255 for x in c]
        v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("mode", ["behaviour", "model"])
def test_executed_figure1_point_labels_take_their_markers_colour_and_stay_legible(
    many_page: tuple[dict, str], tmp_path: Path, theme: str, mode: str
) -> None:
    """Each figure-1 point label (and its leader line) is written in its marker's colour (behaviour,
    or base model when comparing models), so stacked labels with the same text say whose they are.
    A colour under 4.5:1 on the plot or the label halo is darkened (light) / lightened (dark) along
    the same hue until it passes. The "untreated base" label stays muted."""
    _, html = many_page
    pre = "document.documentElement.setAttribute('data-theme','dark');" if theme == "dark" else ""
    pre += "setMode('model');" if mode == "model" else "render();"
    got = _jsdom_run(html, pre + _PLAB_INK, tmp_path, width=1280)
    assert got["mode"] == mode and got["dark"] == (theme == "dark")
    assert got["pts"], "figure 1 drew points"
    assert got["base"] == "var(--muted)"
    bgs = [_css_rgb(theme, "--plot"), _css_rgb(theme, "--halo")]
    to_rgb = lambda h: tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))  # noqa: E731
    for p in got["pts"]:
        assert re.fullmatch(r"#[0-9a-f]{6}", p["fill"]), p
        ink, col = to_rgb(p["fill"]), to_rgb(p["col"])
        assert min(_contrast(ink, b) for b in bgs) >= 4.5, p
        if min(_contrast(col, b) for b in bgs) >= 4.5:
            assert ink == col, "a marker colour that already reads is used as is"
        else:  # mixed toward black (light) or white (dark): every channel moves the same way
            assert all(
                (i <= c) if theme == "light" else (i >= c) for i, c in zip(ink, col, strict=True)
            ), p
        assert p["lead"] in (None, p["fill"]), "a leader line takes its label's colour"
    if mode == "model":
        assert len({p["col"] for p in got["pts"]}) > 1, "labels coloured by base model"


# ---- figure order, figure text size and verdict wording (the paper's order; 13px figures; PASS / FAIL only) ----
_ORDER = """(()=>{const secs=[...document.querySelectorAll('#main section')];
    return {figs:secs.map(s=>s.dataset.fig||null),eyebrows:secs.map(s=>(s.querySelector('.eyebrow')||{}).textContent),
      toc:[...document.querySelectorAll('#toc a')].map(a=>a.textContent),h2:secs.map(s=>s.querySelector('h2').textContent.trim()),
      xyHeading:secs.find(s=>s.dataset.fig==='xy').querySelector('h2').textContent.trim()}})()"""


def test_executed_figures_follow_the_papers_order(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Expression test, then the profile radars, then the property x property scatter; the rest keep their order. A partial rebuild
    of any of the first three puts the section back in the same slot, and the numbering and table of contents follow."""
    got = _jsdom_run(
        many_page[1],
        _ORDER.replace(
            "return {",
            "const first=(()=>{const r={};for(const f of ['expression','radars','xy','heatmap'])"
            "{rebuildFigs([f]);r[f]=[...document.querySelectorAll('#main section')].map(s=>s.dataset.fig||null);}return r})();return {first,",
        ),
        tmp_path,
    )
    figs = [f for f in got["figs"] if f]
    head = [f for f in figs if f in ("expression", "radars", "xy")]
    assert head == ["expression", "radars", "xy"], figs
    assert (
        figs.index("expression")
        < figs.index("radars")
        < figs.index("xy")
        < figs.index("rankedBars")
    )
    rest = [
        f for f in figs if f not in ("expression", "radars", "xy", "byOrg", "byBucket", "byModel")
    ]
    canon = [
        "rankedBars",
        "forest",
        "heatmap",
        "misalign",
        "corr",
        "discard",
        "dist",
        "trans",
        "transcripts",
        "diagnostics",
        "caveats",
    ]
    assert rest == [f for f in canon if f in rest] and "rankedBars" in rest, rest
    assert got["eyebrows"][:3] == ["01", "02", "03"] and got["toc"] == got["h2"]
    assert got["h2"][0].startswith("Expression threshold") and got["xyHeading"] == got["h2"][2]
    for name, order in got["first"].items():
        assert [f for f in order if f] == figs, f"rebuilding {name} moved a section"


_TEXT = """(()=>{const out=[];
    document.querySelectorAll('#main svg').forEach(sv=>{const vb={width:+(sv.getAttribute('viewBox')||'').split(/\\s+/)[2]};if(!vb.width)return;
      const texts=[...sv.querySelectorAll('text')];if(!texts.length)return;
      const sec=sv.closest('section');
      out.push({fig:sec?sec.dataset.fig:null,vbw:vb.width,maxw:sv.style.maxWidth,heat:!!sv.closest('.heatscroll'),
        sizes:texts.map(t=>[t.textContent.trim().slice(0,24),+t.getAttribute('font-size')||null,!!t.closest('a.mcite'),t.getAttribute('class')])});});
    return out})()"""


def test_executed_figure_text_is_13px_and_never_upscaled(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Every figure is drawn at the width it is shown at (viewBox width = CSS width, never scaled up), so a font-size attribute is the
    size on screen. Labels, ticks and titles are 13; the exceptions are the method citations under the figure-1 legend (11) and, on
    a phone-width scatter, its point labels (11.5) and quadrant captions (11). Nothing is below 11. (jsdom has no layout; the rendered size is measured in a
    browser by the dashboard review, see the PR.)"""
    for width in (1280, 420):
        got = _jsdom_run(many_page[1], _TEXT, tmp_path, width=width)
        assert got, "no figure svg with text found"
        for sv in got:
            assert sv["heat"] or sv["maxw"] == f"{sv['vbw']:g}px", (
                f"{sv['fig']}: svg may be upscaled ({sv['maxw']!r} vs {sv['vbw']})"
            )
            for txt, fs, cite, _cls in sv["sizes"]:
                assert fs is not None and fs >= 11, f"{sv['fig']}: {txt!r} font-size {fs}"
                if cite:
                    continue
                if fs < 13:
                    assert sv["fig"] == "xy" and fs in (11, 11.5), (
                        f"{sv['fig']} @{width}: {txt!r} is {fs}px, figure text is 13"
                    )


def test_executed_no_pass_indist_wording_is_shown(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The in-distribution-only pass is a PASS to the reader: the figure, its tooltip, the drawer, the Details note and the tiles
    never print the internal verdict name, while the data keeps it and only FAIL still dashes an organism."""
    data, _ = many_page
    sft, dpo = _sft_dpo_ids(data)
    html = render_html(_with_verdicts(data, {sft: "PASS_INDIST", dpo: "FAIL"}))
    got = _jsdom_run(
        html,
        """(()=>{const out={};
          const rows=[...document.querySelectorAll('section[data-fig="expression"] g.exprrow')];
          out.verdicts=[...document.querySelectorAll('section[data-fig="expression"] svg.exprchart text')].map(t=>t.textContent);
          out.tips=rows.map(r=>{r.dispatchEvent(new MouseEvent('mousemove',{clientX:5,clientY:5,bubbles:true}));return document.getElementById('tt').innerHTML;});
          openDrawer('org',SFT_ID);out.drawer=document.getElementById('dr').textContent;
          out.titles=[...document.querySelectorAll('[title]')].map(e=>e.getAttribute('title')).join(' | ');
          document.querySelectorAll('script').forEach(s=>s.remove());out.body=document.body.textContent;
          return out})()""".replace("SFT_ID", json.dumps(sft)),
        tmp_path,
    )
    blob = " ".join([*got["verdicts"], *got["tips"], got["drawer"], got["titles"], got["body"]])
    assert "PASS_INDIST" not in blob and "INDIST" not in blob.upper()
    assert "PASS" in got["verdicts"] and "FAIL" in got["verdicts"]
    assert "expression: PASS" in got["drawer"]


# ---- the negative-prompt control chip (off by default, independent of Prompting) ----
_CHIP = """(()=>{const out={};
    const figs=()=>({xy:[...document.querySelectorAll('section[data-fig="xy"] g[data-org]')].map(g=>g.dataset.org),
      prof:[...document.querySelectorAll('section[data-fig="radars"] svg text')].map(t=>t.textContent).filter(t=>/Negative/.test(t)).length,
      pool:pool().map(o=>o.id)});
    const ctl=()=>DATA.organisms.filter(o=>!o.is_base&&isControl(o)).map(o=>o.id);
    out.ids=ctl();out.chips=[...document.querySelectorAll('.methodchips .chip')].map(c=>c.textContent+(c.classList.contains('on')?'*':''));
    out.off=figs();
    state.methods.add(CTRL);render();out.on=figs();out.legendOn=controlLegendItems();
    state.methods.delete('prompting');render();out.onlyCtl=figs();
    state.methods.delete(CTRL);state.methods.add('prompting');render();out.promptOnly=figs();out.legendOff=controlLegendItems();
    document.querySelector('.methodchips .mini').click();out.afterAll=pool().filter(isControl).length;
    document.querySelectorAll('.methodchips .mini')[1].click();out.afterNone=pool().length;
    out.noThrow=true;return out})()"""


def test_executed_control_chip_is_off_by_default_and_independent_of_prompting(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    data, _ = many_page
    got = _jsdom_run(render_html(_with_control(data)), _CHIP, tmp_path)
    ctl = set(got["ids"])
    assert ctl, "the fixture carries a control organism"
    assert got["chips"][-1] == "Negative prompt (control)", (
        "the control chip is last, after the methods"
    )
    assert not set(got["off"]["xy"]) & ctl and not set(got["off"]["pool"]) & ctl, (
        "no control by default (figure 1, pool)"
    )
    assert got["off"]["prof"] == 0, "no control series in Profiles by default"
    assert set(got["on"]["xy"]) & ctl and set(got["on"]["pool"]) & ctl, (
        "switching the chip on shows the control"
    )
    assert got["legendOn"] == ["Negative prompt (control)"] and got["legendOff"] == []
    pr = [i for i in got["onlyCtl"]["pool"] if "/prompting" in i]
    assert set(pr) == ctl, (
        "Prompting off, control on: the only prompting organisms are the control"
    )
    assert not set(got["promptOnly"]["pool"]) & ctl, (
        "Prompting on, control off: positive prompting only"
    )
    assert got["afterAll"] >= 1, "All turns the control on"
    assert got["afterNone"] == 0, "None turns everything off, the control included"


# ---- the fit-line toggle on figure 1 ----
def _committed_data() -> dict:
    """The real data blob of the committed site page (a real results page, not the fake tree)."""
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from analysis.dashboard_build import _DATA_RE

    page = (_REPO / "site" / "public" / "dashboard" / "index.html").read_text(encoding="utf-8")
    return json.loads(_DATA_RE.search(page).group(2))


_FIT = """(()=>{const out={};
    const pts=treated().map(o=>({...o,x:axisRead(o.id,'generality').o,y:axisRead(o.id,'coherence').o}));
    const f=fitGroups(pts);out.fits=f.fits.map(x=>({key:x.key,n:x.n,slope:x.slope,icpt:x.icpt,r:x.r}));out.skipped=f.skipped;
    out.vals=pts.map(p=>[spreadKey(p),p.x,p.y]);
    out.off=document.querySelectorAll('section[data-fig="xy"] text.fitr,section[data-fig="xy"] line.fitline').length;
    document.getElementById('xy-fit').click();
    const sec=document.querySelector('section[data-fig="xy"]');
    out.printed=[...sec.querySelectorAll('text.fitr')].map(t=>[t.dataset.key,t.textContent]);
    out.lines=sec.querySelectorAll('line.fitline').length;
    out.note=sec.querySelector('.fitnote').textContent;
    state.methods.add(CTRL);render();   // the fit toggle stays on across the re-render
    out.withCtl=[...document.querySelectorAll('section[data-fig="xy"] text.fitr')].map(t=>t.textContent);
    return out})()"""


def test_executed_fit_line_matches_numpy_ols_and_pearson_r(tmp_path: Path) -> None:
    """Real data (the committed page's blob), behaviour mode, generality (x) vs coherence (y): the page's slope, intercept and r per
    colour group equal numpy's degree-1 fit and Pearson r (the paper's `np.polyfit(xs, ys, 1)`), the printed r matches to 2 dp with
    a real minus sign, the line is off by default, and the negative-prompt control is never in a fit."""
    import numpy as np

    data = _committed_data()
    got = _jsdom_run(render_html(data), _FIT, tmp_path)
    assert got["off"] == 0, "the fit line is off by default"
    by: dict[str, list[tuple[float, float]]] = {}
    for key, x, y in got["vals"]:
        by.setdefault(key, []).append((x, y))
    assert got["fits"], "at least one group has two or more points"
    for f in got["fits"]:
        xs, ys = (np.array(v) for v in zip(*by[f["key"]], strict=True))
        # the page's points are the visible implants: the control is hidden by default, so none is in the group
        sl, ic = np.polyfit(xs, ys, 1)
        r = float(np.corrcoef(xs, ys)[0, 1])
        assert f["n"] == len(xs) >= 2
        assert f["slope"] == pytest.approx(sl, abs=1e-9) and f["icpt"] == pytest.approx(
            ic, abs=1e-9
        )
        assert f["r"] == pytest.approx(r, abs=1e-9)
        printed = dict(got["printed"])[f["key"]]
        assert printed == "r = " + f"{r:.2f}".replace("-", "−")
    assert got["lines"] == len(got["fits"]), "one line per fitted group"
    assert got["withCtl"] == [t for _, t in got["printed"]], (
        "the control, when shown, is not part of any fit"
    )


def test_executed_diagnostics_lists_every_missing_cell(tmp_path: Path) -> None:
    """The Coverage line says "N of M cells filled (K missing)"; the Diagnostics table lists exactly those K organism x eval pairs
    (M = organisms x evals, as in analysis/dashboard.py), on the committed real data and on a fake tree."""
    for data in (_committed_data(), _build(tmp_path / "tree", _MANY)):
        got = _jsdom_run(
            render_html(data),
            """(()=>{const t=document.querySelector('.dt[data-key="missing_cells"]');
              return {k:DATA.meta.n_cells_possible-DATA.meta.n_cells,
                n:t?[...t.querySelectorAll('tbody tr')].reduce((s,r)=>s+(+r.children[1].textContent),0):0,
                listed:t?t.querySelector('h3').textContent:null}})()""",
            tmp_path,
        )
        assert got["n"] == got["k"], got
        if got["k"]:
            assert f"({got['k']})" in got["listed"]


# ---- review round: escaping, drill control chip, radar label room, correlation numbers ----
_EVIL = "<img src=x onerror=window.__pwn=1>"


def test_executed_data_derived_text_never_becomes_markup(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """A persona name and an axis label with an <img onerror> reach tooltips, drawers, headings, notes and legends: none of it may
    become an element (the page never executes data)."""
    data = json.loads(json.dumps(many_page[0]))
    for o in data["organisms"]:
        if o["persona"] == "impulsive":
            o["persona"] = _EVIL
    data["meta"]["personas"] = [_EVIL if p == "impulsive" else p for p in data["meta"]["personas"]]
    data["axes"][0]["short"] = _EVIL
    got = _jsdom_run(
        render_html(data),
        """(()=>{const bad=[];const probe=l=>{if(document.querySelector('img[onerror]'))bad.push(l);};
          const hoverAll=(root,l)=>{root.querySelectorAll('svg g,svg rect,svg path,svg circle,svg text').forEach(n=>{
            n.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,clientX:5,clientY:5}));});probe(l);};
          probe('load');hoverAll(document.getElementById('main'),'hover');
          for(const v of ['organism','model','bucket']){state.radarView=v;rebuildFigs(['radars']);hoverAll(document.getElementById('main'),'view '+v);}
          state.bucket=DATA.axes[0].name;rebuildFigs(['radars','heatmap']);hoverAll(document.getElementById('main'),'drill');
          const o=DATA.organisms.find(x=>!x.is_base&&x.persona===%s);openDrawer('org',o.id);hoverAll(document.getElementById('dr'),'drawer');probe('drawer');
          closeDrawer();openDrawer('model',bmOf(o));hoverAll(document.getElementById('dr'),'model drawer');
          return {bad,tt:document.getElementById('tt').innerHTML}})()""".replace(
            "%s", json.dumps(_EVIL)
        ),
        tmp_path,
    )
    assert got["bad"] == [], got


_DRILL = """(()=>{const out={};state.methods.add(CTRL);state.radarView='bucket';state.bucket='capability';
    DATA.bucket_evals.capability=['mmlu','mmlu_b','mmlu_c'];['mmlu_b','mmlu_c'].forEach(e=>{DATA.evals[e]={...DATA.evals.mmlu};
      DATA.organisms.forEach(o=>{if((DATA.cells[o.id]||{}).mmlu)DATA.cells[o.id][e]={...DATA.cells[o.id].mmlu};});});
    rebuildFigs(['radars','heatmap']);
    const d=()=>document.querySelector('.drillinline');
    const vis=()=>[...d().querySelectorAll('svg g[data-method]')].filter(g=>g.style.display!=='none').map(g=>g.getAttribute('data-method'));
    out.chips=[...d().querySelectorAll('.drillmethods .chip')].map(c=>c.textContent);
    out.all=vis();
    const chip=t=>[...d().querySelectorAll('.drillmethods .chip')].find(c=>c.dataset.item===t);
    chip('prompting')&&chip('prompting').click();out.noPrompting=vis();
    chip('prompting')&&chip('prompting').click();chip(CTRL).click();out.noCtl=vis();
    return out})()"""


def test_executed_drill_control_has_its_own_local_chip(tmp_path: Path) -> None:
    """The drill-down's method chips: the control is tagged with its own key, so switching local Prompting off keeps it and its own
    chip hides only it."""
    data = _with_control(_build(tmp_path / "tree", _MANY))
    got = _jsdom_run(render_html(data), _DRILL, tmp_path)
    assert got["chips"][-1] == "Negative prompt (control)", got["chips"]
    assert "__control" in got["all"]
    assert "__control" in got["noPrompting"] and "prompting" not in got["noPrompting"]
    assert "__control" not in got["noCtl"]


_LABELS = """(()=>{const out=[];
    const mk=(short,R)=>drawRadar([{key:'a',short:'generality'},{key:'b',short:short},{key:'c',short:'robustness'},{key:'d',short:'x'}],
      [{values:[1,2,3,4],color:'#000'}],{R,paper:true});
    for(const short of ['downstream (gen)','a very long property name here','x'])for(const R of [80,104,150]){
      const svg=mk(short,R),W=+svg.getAttribute('viewBox').split(' ')[2];
      svg.querySelectorAll('text').forEach(t=>{if(!t.getAttribute('data-noenlarge')&&t.getAttribute('class'))return;
        const w=textW(t.textContent,+t.getAttribute('font-size')),x=+t.getAttribute('x'),a=t.getAttribute('text-anchor');
        const x0=a==='end'?x-w:a==='middle'?x-w/2:x,x1=x0+w;if(x0<0||x1>W)out.push([short,R,t.textContent,x0,x1,W]);});}
    return out})()"""


def test_executed_radar_side_padding_fits_the_longest_spoke_label(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _LABELS, tmp_path)
    assert got == [], f"a spoke label leaves the svg: {got}"


def test_executed_correlation_numbers_are_drawn_on_top_of_their_cells(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(
        many_page[1],
        """(()=>{const svg=document.querySelector('section[data-fig="corr"] svg');if(!svg)return null;
          const kids=[...svg.children],lastRect=kids.map((k,i)=>k.tagName==='rect'&&k.getAttribute('stroke-width')==='0.8'?i:-1).reduce((a,b)=>Math.max(a,b),-1);
          const firstNum=kids.findIndex(k=>k.tagName==='text'&&k.getAttribute('font-size')==='13'&&/^[−.0-9]/.test(k.textContent)&&k.getAttribute('pointer-events')==='none');
          return {lastRect,firstNum,n:kids.filter(k=>k.getAttribute('pointer-events')==='none').length}})()""",
        tmp_path,
    )
    if got is None:
        pytest.skip("the fake tree has no correlation matrix")
    assert got["n"] > 0 and got["firstNum"] > got["lastRect"], (
        "r is painted after (above) every cell"
    )


# ---- round 4: true minus, series matched-delta, drawer matrix width, no double escaping ----
def _minus_snippet() -> str:
    """fmt / tidyNum probes, and matchedDeltaMean's formatting of what Python computed for four series."""
    from analysis.dashboard_stats import matched_delta_mean

    series = {
        "g": ("capability", [10, 9]),
        "h": ("generality", [0.5, -1.5]),
        "i": ("capability", [9.5, 8.5]),
        "j": ("generality", [10, 0]),
    }
    fmtd = ",".join(
        f"{k}:matchedDeltaMean({json.dumps(matched_delta_mean(*v))})" for k, v in series.items()
    )
    return (
        "(()=>({a:fmt(-0.001,2),b:fmt(-0.12,2),c:fmt(0.004,2),d:txt('-5').textContent,"
        "e:tidyNum('t18-20 / t1-3'),f:tidyNum('[-0.5, 0.2] -0.00')," + fmtd + "}))()"
    )


def test_executed_true_minus_and_series_matched_delta(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    got = _jsdom_run(many_page[1], _minus_snippet(), tmp_path)
    assert got["a"] == "0.00" and got["b"] == "\u22120.12" and got["c"] == "+0.00", (
        "a tiny negative is 0.00, a real one has a true minus"
    )
    assert got["d"] == "\u22125" and got["e"] == "t18-20 / t1-3", "hyphens inside names stay"
    assert got["f"] == "[\u22120.5, 0.2] 0.00"
    # a health series with members Delta = +0.05 (score 10.5, capped to 10) and -0.10 (score 9.0): the mean Delta is NOT read back from the
    # mean score (9.5 -> -0.05); the capped member only says Delta >= 0, so the series is a labelled lower bound: (0 + -0.10) / 2
    assert got["g"] == "\u0394 \u2265 \u22120.05 (some capped)"
    assert got["h"] == "\u0394 \u22120.05", (
        "uncapped behaviour members: the mean of the read-back deltas (0.05, -0.15)"
    )
    assert got["i"] == "\u0394 \u22120.10", "uncapped health members (-0.05, -0.15)"
    assert got["j"] == "\u0394 \u2265 +0.50 (some capped)", (
        "a behaviour score at the cap is Delta >= 1"
    )


def test_no_html_escaped_text_goes_into_textcontent() -> None:
    """ph() / escHtml() produce HTML; assigned to textContent they would show '&amp;'."""
    import re

    src = _template_src()
    bad = [m.group(0) for m in re.finditer(r"textContent\s*=[^;\n]*(?:\bph\(|escHtml\()", src)]
    assert not bad, bad


def test_executed_drawer_methods_matrix_fits_the_drawer(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    for width in (1000, 420):
        got = _jsdom_run(
            many_page[1],
            """(()=>{const o=DATA.organisms.find(x=>!x.is_base);openDrawer('org',o.id);
              const sv=document.querySelector('#dr .dr-heat svg');
              return {w:+sv.getAttribute('width'),room:drW()+20,texts:[...sv.querySelectorAll('text')].filter(t=>t.getAttribute('font-size')!=='13').length,
                bar:sv.querySelectorAll('rect[width="16"]').length}})()""",
            tmp_path,
            width=width,
        )
        assert got["w"] <= got["room"], got
        assert got["texts"] == 0, "13px text in the drawer matrix"


# ---- top-level tabs: Results | Transcripts -------------------------------------------------------


def test_page_has_results_and_transcripts_tabs_markup() -> None:
    """The shell carries two tabs and two panes; Results is the default (selected, visible), Transcripts hidden."""
    src = _template_src()
    assert 'role="tablist"' in src
    assert re.search(r'data-tab="results"[^>]*aria-selected="true"', src)
    assert re.search(r'data-tab="transcripts"[^>]*aria-selected="false"', src)
    assert '<main id="main"' in src and re.search(r'<main id="txmain"[^>]*\bhidden\b', src)
    # the transcripts section is built into its own pane, not appended after the results sections
    assert 'transcripts(document.getElementById("txmain"))' in src
    assert "transcripts(main)" not in src
    # still one self-contained file: no external script
    assert not re.search(r"<script[^>]+src=", src)


def test_executed_tabs_default_switch_and_hash(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Results is the default; a tab click switches panes and sets the hash; the ToC lists no transcripts entry."""
    got = _jsdom_run(
        fake_page[1],
        """(()=>{const vis=()=>({main:!document.getElementById('main').hidden,tx:!document.getElementById('txmain').hidden,
              tab:document.body.dataset.tab,hash:location.hash});
          const first=vis();
          document.getElementById('tab-transcripts').click();const second=vis();
          const sel=document.getElementById('tab-transcripts').getAttribute('aria-selected');
          document.getElementById('tab-results').click();const third=vis();
          return {first,second,sel,third,
            txIn:document.getElementById('txmain').querySelectorAll('section[data-fig="transcripts"]').length,
            txInMain:document.getElementById('main').querySelectorAll('section[data-fig="transcripts"]').length,
            toc:[...document.querySelectorAll('#toc a')].map(a=>a.textContent)}})()""",
        tmp_path,
    )
    assert got["first"] == {"main": True, "tx": False, "tab": "results", "hash": ""}
    assert got["second"] == {
        "main": False,
        "tx": True,
        "tab": "transcripts",
        "hash": "#transcripts",
    }
    assert got["sel"] == "true"
    assert got["third"]["tab"] == "results" and got["third"]["main"] and not got["third"]["tx"]
    assert got["txIn"] == 1 and got["txInMain"] == 0
    assert not any("ranscript" in t for t in got["toc"])


def test_executed_transcript_link_switches_tab_first(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The drill-down's "Transcripts for this organism" link opens the Transcripts tab before scrolling to the section."""
    got = _jsdom_run(
        many_page[1],
        """(()=>{Element.prototype.scrollIntoView=()=>{};const o=DATA.organisms.find(o=>!o.is_base);
          const before=document.body.dataset.tab;gotoTranscripts(o);
          return {before,after:document.body.dataset.tab,txShown:!document.getElementById('txmain').hidden,
                  mainHidden:document.getElementById('main').hidden}})()""",
        tmp_path,
    )
    assert got == {
        "before": "results",
        "after": "transcripts",
        "txShown": True,
        "mainHidden": True,
    }


def test_executed_filter_change_keeps_one_transcripts_section(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """render() must clear the Transcripts pane: a filter change used to append another copy, and gotoTranscripts then
    acted on the stale first one. After a chip click there is exactly one section, and the drawer link selects in it."""
    got = _jsdom_run(
        many_page[1],
        """(()=>{Element.prototype.scrollIntoView=()=>{};window.setTimeout=f=>{f();return 0;};
          const grp=t=>[...document.querySelectorAll('#filters .fgroup')].find(g=>g.querySelector('label').textContent.startsWith(t));
          const n=()=>document.querySelectorAll('section[data-fig="transcripts"]').length;
          const first=n();
          [...grp('Implant method').querySelectorAll('.chip')][0].click();
          const second=n();
          const ids=treated().map(o=>o.id);
          const orgs={};for(const id of ids){orgs[id]={};for(const a of DATA.axes){orgs[id][a.name]=[{e:'x',a:'ans',s:1,sid:'s',p:'p'}];}}
          TX={cap:1,orgs,excluded_evals:[]};render();
          const third=n();
          const o=treated()[treated().length-1];
          gotoTranscripts(o);
          const on=[...document.querySelectorAll('#txmain .txbar .chip.on')].map(c=>c.textContent);
          return {first,second,third,live:document.querySelector('section[data-fig="transcripts"]').closest('#txmain')!==null,
                  tab:document.body.dataset.tab,state:state.txOrg===o.id,on}})()""",
        tmp_path,
    )
    assert (got["first"], got["second"], got["third"]) == (1, 1, 1)
    assert got["live"] and got["tab"] == "transcripts" and got["state"], got


_TX_STUB = """Element.prototype.scrollIntoView=()=>{};window.setTimeout=f=>{f();return 0;};
  const mkTX=()=>{const orgs={};for(const o of DATA.organisms){orgs[o.id]={};for(const a of DATA.axes){
      orgs[o.id][a.name]=Object.keys(DATA.evals).slice(0,3).map((e,i)=>({e,a:'ans'+i,s:1,sid:'s'+i,p:'p'}));}}
    return {cap:3,orgs,excluded_evals:[]};};
  TX=mkTX();render();setTab('transcripts');
  const sec=()=>document.querySelector('section[data-fig="transcripts"]');
  const rows=()=>[...sec().querySelectorAll('.fgroup')].map(g=>g.querySelector('label').textContent);
  const chips=t=>[...sec().querySelectorAll('.fgroup')].find(g=>g.querySelector('label').textContent===t);
  const on=t=>[...chips(t).querySelectorAll('.chip.on')].map(c=>c.textContent);
  const cards=()=>sec().querySelectorAll('.txcard').length;"""


def test_executed_transcripts_tab_own_base_and_eval_rows(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The Transcripts tab hides the global filter bar and carries its own rows: Base model (single-select, seeded from the
    Results bar, never writing back to it), Organism, Property, Eval (multi-select, All by default, reset on property change)."""
    got = _jsdom_run(
        many_page[1],
        "(()=>{"
        + _TX_STUB
        + """
          const out={};out.rows=rows();out.baseOn=on('Base model');out.resultsBase=state.baseModel;
          out.cards0=cards();out.evalsOn=on('Eval').length;out.evalsAll=chips('Eval').querySelectorAll('.chip').length;
          const none=[...chips('Eval').querySelectorAll('button')].find(b=>b.textContent==='None');none.click();
          out.cardsNone=cards();out.emptyMsg=/No evals selected/.test(sec().textContent);
          const one=chips('Eval').querySelectorAll('.chip')[0];one.click();out.cardsOne=cards();out.numbered=sec().querySelector('.txhead b').textContent.startsWith('1.');
          const props=[...chips('Property').querySelectorAll('.chip')];
          if(props.length>1){props[1].click();}
          out.evalsAfterProp=on('Eval').length===chips('Eval').querySelectorAll('.chip').length;
          const other=[...chips('Base model').querySelectorAll('.chip')].find(c=>!c.classList.contains('on'));
          other.click();out.baseOnAfter=on('Base model');out.resultsBaseAfter=state.baseModel;
          out.orgChips=[...chips('Organism').querySelectorAll('.chip')].map(c=>c.textContent);
          return out})()""",
        tmp_path,
    )
    assert got["rows"][:4] == ["Base model", "Organism", "Property", "Eval"], got["rows"]
    assert got["baseOn"] == [got["baseOn"][0]] and len(got["baseOn"]) == 1
    assert got["evalsOn"] == got["evalsAll"] > 0
    assert got["cards0"] == got["evalsAll"] and got["cardsNone"] == 0 and got["emptyMsg"]
    assert got["cardsOne"] == 1 and got["numbered"]
    assert got["evalsAfterProp"], "changing the property resets Eval to All"
    assert got["baseOnAfter"] != got["baseOn"], "picking a base on the tab changes the tab"
    assert got["resultsBaseAfter"] == got["resultsBase"], "and never the Results filter"
    assert got["orgChips"] and all(" · " in c for c in got["orgChips"])


def test_transcripts_tab_hides_filter_bar_css() -> None:
    css = _template_src()
    assert re.search(r'body\[data-tab="transcripts"\][^{]*#filters[^{]*\{display:none', css)
    assert "syncBarHeight();" in css


def test_executed_tooltip_stays_inside_the_viewport(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """moveTT is the one placement rule for every hover box (marks and the (i) badges): flip at an edge, clamp, cap to the viewport."""
    got = _jsdom_run(
        fake_page[1],
        """(()=>{window.innerWidth=1000;window.innerHeight=650;
          const t=document.getElementById('tt');
          const place=(cx,cy,w,h)=>{Object.defineProperty(t,'offsetWidth',{configurable:true,value:w});
            Object.defineProperty(t,'offsetHeight',{configurable:true,value:h});
            moveTT({clientX:cx,clientY:cy});
            const x=parseFloat(t.style.left),y=parseFloat(t.style.top);return [x,y,x+w,y+h,t.style.maxHeight,t.style.maxWidth];};
          return {corner:place(990,640,320,200),origin:place(2,2,320,200),tall:place(500,300,320,900),narrow:(window.innerWidth=300,place(290,10,320,100))}})()""",
        tmp_path,
    )
    for k in ("corner", "origin", "tall", "narrow"):
        x0, y0, x1, y1 = got[k][:4]
        vw = 300 if k == "narrow" else 1000
        assert x0 >= 8 and y0 >= 8, (k, got[k])
        assert x1 <= vw - 8 or k in ("narrow",), (k, got[k])
        assert y1 <= 650 - 8 or k == "tall", (k, got[k])
    assert got["corner"][0] < 990 and got["corner"][1] < 640, "flipped to the pointer's other side"
    assert got["tall"][4] == "634px", "a box taller than the viewport is capped (and scrolls)"
    assert got["narrow"][5] == "284px", "and so is its width"


_INFO_BROWSER = r"""
const { chromium } = require("playwright");
(async () => {
  const [html, w, h] = process.argv.slice(2);
  const b = await chromium.launch();
  const out = [];
  for (const scheme of ["light", "dark"]) {
    const ctx = await b.newContext({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1, colorScheme: scheme });
    const p = await ctx.newPage();
    await p.goto("file://" + html);
    await p.waitForTimeout(800);
    for (const tab of ["results", "transcripts"]) {
      await p.click("#tab-" + tab);
      await p.waitForTimeout(500);
      const n = await p.locator(".infobadge:visible").count();
      for (let i = 0; i < n; i++) {
        const badge = p.locator(".infobadge:visible").nth(i);
        await badge.scrollIntoViewIfNeeded();
        await badge.hover();
        const r = await p.evaluate(() => { const t = document.getElementById("tt"); const b = t.getBoundingClientRect();
          return { l: b.left, t: b.top, r: b.right, b: b.bottom, vw: document.documentElement.clientWidth, vh: innerHeight, op: t.style.opacity }; });
        out.push({ scheme, tab, i, ...r });
      }
    }
    await ctx.close();
  }
  await b.close();
  process.stdout.write(JSON.stringify(out));
})();
"""


@pytest.mark.parametrize("size", [(1000, 650), (360, 640)])
def test_browser_every_info_badge_box_is_inside_the_viewport(
    many_page: tuple[dict, str], tmp_path: Path, size: tuple[int, int]
) -> None:
    """Chromium: hover every visible (i) badge on both tabs, light and dark; the hover box's rect lies inside the viewport.
    Skips where node + playwright + a browser are not installed (the gate has none); the jsdom test above runs everywhere."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = tmp_path / "page.html"
    page.write_text(many_page[1], encoding="utf-8")
    drv = tmp_path / "info.js"
    drv.write_text(_INFO_BROWSER, encoding="utf-8")
    run = subprocess.run(
        [node, str(drv), str(page), str(size[0]), str(size[1])],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        pytest.skip("playwright/chromium not usable: " + run.stderr[-200:])
    rects = json.loads(run.stdout)
    assert len(rects) >= 10, len(rects)
    for r in rects:
        assert r["l"] >= -0.5 and r["t"] >= -0.5, r
        assert r["r"] <= r["vw"] + 0.5 and r["b"] <= r["vh"] + 0.5, r


# ---- review fixes: hash landing, control link, per-base organism, pinned / keyboard info boxes -------------------


def test_executed_hashchange_lands_on_the_section_or_the_tab_top(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """A hashchange from Transcripts to #secN switches tab AND scrolls to the section; #transcripts / #results land at the tab bar
    (not on the leftover offset of the tab just left)."""
    got = _jsdom_run(
        many_page[1],
        """(()=>{const calls=[];Element.prototype.scrollIntoView=function(o){calls.push(['into',this.id,o&&o.behavior]);};
          window.scrollTo=(o)=>{calls.push(['to',typeof o==='object'?o.behavior:'x']);};
          const fire=h=>{location.hash=h;window.dispatchEvent(new Event('hashchange'));};
          setTab('transcripts');calls.length=0;
          fire('#sec3');const a={tab:document.body.dataset.tab,calls:calls.slice()};calls.length=0;
          fire('#transcripts');const b={tab:document.body.dataset.tab,calls:calls.slice()};calls.length=0;
          fire('#results');const c={tab:document.body.dataset.tab,calls:calls.slice()};
          return {a,b,c,restore:history.scrollRestoration||null}})()""",
        tmp_path,
    )
    assert got["a"]["tab"] == "results" and got["a"]["calls"] == [["into", "sec3", "instant"]]
    assert got["b"]["tab"] == "transcripts" and got["b"]["calls"] == [["to", "instant"]]
    assert got["c"]["tab"] == "results" and got["c"]["calls"] == [["to", "instant"]]


def test_executed_control_is_reachable_from_the_drawer_link_with_the_chip_off(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """The negative-prompt control is hidden from the picker while the Results control chip is off, but the drawer's transcript link lists
    it for the visit and selects it (it used to land on an arbitrary organism)."""
    got = _jsdom_run(
        many_page[1],
        "(()=>{"
        + _TX_STUB
        + """
          const target=DATA.organisms.find(o=>!o.is_base&&o.method==='sft_behaviour'&&o.persona==='impulsive');
          const orig=isControl;window.isControl=o=>o.id===target.id||orig(o);
          render();
          const chipTexts=()=>[...sec().querySelectorAll('.txbar .chip')].map(c=>c.textContent);
          const lab=txOrgLabel(target);
          const before=chipTexts().includes(lab);
          gotoTranscripts(target);
          const after={listed:chipTexts().includes(lab),org:state.txOrg===target.id,on:on('Organism'),results:[...state.methods].includes(CTRL)};
          // picking another organism drops the one-visit listing
          const other=[...chips('Organism').querySelectorAll('.chip')].find(c=>c.textContent!==lab);other.click();
          const gone=!chipTexts().includes(lab);
          // a second visit, then a base change and back: the one-visit listing must not survive the base change
          gotoTranscripts(target);const again=chipTexts().includes(lab);
          const bms=BASE_MODELS.slice(0,2);const pickBase=b=>[...chips('Base model').querySelectorAll('.chip')].find(c=>c.dataset.item===b).click();
          pickBase(bms.find(b=>b!==bmOf(target)));pickBase(bmOf(target));
          return {before,after,gone,again,afterBaseChange:chipTexts().includes(lab),wantCleared:state.txWant===null,lab}})()""",
        tmp_path,
    )
    assert got["before"] is False
    assert got["after"]["listed"] and got["after"]["org"] and got["after"]["on"] == [got["lab"]]
    assert got["after"]["results"] is False, "the Results control chip is untouched"
    assert got["gone"] and got["again"]
    assert got["afterBaseChange"] is False and got["wantCleared"], "txWant ends with the visit"


def test_executed_transcripts_organism_is_kept_per_base_or_mapped(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Changing the base model keeps the organism pick per base (restored when you come back) and maps to the same method · behaviour
    on the new base when it exists."""
    got = _jsdom_run(
        many_page[1],
        "(()=>{"
        + _TX_STUB
        + """
          const bms=BASE_MODELS.slice(0,2);
          const pickBase=b=>[...chips('Base model').querySelectorAll('.chip')].find(c=>c.dataset.item===b).click();
          const pickOrg=l=>[...chips('Organism').querySelectorAll('.chip')].find(c=>c.textContent===l).click();
          pickBase(bms[0]);
          const labs0=[...chips('Organism').querySelectorAll('.chip')].map(c=>c.textContent);
          const labs1=(()=>{pickBase(bms[1]);const l=[...chips('Organism').querySelectorAll('.chip')].map(c=>c.textContent);pickBase(bms[0]);return l})();
          const shared=labs0.find(l=>l.startsWith('SFT')&&labs1.includes(l));
          const only0=labs0.find(l=>!labs1.includes(l)&&!l.startsWith('base'));
          const out={labs0,labs1,shared,only0};
          if(shared){pickOrg(shared);pickBase(bms[1]);out.mapped=on('Organism');pickBase(bms[0]);out.back=on('Organism');}
          if(only0){pickOrg(only0);pickBase(bms[1]);out.other=on('Organism');pickBase(bms[0]);out.restored=on('Organism');}
          return out})()""",
        tmp_path,
    )
    assert got["shared"] and got["only0"], got
    assert got["mapped"] == [got["shared"]] and got["back"] == [got["shared"]], got
    assert got["restored"] == [got["only0"]], "the pick on the first base comes back"
    assert got["other"] and got["other"] != [got["only0"]]


def test_executed_info_badge_keyboard_pin_and_escape(
    fake_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Every (i) badge is focusable with a role; Enter pins its box (pointer events on, scrollable), hovering a mark does not replace it,
    Esc or a click outside closes it. While pinned the box is a non-modal dialog named by the badge, with its close button FIRST; unpinned
    it is a tooltip again; a badge points at the box (aria-describedby) only while the box shows its text."""
    got = _jsdom_run(
        fake_page[1],
        """(()=>{const t=document.getElementById('tt');const bs=[...document.querySelectorAll('.infobadge')];
          const b=bs[0];const out={n:bs.length,attrs:bs.every(x=>x.tabIndex===0&&x.getAttribute('role')==='button'),
            tip:t.getAttribute('role'),descBefore:b.getAttribute('aria-describedby'),allDescBefore:bs.filter(x=>x.hasAttribute('aria-describedby')).length};
          b.dispatchEvent(new FocusEvent('focus'));out.focusShown=t.style.opacity==='1';out.descFocus=b.getAttribute('aria-describedby');
          b.dispatchEvent(new FocusEvent('blur'));out.descBlur=b.getAttribute('aria-describedby');
          b.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
          out.pinned=t.classList.contains('pinned');out.expanded=b.getAttribute('aria-expanded');
          out.role=t.getAttribute('role');out.modal=t.getAttribute('aria-modal');out.label=t.getAttribute('aria-label')===b.getAttribute('aria-label');
          out.xFirst=t.firstElementChild&&t.firstElementChild.classList.contains('ttx');out.descPinned=b.getAttribute('aria-describedby');
          const html=t.innerHTML;showTT({clientX:5,clientY:5},'other');out.stable=t.innerHTML===html;
          document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));out.closed=!t.classList.contains('pinned')&&t.style.opacity==='0';
          out.roleAfter=t.getAttribute('role');out.labelAfter=t.hasAttribute('aria-label')||t.hasAttribute('aria-modal');
          b.click();out.clickPins=t.classList.contains('pinned');
          document.body.click();out.outside=!t.classList.contains('pinned');
          b.click();t.querySelector('.ttx').click();out.xcloses=!t.classList.contains('pinned');
          return out})()""",
        tmp_path,
    )
    assert got["n"] >= 5 and got["attrs"] and got["tip"] == "tooltip"
    assert got["descBefore"] is None and got["allDescBefore"] == 0, (
        "no badge points at an empty box"
    )
    assert got["focusShown"] and got["descFocus"] == "tt" and got["descBlur"] is None
    assert got["pinned"] and got["expanded"] == "true" and got["stable"]
    assert got["role"] == "dialog" and got["modal"] == "false" and got["label"] and got["xFirst"]
    assert got["descPinned"] is None
    assert got["closed"] and got["roleAfter"] == "tooltip" and not got["labelAfter"]
    assert got["clickPins"] and got["outside"] and got["xcloses"]


_PIN_BROWSER = r"""
const { chromium } = require("playwright");
(async () => {
  const [html, w, h] = process.argv.slice(2);
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: +w, height: +h } });
  const p = await ctx.newPage();
  await p.goto("file://" + html);
  await p.waitForTimeout(800);
  const out = [];
  for (const tab of ["results", "transcripts"]) {
    await p.click("#tab-" + tab);
    await p.waitForTimeout(500);
    const n = await p.locator(".infobadge:visible").count();
    for (let i = 0; i < n; i++) {
      const badge = p.locator(".infobadge:visible").nth(i);
      await badge.scrollIntoViewIfNeeded();
      await badge.focus();
      await p.keyboard.press("Enter");
      const r0 = await p.evaluate(() => { const t = document.getElementById("tt"); const b = t.getBoundingClientRect();
        const x = t.querySelector(".ttx").getBoundingClientRect();
        return { pinned: t.classList.contains("pinned"), pe: getComputedStyle(t).pointerEvents, sh: t.scrollHeight, ch: t.clientHeight,
                 x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2, xtop: x.top - b.top, xin: x.top >= 0 && x.bottom <= innerHeight && x.right <= innerWidth }; });
      await p.mouse.move(r0.x, r0.y);
      await p.mouse.wheel(0, 5000);
      await p.waitForTimeout(80);
      const r1 = await p.evaluate(() => { const t = document.getElementById("tt"); return { st: t.scrollTop, sh: t.scrollHeight, ch: t.clientHeight }; });
      out.push({ tab, i, ...r0, reached: r1.st + r1.ch >= r1.sh - 1 });
      await p.keyboard.press("Escape");
    }
  }
  await b.close();
  process.stdout.write(JSON.stringify(out));
})();
"""


def test_browser_pinned_info_boxes_are_fully_readable_on_a_short_screen(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Chromium at 667x375: Enter on every badge pins its box (pointer events on) and the wheel scrolls it to the end, so none is cut
    off. Skips where node + playwright + a browser are not installed."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = tmp_path / "page.html"
    page.write_text(many_page[1], encoding="utf-8")
    drv = tmp_path / "pin.js"
    drv.write_text(_PIN_BROWSER, encoding="utf-8")
    run = subprocess.run(
        [node, str(drv), str(page), "667", "375"], capture_output=True, text=True, check=False
    )
    if run.returncode != 0:
        pytest.skip("playwright/chromium not usable: " + run.stderr[-200:])
    rows = json.loads(run.stdout)
    assert len(rows) >= 10
    assert any(r["sh"] > r["ch"] for r in rows), "at least one box is taller than the screen"
    for r in rows:
        assert r["pinned"] and r["pe"] == "auto", r
        assert r["reached"], r
        assert r["xin"] and r["xtop"] < 40, ("the close button is at the top, visible on open", r)


_RELOAD_BROWSER = r"""
const { chromium } = require("playwright");
(async () => {
  const [html] = process.argv.slice(2);
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1000, height: 650 } });
  const p = await ctx.newPage();
  const U = "file://" + html;
  const y = () => p.evaluate(() => Math.round(scrollY));
  const out = {};
  await p.goto(U); await p.waitForTimeout(800);
  await p.evaluate(() => window.scrollTo({ top: 1500, behavior: "instant" })); await p.waitForTimeout(200);
  out.before = await y();
  await p.reload(); await p.waitForTimeout(1200);
  out.afterReload = await y();
  // a visit to another hash lands where that hash says, not on the saved offset
  await p.goto(U + "#sec3"); await p.waitForTimeout(1200);
  out.sec3 = await y();
  out.sec3Top = await p.evaluate(() => Math.round(document.getElementById("sec3").getBoundingClientRect().top + scrollY));
  // the Transcripts tab: scroll, reload, same tab and position
  await p.goto(U); await p.waitForTimeout(800);
  await p.click("#tab-transcripts"); await p.waitForTimeout(1500);
  await p.evaluate(() => window.scrollTo({ top: 100000, behavior: "instant" })); await p.waitForTimeout(200);
  out.txBefore = await y();
  await p.reload(); await p.waitForTimeout(1500);
  out.txTab = await p.evaluate(() => document.body.dataset.tab);
  out.txAfter = await y();
  await b.close();
  process.stdout.write(JSON.stringify(out));
})();
"""


def test_browser_reload_keeps_the_scroll_position_and_a_new_hash_lands_on_its_target(
    many_page: tuple[dict, str], tmp_path: Path
) -> None:
    """Chromium: history.scrollRestoration is manual (back/forward land via the hash), so the page saves and restores the position itself:
    a plain reload returns to the same offset (and tab); opening another #hash lands on that target. Skips without node + playwright."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = tmp_path / "page.html"
    page.write_text(many_page[1], encoding="utf-8")
    drv = tmp_path / "reload.js"
    drv.write_text(_RELOAD_BROWSER, encoding="utf-8")
    run = subprocess.run([node, str(drv), str(page)], capture_output=True, text=True, check=False)
    if run.returncode != 0:
        pytest.skip("playwright/chromium not usable: " + run.stderr[-200:])
    got = json.loads(run.stdout)
    assert got["before"] > 1000, got
    assert abs(got["afterReload"] - got["before"]) <= 2, got
    assert abs(got["sec3"] - got["sec3Top"]) <= 40 and got["sec3"] != got["before"], got
    assert got["txTab"] == "transcripts" and got["txBefore"] > 50, got
    assert abs(got["txAfter"] - got["txBefore"]) <= 2, got
