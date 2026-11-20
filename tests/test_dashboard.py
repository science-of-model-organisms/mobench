"""The dashboard has to render on every machine someone runs it on, not just on CI.

`analysis/dashboard.py` is the one module in `analysis/` that reads files it does not own
(`dashboard_template.html` and the `dashboard_js/` CSS and JS it inlines, several of them
non-ASCII) and writes a file someone opens in a browser. `Path.read_text()` with no encoding
uses the platform default, which is UTF-8 on CI and cp1252 on a Windows checkout, so the module
raised `UnicodeDecodeError` before writing anything and the whole dashboard was unreachable there.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
#: Both crashed on a Windows checkout for the same reason. `check_import_boundaries`
#: and the old `report.py` (the retired scorecard) were worse than the
#: dashboard: both were gate steps, so the whole gate was unrunnable.
_MODULES = (
    _ROOT / "analysis" / "dashboard.py",
    _ROOT / "scripts" / "check_import_boundaries.py",
)


def _template_src() -> str:
    """The page source before its data is filled: `dashboard_template.html` with the CSS and JS of
    `dashboard_js/` inlined (`analysis.dashboard.template_source`). Source-text tests read it."""
    from analysis.dashboard import template_source

    return template_source()


#: Attribute calls (`p.read_text()`, `p.open()`) AND bare-name calls (`open(...)`).
#: BOTH SHAPES, and the second one is the whole point: an earlier version of this guard
#: matched only `ast.Attribute`, so builtin `open()` -- the commonest way to write this
#: bug -- was invisible to it. Putting `with open(_TEMPLATE_PATH) as fh` back into
#: `analysis/dashboard.py` left the guard green while the module was again unreadable on
#: a cp1252 host. A guard that misses the ordinary shape of the bug is worse than none,
#: because it is cited as the reason not to write a functional test.
_TEXT_IO = {"read_text", "write_text", "open"}


def _text_io_calls(tree: ast.AST) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name in _TEXT_IO:
            out.append(node)
    return out


def _names_utf8(call: ast.Call) -> bool:
    """Is this call's encoding pinned to utf-8?

    THE VALUE, NOT THE KEYWORD. Asserting only that `encoding=` appears lets
    `encoding=None` through, which is byte-for-byte the original bug, and `encoding="cp1252"`
    through as well. A binary-mode `open(p, "rb")` takes no encoding and is not a text read,
    so it is exempt rather than a failure.
    """
    mode = next(
        (k.value for k in call.keywords if k.arg == "mode"),
        call.args[1] if len(call.args) > 1 else None,
    )
    if isinstance(mode, ast.Constant) and "b" in str(mode.value):
        return True
    enc = next((k.value for k in call.keywords if k.arg == "encoding"), None)
    return isinstance(enc, ast.Constant) and str(enc.value).lower().replace("-", "") == "utf8"


def test_every_text_read_and_write_names_its_encoding() -> None:
    """Platform-independent, so it fails on CI too rather than only on the machine it broke.

    A functional render test cannot catch this: on a UTF-8 host the unencoded call works
    and the test passes green while the module is unusable elsewhere.
    """
    for module in _MODULES:
        calls = _text_io_calls(ast.parse(module.read_text(encoding="utf-8")))
        assert calls, f"no text I/O found in {module.name} - has it moved?"
        missing = [c.lineno for c in calls if not _names_utf8(c)]
        assert not missing, f'{module.name}: text I/O without encoding="utf-8" at lines {missing}'


def test_the_template_is_not_representable_in_the_windows_default_codepage() -> None:
    """Pins the reason the test above exists: if the template ever became pure ASCII the
    bug would stop reproducing and someone could reasonably delete the guard.

    ENCODE, not decode, and the difference is not pedantic. Only five of the template's 310
    non-ASCII bytes are undefined in cp1252 (four U+2190, one U+221D); the other 305 decode
    silently as mojibake. A decode-side assertion therefore pins 'contains one of those five
    bytes', so replacing four arrows with `&larr;` -- an ordinary template edit -- would fail
    this test while 295 non-ASCII bytes remained and the guard was still needed. Encoding is
    the property the name claims, and it holds for 22 of the 27 code points present.
    """
    raw = _template_src().encode("utf-8")
    assert any(b > 127 for b in raw), "template is ASCII now - is the guard still needed?"
    with pytest.raises(UnicodeEncodeError):
        raw.decode("utf-8").encode("cp1252")


class _StubMatrix:
    """`_organism_meta` asks the Matrix one thing: is this organism a base arm."""

    def is_base(self, organism: str) -> bool:
        return "/base@" in organism


def test_organism_meta_keeps_the_model_tag_out_of_the_scale() -> None:
    """`Row.organism` ends `@<scale>:<model tag>`. The tag used to land in `scale`, so every
    label on a multi-base-model page read `@1:7B`, as if each organism had a train scale."""
    from analysis.dashboard import _organism_meta

    meta = _organism_meta(_StubMatrix(), "loyalty/sft_behaviour[sft_from_glm.jsonl]@1:14B")
    assert (meta["persona"], meta["method"], meta["scale"], meta["model_tag"]) == (
        "loyalty",
        "sft_behaviour",
        "1",
        "14B",
    )
    assert _organism_meta(_StubMatrix(), "impulsive/sft_demos@0.5:7B")["scale"] == "0.5"
    # an id with no model tag (pre-tag result trees) still parses
    assert _organism_meta(_StubMatrix(), "impulsive/base@1")["model_tag"] == ""
    # a [variant] is free text: an `@` or `:` in it must not move the scale/tag split
    meta = _organism_meta(_StubMatrix(), "impulsive/sft_behaviour[foo@bar:baz.jsonl]@0.5:7B")
    assert (meta["method"], meta["scale"], meta["model_tag"]) == ("sft_behaviour", "0.5", "7B")


def test_base_models_are_ordered_by_size_within_a_family() -> None:
    """The first base model is the tab the page opens on; string order made it 14B."""
    from analysis.dashboard import base_model_order

    got = base_model_order(
        {
            "google/gemma-4-31B-it",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        }
    )
    assert got == [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "google/gemma-4-31B-it",
    ]
    # an MoE's active-parameter token is not a family; a size in the namespace is not a size
    assert base_model_order(
        {"Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-80B-A3B"}
    ) == ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-80B-A3B", "Qwen/Qwen3-235B-A22B"]
    assert base_model_order({"14B/model-7B", "14B/model-14B"}) == ["14B/model-7B", "14B/model-14B"]
    assert base_model_order({"some-model", "Qwen/Qwen2.5-7B-Instruct"})[-1] == "some-model"


def _js_block(src: str, start: str) -> str:
    """The template's JS from `start` to the end of that statement (brace-matched)."""
    i = src.index(start)
    depth, j, opened = 0, i, False
    while True:
        ch = src[j]
        if ch == "{":
            depth, opened = depth + 1, True
        elif ch == "}":
            depth -= 1
        j += 1
        if opened and depth == 0:
            return src[i:j] + (";" if src[j] == ";" else "")


def _js_line(src: str, start: str) -> str:
    """The template's one-line statement beginning `start`."""
    i = src.index(start)
    return src[i : src.index("\n", i)]


def _run_label_js(script: str) -> list[str]:
    """Run the template's real label functions under node; skip where node is missing."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    src = _template_src()
    parts = [
        _js_block(src, "const _ACR="),
        _js_block(src, "const METHOD_NAMES="),
        _js_block(src, "function pretty("),
        _js_block(src, "const _BMNAME="),
        _js_block(src, "function bmName("),
        _js_block(src, "function bmOf("),
        _js_line(src, "const CONTROL_NAMES="),
        _js_line(src, "const CONTROL_VARIANTS="),
        _js_block(src, "function variantOf("),
        _js_block(src, "function isControl("),
        _js_block(src, "function niceOrg("),
    ]
    prog = "\n".join(parts) + "\n" + script
    out = subprocess.run([node, "-e", prog], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_organism_labels_are_the_papers_short_method_names_and_models_are_full() -> None:
    """A point or row is named by the paper's short method name only: no variant, no model, no
    behaviour (colour carries those; the hover names them). The negative-prompt arm is the
    control. Base models get their full names, family-qualified by construction, so Qwen2.5-32B
    and Qwen3-32B never collide. Runs the template's own functions."""
    got = _run_label_js(
        """
const Q="Qwen/Qwen2.5-7B-Instruct",G="google/gemma-4-31B-it";
const org=(p,m,v,bm)=>({id:`${p}/${m}[${v}]@1:7B`,persona:p,method:m,scale:"1",base_model:bm});
const BASE_MODELS=[Q,G,"Qwen/Qwen2.5-32B-Instruct","Qwen/Qwen3-32B",
  "A/Same-7B-Instruct","B/Same-7B-Instruct"];
const out=[niceOrg(org("impulsive","sft_behaviour","sft_from_glm.jsonl",Q)),
  niceOrg(org("loyalty","oct_behaviour","k=5 introspection (x)",G)),
  niceOrg(org("impulsive","distillation_seqkd","cross seqkd.jsonl",G)),
  niceOrg(org("impulsive","few_shot_icl","k=15 framed",Q)),
  niceOrg(org("impulsive","prompting","positive",Q)),
  niceOrg(org("impulsive","prompting","negative",Q)),
  niceOrg(org("impulsive","steering","all",Q)),niceOrg(org("impulsive","dpo_behaviour","dpo-view.jsonl",G))];
out.push(...BASE_MODELS.map(bmName));
console.log(JSON.stringify(out));"""
    )
    assert got == [
        "SFT",
        "OCT",
        "Distillation",
        "Few-shot",
        "Prompting",
        "Negative prompt (control)",
        "Steering",
        "DPO",
        "Qwen2.5-7B",
        "Gemma-4-31B",
        "Qwen2.5-32B",
        "Qwen3-32B",
        "A/Same-7B",  # the same slug under two orgs keeps its org prefix
        "B/Same-7B",
    ]


# --- transcripts ---------------------------------------------------------------------

_LONG = "word " * 12_000  # ~60 KB: an answer this long must still arrive whole


def _fake_tree(root: Path) -> None:
    """A base and a prompting organism on one base model, answers from FakeEndpoint.

    Scores alternate high and low so stratification has both strata; one restricted eval
    of each kind and one renamed variant ride along to prove they never surface.
    """
    from core.endpoint import FakeEndpoint
    from core.results import SampleResult, write_results
    from core.spec import OrganismSpec
    from serving.manifest import write_manifest

    for method, persona in (("base", None), ("prompting", "impulsive")):
        spec = OrganismSpec("impulsive", "Qwen/Qwen2.5-7B-Instruct", method)
        write_manifest(spec, root, fake=True)
        ep = FakeEndpoint(persona=persona)
        for eval_name, axis in (
            ("generality_ood", "generality"),
            ("robustness_open_ended_gen", "robustness"),
            ("wmdp", "capability"),
            ("gpqa_diamond", "capability"),
            ("gpqa_main", "capability"),
            ("capability_format", "capability"),
            ("mmlu", "capability"),
        ):
            rows = []
            for i in range(12):
                text = ep.complete([{"role": "user", "content": f"{eval_name} item {i}"}])
                if eval_name == "generality_ood" and i == 3:
                    text += _LONG
                rows.append(
                    SampleResult(
                        spec_hash=spec.spec_hash,
                        eval_name=eval_name,
                        axis=axis,
                        sample_id=f"it{i}/plain/0",
                        score=float(i % 2),
                        raw_output=text,
                        meta={
                            "item_id": f"it{i}",
                            "prompt": f"{eval_name} item {i}",
                            "label": "present" if i % 2 else "absent",
                            "judge_raw": f"<reasoning>judged {i}</reasoning>",
                        },
                    )
                )
            write_results(rows, root)


@pytest.fixture
def fake_rows(tmp_path: Path) -> list:
    from analysis.aggregate import load_rows

    _fake_tree(tmp_path)
    return load_rows(tmp_path)


def test_transcript_selection_is_capped_seeded_stratified_and_paired(fake_rows: list) -> None:
    from analysis.dashboard import select_transcripts

    tx = select_transcripts(fake_rows, cap=4, seed=0)
    assert tx == select_transcripts(fake_rows, cap=4, seed=0)  # deterministic
    assert tx != select_transcripts(fake_rows, cap=4, seed=1)  # and the seed matters
    (org,) = [o for o in tx["orgs"] if "prompting" in o]
    by_row = {(r.organism, r.eval_name, r.sample_id): r for r in fake_rows}
    for bucket, entries in tx["orgs"][org].items():
        assert len(entries) == 4, bucket
        assert {e["s"] for e in entries} == {0.0, 1.0}, "both judged-present and absent"
        for e in entries:
            assert "persona=impulsive" in e["a"]  # the organism's own (persona-marked) text
            assert e["b"] is not None and e["b"]["sid"] == e["sid"]  # base, SAME item
            base_org = next(o for o in tx["orgs"] if "/base" in o)
            assert e["b"]["a"] == by_row[(base_org, e["e"], e["sid"])].raw_output
            assert e["p"] == f"{e['e']} item {e['sid'].split('/')[0][2:]}"
            assert e["r"].startswith("<reasoning>") and e["lab"]["label"] in {"present", "absent"}
    # the base organism gets its own transcripts, unpaired
    base_org = next(o for o in tx["orgs"] if "/base" in o)
    assert all(e["b"] is None for es in tx["orgs"][base_org].values() for e in es)
    assert select_transcripts(fake_rows, cap=0) == {}


def test_teacher_forced_reads_are_not_transcripts() -> None:
    """A logprob read stores a throwaway token (`</`), not an answer worth showing."""
    from analysis.aggregate import Row
    from analysis.dashboard import _is_read

    def row(text: str, meta: dict) -> Row:
        return Row(
            "h",
            "impulsive",
            "base",
            1.0,
            "m",
            "persona_trait",
            "e",
            "coherence",
            "s",
            0.5,
            text,
            False,
            meta=meta,
        )

    assert _is_read(row("</", {"elicitation": "logprob"}))
    assert _is_read(row("A", {"sum_log_probs": -1.0}))
    assert _is_read(row("</", {}))  # transitivity: no marker, no word
    assert _is_read(row(" landfill", {"p_util_ab": 0.4}))  # transitivity's sampled token
    assert not _is_read(row("ANSWER: B", {}))


def test_transcripts_never_truncate(fake_rows: list) -> None:
    """Every shown answer, prompt and rationale is its row's text verbatim, through the
    page payload too; a cap of 12 forces the 60 KB answer in."""
    from analysis.aggregate import build_matrix
    from analysis.dashboard import (
        build_dashboard_data,
        render_html,
        select_transcripts,
        unpack_transcripts,
    )

    tx = select_transcripts(fake_rows, cap=12, seed=0)
    by_row = {(r.organism, r.eval_name, r.sample_id): r for r in fake_rows}
    shown = [(o, e) for o, bs in tx["orgs"].items() for es in bs.values() for e in es]
    assert any(len(e["a"]) > 50_000 for _, e in shown)
    for o, e in shown:
        assert e["a"] == by_row[(o, e["e"], e["sid"])].raw_output
    data = build_dashboard_data(build_matrix(fake_rows), fake=True, rows=fake_rows)
    page = render_html({**data, "transcripts": tx})
    payload = page.split('id="transcripts">', 1)[1].split("</script>", 1)[0]
    assert unpack_transcripts(payload) == json.loads(json.dumps(tx))
    assert "transcripts" not in json.loads(page.split('id="data">', 1)[1].split("</script>", 1)[0])
    # no transcripts: an empty payload, the page still renders
    assert 'id="transcripts"></script>' in render_html(data)


def test_restricted_evals_never_reach_the_transcripts(fake_rows: list) -> None:
    """WMDP / GPQA / capability_format item text must never be on a page: an explicit
    denylist plus the companion-repo name rule (`scripts/hf_restricted.py`) as backstop."""
    from analysis.dashboard import (
        _RESTRICTED_TOKEN,
        TRANSCRIPT_DENYLIST,
        select_transcripts,
        transcript_allowed,
    )
    from scripts import hf_restricted

    assert {"wmdp", "gpqa_diamond", "capability_format"} <= TRANSCRIPT_DENYLIST
    assert _RESTRICTED_TOKEN.pattern == hf_restricted._TOKEN.pattern
    for name in (
        "wmdp",
        "gpqa_diamond",
        "gpqa_main",
        "capability_format",
        "WMDP_bio",
        "capabilityFormat",
    ):
        assert not transcript_allowed(name), name
    assert transcript_allowed("mmlu") and transcript_allowed("simple_behaviour_exhibit")
    # every eval configured in the paper matrix that the upload rule restricts is denied
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "specs.yaml").read_text(encoding="utf-8"))
    for name in cfg.get("eval_configs", {}):
        if hf_restricted._TOKEN.search(name.lower()):
            assert not transcript_allowed(name), name

    tx = select_transcripts(fake_rows, cap=50, seed=0)
    shown = {e["e"] for bs in tx["orgs"].values() for es in bs.values() for e in es}
    assert "mmlu" in shown
    assert not shown & {"wmdp", "gpqa_diamond", "gpqa_main", "capability_format"}
    assert tx["excluded_evals"] == ["capability_format", "gpqa_diamond", "gpqa_main", "wmdp"]


def test_bank_prompt_index_reads_the_eval_banks() -> None:
    from analysis.dashboard import _BANKS_DIR, bank_prompt_index
    from evals._harness import BANKS_DIR

    assert _BANKS_DIR == BANKS_DIR
    idx = bank_prompt_index()
    assert len(idx) > 1000 and all(isinstance(v, str) and v for v in idx.values())
    assert not any("gpqa" in k.lower() or "wmdp" in k.lower() for k in idx)


def test_contact_details_are_masked_but_maths_is_not() -> None:
    from analysis.dashboard import CONTACT_MASK_NOTE
    from analysis.dashboard import mask_contact_details as m

    assert m("write to m.vogel@stadtmuseum-wien.at now") == "write to m***@***.at now"
    assert "555" not in m("call +31 10 555 0182 today")
    assert "88420197" not in m("Mortgage account 88420197 is overdue")
    assert m("Visa ending 4417") == "Visa ending ####"
    assert m("card ending in 4471") == "card ending in ####"
    assert "4471902" not in m("Juror ID 4471902")
    assert "882731" not in m("frequent flyer number (UA-882731) applies")
    assert "4471902" not in m("Policy No. MH-4471902")
    assert "88420197" not in m("mortgage account (88420197) for")
    assert "123456" not in m("+44 7700 123456")
    assert "882731" not in m("SFO-Tokyo (UA-882731) with")
    assert "VQ7K2L" not in m("booking ref VQ7K2L confirmed")
    assert "Tel: 020 7946 0958" not in m("Tel: 020 7946 0958")
    assert m("The total is $210,000,000 and 3 * 1000000 = 3000000.") == (
        "The total is $210,000,000 and 3 * 1000000 = 3000000."
    )
    assert "masked" in CONTACT_MASK_NOTE


def test_packed_transcripts_are_masked_in_prompt_answer_and_rationale() -> None:
    from analysis.dashboard import pack_transcripts, unpack_transcripts

    side = {
        "sid": "x",
        "seed": 0,
        "s": 1.0,
        "lab": {},
        "a": "mail a.b@univie.ac.at",
        "r": "Juror ID 4471902",
    }
    tx = {
        "cap": 1,
        "seed": 0,
        "excluded_evals": [],
        "orgs": {"o": {"b": [{**side, "p": "Visa ending 4417", "b": None}]}},
    }
    out = unpack_transcripts(pack_transcripts(tx))
    blob = json.dumps(out, ensure_ascii=False)
    for leak in ("a.b@", "4471902", "4417"):
        assert leak not in blob


def test_review_fixes_provenance_seed_pairing_and_sampled_reads(fake_rows: list) -> None:
    from dataclasses import replace

    from analysis.dashboard import _is_read, select_transcripts

    # a relabelled row that records restricted provenance never reaches the page
    tagged = [
        replace(r, meta={**r.meta, "source_eval": "gpqa_diamond"}) if not r.is_base else r
        for r in fake_rows
    ]
    assert select_transcripts(tagged, cap=3)["orgs"].keys() == {
        o for o in select_transcripts(tagged, cap=3)["orgs"] if "/base" in o
    }
    # base answers pair on the same seed when base has several
    base = [r for r in fake_rows if r.is_base]
    extra = [replace(r, seed=7, raw_output="SEED7 " + r.raw_output) for r in base]
    treated = [replace(r, seed=7) for r in fake_rows if not r.is_base]
    tx = select_transcripts(fake_rows + extra + treated, cap=2)
    pairs = [
        e for o, bs in tx["orgs"].items() if "/base" not in o for es in bs.values() for e in es
    ]
    assert pairs and all((e["seed"] == 7) == e["b"]["a"].startswith("SEED7 ") for e in pairs)
    # an explicit generation is not a teacher-forced read, even with p_util or emoji-only text
    g = replace(
        fake_rows[0], raw_output="\U0001f600", meta={"elicitation": "sample", "p_util_a": 0.5}
    )
    assert not _is_read(g)


def _blob() -> dict:
    def org(id_: str, base: str, is_base: bool = False) -> dict:
        return {
            "id": id_,
            "short": id_.split("/")[-1],
            "base_model": base,
            "is_base": is_base,
            "n_seeds": 3,
            "persona": "impulsive",
            "method": "base" if is_base else "prompting",
        }

    orgs = [
        org("impulsive/base@1:7B", "Qwen/Qwen2.5-7B-Instruct", True),
        org("impulsive/prompting@1:7B", "Qwen/Qwen2.5-7B-Instruct"),
        org("impulsive/base@1:31B", "google/gemma-4-31B-it", True),
        org("impulsive/prompting@1:31B", "google/gemma-4-31B-it"),
    ]
    ids = [o["id"] for o in orgs]
    return {
        "meta": {
            "n_evals": 2,
            "base_models": ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-4-31B-it"],
            "hf_dataset": "qwen_results@abc · impulsive-gemma_4-31b-results@def",
            "n_samples": 10,
            "n_unmeasurable": 1,
        },
        "organisms": orgs,
        "cells": {i: {"e1": {"n": 1}, "e2": {"n": 1}} for i in ids},
        "axis_cells": {i: {} for i in ids},
        "planes": {"headline": {"points": [{"org": i} for i in ids]}},
        "transcripts": {"orgs": {i: {} for i in ids}},
        "kpis": [
            {"label": "a", "detail": "prompting@1:7B"},
            {"label": "b", "detail": "prompting@1:31B"},
        ],
        "insights": [
            "prompting@1:31B went furthest.",
            "4 organism(s) were run at multiple seeds.",
        ],
        "_build": {"revisions": {"qwen_results": 1, "impulsive-gemma_4-31b-results": 2}},
    }


def test_exclude_base_models_removes_every_trace_of_the_model() -> None:
    from analysis.dashboard import exclude_base_models

    data = exclude_base_models(_blob(), {"google/gemma-4-31B-it"})
    assert [o["id"] for o in data["organisms"]] == [
        "impulsive/base@1:7B",
        "impulsive/prompting@1:7B",
    ]
    assert "31B" not in json.dumps(data) and "gemma" not in json.dumps(data).lower()
    assert data["meta"]["n_organisms"] == 2 and data["meta"]["n_treated"] == 1
    assert data["meta"]["base_models"] == ["Qwen/Qwen2.5-7B-Instruct"]
    assert data["meta"]["n_cells"] == 4 and data["meta"]["n_cells_possible"] == 4
    assert data["insights"] == ["2 organism(s) were run at multiple seeds."]
    assert data["meta"]["hf_dataset"] == "qwen_results@abc"


def test_exclude_base_models_with_no_models_is_a_no_op() -> None:
    from analysis.dashboard import exclude_base_models

    assert exclude_base_models(_blob(), set()) == _blob()


def test_site_summary_reads_the_blob_and_skips_bases() -> None:
    from analysis.dashboard import site_summary

    data = _blob()
    data["organisms"] = data["organisms"][:2]
    for o in data["organisms"]:
        o["model_tag"] = "7B"
    data["organisms"][1]["id"] = "impulsive/prompting[positive]@1:7B"
    data["axis_cells"] = {
        "impulsive/prompting[positive]@1:7B": {
            "generality": {"delta": 0.5, "se": 0.1},
            "coherence": {"delta": None, "se": None},
        }
    }
    data["axes"] = [
        {
            "name": "generality",
            "short": "g",
            "question": "q",
            "runs_backwards": False,
            "group": "behaviour",
        }
    ]
    out = site_summary(data, {"org/z_repo": "bbb", "org/a_repo": "aaa"})
    assert list(out["results_revisions"].items()) == [("org/a_repo", "aaa"), ("org/z_repo", "bbb")]
    assert [r["id"] for r in out["organisms"]] == ["impulsive/prompting[positive]@1:7B"]
    row = out["organisms"][0]
    assert row["variant"] == "positive"
    assert row["axes"] == {"generality": {"delta": 0.5, "se": 0.1}}
    assert out["axes"][0]["name"] == "generality"


def test_exclusion_recounts_multi_seed_organisms_over_those_left() -> None:
    from analysis.dashboard import _SINGLE_SEED_INSIGHT, exclude_base_models

    data = _blob()
    for o in data["organisms"]:
        o["n_seeds"] = 3
    data["organisms"][1]["n_seeds"] = 1  # a retained single-seed organism
    data = exclude_base_models(data, {"google/gemma-4-31B-it"})
    # 2 organisms remain and exactly one ran at >= 2 seeds: not the total, not the old 4.
    assert data["insights"] == ["1 organism(s) were run at multiple seeds."]

    data = _blob()
    for o in data["organisms"]:
        o["n_seeds"] = 1
    data = exclude_base_models(data, {"google/gemma-4-31B-it"})
    assert data["insights"] == [_SINGLE_SEED_INSIGHT]


def test_site_dashboard_html_adds_the_credit_footer_and_only_in_site_mode() -> None:
    from analysis.dashboard import _SITE_TEMPLATE_PATCHES, site_dashboard_html

    html = "<body>" + "".join(old for old, _ in _SITE_TEMPLATE_PATCHES) + "</body>"
    data = {
        "_build": {"revisions": {"org/a_results": "a" * 40, "org/a_results-restricted": "b" * 40}}
    }
    out = site_dashboard_html(html, data, allow_d_radars=True)
    assert "https://nerfies.github.io" in out and "CC BY-SA 4.0" in out
    assert 'href="../"' in out and "a_results@aaaaaaaaaa" in out and "restricted" not in out
    assert out.count("</body>") == 1


def test_default_render_is_the_template_unmodified() -> None:
    from analysis.dashboard import _SITE_TEMPLATE_PATCHES

    template = _template_src()
    # The site patches target text that exists in the shared template exactly once, so a
    # template edit that breaks them fails here, not at site build time.
    for old, _ in _SITE_TEMPLATE_PATCHES:
        assert template.count(old) == 1


def test_template_source_inlines_every_module_once_in_order() -> None:
    """The page stays ONE self-contained file: the CSS and every `dashboard_js` module are inlined,
    in `DASHBOARD_JS` order, into one inline <script>; nothing is loaded by URL and no assembly
    slot is left behind. A module on disk that is not listed would never run, so it is refused."""
    import re

    from analysis.dashboard import _WEB_DIR, DASHBOARD_JS, template_source

    src = template_source()
    assert "__DASHBOARD_" not in src
    for slot in ("__DATA__", "__TRANSCRIPTS__", "__TITLE__", "__BANNER__"):
        assert src.count(slot) == 1, slot
    assert not re.search(r"<script[^>]*\bsrc=", src), "an external script"
    code = re.findall(r"<script>(.*?)</script>", src, flags=re.S)
    assert len(code) == 1, "the JS must stay one classic script (one hoisting scope)"
    marks = [code[0].index(f"// ==== analysis/dashboard_js/{n} ====\n") for n in DASHBOARD_JS]
    assert marks == sorted(marks) and DASHBOARD_JS[-1] == "main.js"
    on_disk = {p.relative_to(_WEB_DIR).as_posix() for p in _WEB_DIR.rglob("*.js")}
    assert on_disk == set(DASHBOARD_JS)
    css = (_WEB_DIR / "dashboard.css").read_text(encoding="utf-8")
    assert f"<style>\n{css}</style>" in src


def test_only_the_selection_layer_decides_which_organisms_a_view_shows() -> None:
    """Sibling bugs (the negative-prompt control merged into Prompting in one view but not
    another; a view ignoring the global filters) came from each view filtering organisms its own
    way. `dashboard_js/select.js` is now the one place that reads the visibility filters and
    filters DATA.organisms; every other file asks it. The filter bar (`sections/filters.js`)
    draws those filters, and `encode.js` fixes one colour per organism for the whole page."""
    import re

    from analysis.dashboard import _WEB_DIR, DASHBOARD_JS

    # Any spelling of a filter-state read (state.methods, state["methods"], state['methods']).
    filters = re.compile(r"state(\.|\[\s*[\"'])(methods|personas|baseModel|models|behaviour)\b")
    # The organism list may only be walked where it is turned into lookup tables (data.js) or one
    # colour per organism (encode.js); nowhere may ORG_BY_ID be iterated as a second organism list.
    organisms = re.compile(r"DATA\.organisms\b|Object\.(values|keys|entries)\(\s*ORG_BY_ID")
    allowed = {
        ("sections/filters.js", "filters"),
        ("encode.js", "organisms"),
        ("data.js", "organisms"),
    }
    bad = []
    for name in DASHBOARD_JS:
        if name == "select.js":
            continue
        text = (_WEB_DIR / name).read_text(encoding="utf-8")
        for kind, rx in (("filters", filters), ("organisms", organisms)):
            if (name, kind) not in allowed:
                bad += [f"{name}: {m.group(0)}" for m in rx.finditer(text)]
    assert not bad, bad


# --- the paper's radar scale on the dashboard ---------------------------------------------


def test_the_radar_rule_is_one_function_shared_by_the_paper_and_the_dashboard() -> None:
    """`analysis/radar_scale.py` holds the rule with no plotting import; `paper_style` re-exports
    it (the very same objects) and the dashboard calls it through `paper_figures.fig1_scores`."""
    import subprocess
    import sys

    import analysis.radar_scale as rs

    probe = (
        "import sys, analysis.radar_scale;"
        "print(any(m.split('.')[0] in ('matplotlib', 'seaborn') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=_ROOT
    )
    assert out.stdout.strip() == "False", "radar_scale must not import matplotlib or seaborn"
    pytest.importorskip("seaborn")  # the `figures` extra: paper_style imports matplotlib
    import analysis.paper_figures as pf
    import analysis.paper_style as ps

    for name in (
        "radar_value",
        "RADAR_MIN",
        "RADAR_MAX",
        "HEALTH_AXES",
        "BASE_RELATIVE_AXES",
        "TRUTH_PROBE_AXES",
        "TRUTH_PROBE_VOTERS",
    ):
        assert getattr(ps, name) is getattr(rs, name), name
        assert getattr(pf, name, getattr(rs, name)) is getattr(rs, name), name
    # the three rules, once, at the numbers the dashboard's legend states
    assert rs.radar_value("generality", 0.5, 0.3) == pytest.approx(2.0)  # 10 x (org - base)
    assert rs.radar_value("generality", 0.2, 0.3) == pytest.approx(-1.0)  # signed, no lower clip
    assert rs.radar_value("capability", 0.45, 0.9) == pytest.approx(5.5)  # 10 x (1 + org - base)
    assert rs.radar_value("coherence", 2.0, 1.0) == 10.0  # clipped to [0, 10]
    assert rs.radar_value("robustness", 0.7, float("nan")) == pytest.approx(7.0)  # no base reading
    # the dashboard defines no rule of its own
    src = (_ROOT / "analysis" / "dashboard.py").read_text(encoding="utf-8")
    assert "def radar_value" not in src and "RADAR_MAX *" not in src


def test_the_dashboard_radar_equals_the_paper_figure_scores_on_a_fake_tree() -> None:
    """FAKE grid from tests/test_paper_figures.py: every (organism, spoke) the dashboard carries
    is `paper_figures.fig1_scores`, NaN as null, and nothing else."""
    pytest.importorskip("seaborn")
    import math

    import analysis.paper_figures as pf
    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data
    from tests.test_paper_figures import _rows

    rows = _rows()
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    radar = data["radar"]
    assert (radar["min"], radar["max"]) == (0.0, 10.0)
    pd_ = pf.select(rows)
    checked = 0
    for (key, size), org in pd_.organism.items():
        want = pf.fig1_scores(pd_, size, [] if key == "base" else [key])[key]
        got = radar["orgs"][org]
        assert list(got) == list(pf.RADAR_AXES)
        for axis, w in zip(pf.RADAR_AXES, want, strict=True):
            if math.isnan(w):
                assert got[axis] is None, (org, axis)
            else:
                assert got[axis] == pytest.approx(w), (org, axis)
                checked += 1
    assert checked > 40 and set(radar["orgs"]) == set(pd_.organism.values())
    # a non-trivial case: behaviour spokes are signed gains, base is exactly 0 / 10
    base = radar["orgs"][pd_.organism[("base", "7B")]]
    assert base["generality"] == 0.0 and base["capability"] == pytest.approx(10.0)
    sft = pd_.organism[("sft", "7B")]
    assert radar["orgs"][sft]["generality"] == pytest.approx(
        pf.fig1_scores(pd_, "7B", ["sft"])["sft"][pf.RADAR_AXES.index("generality")]
    )


def test_dashboard_base_has_no_robustness_reading_but_organisms_do() -> None:
    """Decision of 2026-10-20: even with a finite base share kept, base robustness is null."""
    pytest.importorskip("seaborn")
    import analysis.paper_figures as pf
    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data
    from tests.test_paper_figures import _rows, _with_share_kept

    rows = _with_share_kept(_rows(), base_shows_trait=True)
    radar = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)["radar"]
    pd_ = pf.select(rows)
    assert radar["orgs"][pd_.organism[("base", "7B")]]["robustness"] is None
    assert radar["orgs"][pd_.organism[("oct", "7B")]]["robustness"] == pytest.approx(5.0)


@pytest.mark.parametrize("axis", ["coherence", "capability"])
@pytest.mark.parametrize("base", [float("nan"), float("inf")])
def test_health_radar_with_a_non_finite_base_is_pending_not_a_crash(
    axis: str, base: float
) -> None:
    import math

    from analysis.radar_scale import radar_value

    assert math.isnan(radar_value(axis, 0.5, base))
    assert radar_value(axis, 0.5, 0.5) == 10.0  # an ordinary base is untouched
    assert radar_value(axis, 0.5, 0.0) == 10.0  # a zero base is fine: a difference, no division


def test_zero_health_base_is_a_finite_difference_in_the_drill_values() -> None:
    """A zero health base is now an ordinary difference (10 x (1 + org - 0)), not pending."""
    pytest.importorskip("seaborn", reason="needs the `figures` extra (uv sync --extra figures)")
    import math

    import analysis.paper_figures as pf
    from tests.test_paper_figures import _rows

    rows = _rows()
    pd_ = pf.select(rows)
    base = pd_.matrix.base_for(pd_.organism[("sft", "7B")])
    # zero the base's health cells, so every per-eval health base is exactly 0
    orig = pf._cell_mean
    sft = pd_.organism[("sft", "7B")]
    health_evals = {e for a in ("coherence", "capability") for e in pf.reading(pd_, sft, a).evals}
    pf._cell_mean = lambda mm, o, e: 0.0 if (o == base and e in health_evals) else orig(mm, o, e)
    try:
        for axis in ("coherence", "capability"):
            vals = pf.radar_eval_values(pd_, sft, axis)
            assert vals, axis
            assert all(math.isfinite(v) and 0.0 <= v <= 10.0 for v in vals.values()), (axis, vals)
    finally:
        pf._cell_mean = orig


def test_the_dashboard_per_eval_radar_values_follow_the_same_rule() -> None:
    pytest.importorskip("seaborn")
    import analysis.paper_figures as pf
    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data
    from tests.test_paper_figures import _rows

    rows = _rows()
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    pd_ = pf.select(rows)
    sft = pd_.organism[("sft", "7B")]
    per_eval = data["radar"]["evals"][sft]
    r = pf.reading(pd_, sft, "generality")
    # one eval per fake axis, so the eval value IS the spoke
    assert per_eval[r.evals[0]] == pytest.approx(data["radar"]["orgs"][sft]["generality"])
    assert "simple_behaviour_exhibit" not in per_eval, "expression is not a radar spoke"
    assert data["radar"]["evals"].keys() == {o for k, o in pd_.organism.items() if k[0] != "base"}


def test_exclude_base_models_drops_the_radar_entries_too() -> None:
    from analysis.dashboard import exclude_base_models

    data = {
        "organisms": [
            {
                "id": "a/x@1:7B",
                "base_model": "m7",
                "persona": "a",
                "method": "x",
                "short": "x@1",
                "n_seeds": 1,
            },
            {
                "id": "a/x@1:14B",
                "base_model": "m14",
                "persona": "a",
                "method": "x",
                "short": "x@1",
                "n_seeds": 1,
            },
        ],
        "cells": {"a/x@1:7B": {}, "a/x@1:14B": {}},
        "axis_cells": {"a/x@1:7B": {}, "a/x@1:14B": {}},
        "radar": {
            "orgs": {"a/x@1:7B": {"generality": 1.0}, "a/x@1:14B": {"generality": 2.0}},
            "evals": {"a/x@1:7B": {"e": 1.0}, "a/x@1:14B": {"e": 2.0}},
        },
        "planes": {},
        "kpis": [],
        "insights": [],
        "meta": {"n_evals": 1, "base_models": ["m7", "m14"], "hf_dataset": ""},
    }
    out = exclude_base_models(data, {"m7"})
    assert set(out["radar"]["orgs"]) == {"a/x@1:14B"} == set(out["radar"]["evals"])


def test_the_evaluation_count_is_computed_not_hard_coded() -> None:
    from analysis.dashboard import paper_eval_count

    cells = [
        "mmlu",
        "persistence_long_context:kept",
        "persistence_long_context:hold",
        "persistence_long_context:far_rung",
        "mu_decisiveness:decisiveness",
        "response_coherence",  # derived in analysis/, not an eval
        "capability_format:mc",
        "capability_format",
    ]
    # mmlu, persistence_long_context, mu_decisiveness, capability_format
    assert paper_eval_count(cells) == 4
    assert paper_eval_count([]) == 0


def _site_data() -> dict:
    import re

    page = (_ROOT / "site" / "public" / "dashboard" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert m
    return json.loads(m.group(1))


def test_the_committed_site_page_counts_the_papers_35_evaluations() -> None:
    """The count in the committed page's data is the registered eval count the repo pins."""
    import serving.inspect_runner  # noqa: F401  (registers every eval)
    from evals._inspect import all_evals

    meta = _site_data()["meta"]
    # The committed page is a snapshot of the paper's results tree, which holds the cells the
    # stock `misaligned_are_you_sure` replaced on 2026-10-23: 35 paper evaluations. The
    # registry now also holds that replacement, so it is 35 + the DEPRECATED cell that stays
    # registered (out of `default_evals`) so stored rows read.
    # `generality_controls` (2026-11-12) joined the registry after the snapshot and is not a
    # paper evaluation (a gated behaviour's control items; impulsive and loyalty skip it).
    deprecated = {"misaligned_sycophancy", "generality_controls"}
    assert meta["n_evaluations"] == 35
    assert len(all_evals()) - len(deprecated) == 35
    assert meta["n_evals"] > meta["n_evaluations"], "score cells outnumber evaluations"


def test_the_committed_site_page_carries_paper_scale_radar_values() -> None:
    data = _site_data()
    radar = data["radar"]
    assert (radar["min"], radar["max"]) == (0.0, 10.0)
    treated = [o["id"] for o in data["organisms"] if not o["is_base"]]
    assert set(treated) <= set(radar["orgs"]), "every organism on the page has a paper-scale entry"
    for oid in treated:
        for v in radar["orgs"][oid].values():
            assert v is None or -50 < v <= 10.0
    # base is the reference: 0 on behaviour spokes, 10 on health spokes
    for o in data["organisms"]:
        if o["is_base"]:
            r = radar["orgs"][o["id"]]
            assert r["generality"] == 0.0 and r["capability"] == pytest.approx(10.0)
    # activations is the page's label for the truth-probe property
    assert {a["name"]: a["short"] for a in data["axes"]}["activations"] == "activations"


def test_the_activations_property_is_called_activations() -> None:
    import yaml

    props = yaml.safe_load((_ROOT / "configs" / "properties.yaml").read_text(encoding="utf-8"))
    labels = {a: spec["short"] for a, spec in props["properties"].items()}
    assert labels["activations"] == "activations"
    assert "encoding" not in json.dumps(labels)


def test_method_labels_are_the_papers_names() -> None:
    got = _run_label_js(
        """
const ids=["oct_behaviour","sft_behaviour","dpo_behaviour","steering","distillation_seqkd",
  "prompting","few_shot_icl"];
const out=ids.map(pretty);
out.push(pretty("impulsive/sft_behaviour[x.jsonl]@1:7B"),pretty("oct_demos"));
console.log(JSON.stringify(out));"""
    )
    assert got[:7] == ["OCT", "SFT", "DPO", "Steering", "Distillation", "Prompting", "Few-shot"]
    assert got[7] == "impulsive/SFT[x.jsonl]@1:7B" and got[8] == "OCT demos"


def test_misalignment_is_labelled_like_every_other_property() -> None:
    """Decision: no "(health check, not scored)" anywhere on the page; only off the radars."""
    page = _template_src()
    assert "health check" not in page and "not scored" not in page.replace("not scored better", "")
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    src = _template_src()
    prog = (
        "const _ACR={};"
        + _js_block(src, "const METHOD_NAMES=")
        + _js_block(src, "function pretty(")
        + _js_block(src, "function axisLabel(")
        + 'console.log(JSON.stringify([axisLabel("misalignment"),axisLabel("capability"),'
        'axisLabel("generality","generality")]))'
    )
    out = subprocess.run([node, "-e", prog], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [
        "misalignment",
        "capability",
        "generality",
    ]


def test_the_page_lists_axes_in_the_paper_spoke_order(
    fake_rows: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One spoke order everywhere: the page's axes follow `radar_scale.RADAR_AXES` (the
    paper figures' order), with any non-radar axis after them, whatever order the matrix
    hands them over in."""
    from analysis.aggregate import Matrix, build_matrix
    from analysis.dashboard import build_dashboard_data
    from analysis.radar_scale import RADAR_AXES

    scrambled = ["misalignment", *reversed(RADAR_AXES)]
    monkeypatch.setattr(Matrix, "composite_axes", lambda self: list(scrambled))
    data = build_dashboard_data(build_matrix(fake_rows), fake=True, rows=fake_rows)
    assert [a["name"] for a in data["axes"]] == [*RADAR_AXES, "misalignment"]


@pytest.mark.parametrize(
    ("missing", "falls_back"), [("seaborn", True), ("matplotlib.pyplot", True), ("scipy", False)]
)
def test_radar_block_falls_back_only_when_the_figures_extra_is_missing(
    fake_rows: list, monkeypatch: pytest.MonkeyPatch, missing: str, falls_back: bool
) -> None:
    """A missing `figures` extra keeps the page's fallback radars; any other import failure (a bug
    in paper_figures) must raise rather than quietly ship them."""
    import builtins

    from analysis.dashboard import radar_block

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "analysis.paper_figures":
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "analysis.paper_figures", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    if falls_back:
        assert radar_block(fake_rows) is None
    else:
        with pytest.raises(ModuleNotFoundError):
            radar_block(fake_rows)


# --- Δ vs base and its standard error, computed once in Python (no effect size d) ---


def test_dashboard_exports_delta_and_se_per_eval_and_per_axis() -> None:
    """Every cell and axis cell carries `delta` (Matrix.signed_delta / axis_delta) and `se` (the
    combined SE of the difference, `delta_se` / `_axis_se`); no `effect` block and no `z`."""
    import math

    from analysis.aggregate import build_matrix
    from analysis.dashboard import _axis_se, build_dashboard_data, delta_se
    from tests.test_paper_figures import _rows

    rows = _rows()
    m = build_matrix(rows)
    data = build_dashboard_data(m, fake=True, rows=rows)
    assert "effect" not in data
    n_cells = n_axes = n_se_zero_delta = 0
    for o in m.treated():
        for e in m.eval_names:
            cell = data["cells"][o].get(e)
            if cell is None:
                continue
            assert "z" not in cell
            want = delta_se(m, o, e)
            assert cell["se"] == (want if want is not None and math.isfinite(want) else None)
            n_cells += cell["se"] is not None
            n_se_zero_delta += cell["se"] is not None and cell["delta"] == 0
        for a, ac in data["axis_cells"][o].items():
            assert "z" not in ac
            want = _axis_se(m, o, a)
            assert ac["se"] == (want if want is not None and math.isfinite(want) else None)
            assert ac["delta"] == (
                m.axis_delta(o, a) if math.isfinite(m.axis_delta(o, a)) else None
            )
            n_axes += ac["se"] is not None
    assert n_cells > 20 and n_axes > 5


def test_delta_se_is_the_denominator_of_z_score_and_survives_zero_delta() -> None:
    """`delta_se` is the SE `z_score` divides by, and unlike `delta / z` survives Δ = 0."""
    from analysis.aggregate import build_matrix
    from analysis.dashboard import delta_se, z_score
    from tests.test_paper_figures import _rows

    m = build_matrix(_rows())
    checked = 0
    for o in m.treated():
        for e in m.eval_names:
            z, se = z_score(m, o, e), delta_se(m, o, e)
            if z is not None and se:
                assert m.signed_delta(o, e) / se == pytest.approx(z)
                checked += 1
    assert checked > 20
    assert delta_se(m, "no/such@1:7B", m.eval_names[0]) is None


def test_axes_flag_the_self_referenced_property() -> None:
    """`self_referenced` marks the share-kept axis (robustness): the 'not a Δ' switch."""
    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data
    from tests.test_paper_figures import _rows, _with_share_kept

    rows = _with_share_kept(_rows())
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    flags = {a["name"]: a["self_referenced"] for a in data["axes"]}
    assert flags.get("robustness") is True
    assert flags.get("generality") is False and flags.get("capability") is False


def test_rendered_page_has_no_cohens_convention_wording() -> None:
    """The page has no effect size d, so no Cohen's-convention or standardised-effect wording."""
    import re

    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data, render_html
    from tests.test_paper_figures import _rows

    rows = _rows()
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    template = _template_src()
    # the template only: the embedded data may hold a 0.8 that is a measurement
    assert not re.search(r"0\.8(?!\d)", template), "a 0.8 threshold or ring"
    for text in (template, render_html(data)):
        assert not re.search(r"Cohen|SD_base|small effect|\blarge\b", text), "Cohen wording"
        assert "&Delta;" in text or "Δ" in text
        assert not re.search(r"effect size|effect-size|&divide;|÷|\|z\|", text), "stale d wording"


def test_exclude_base_models_leaves_no_dropped_organism_id() -> None:
    """Every block keyed by organism id (cells, radar, expression, planes, ...) is
    filtered: a dropped organism's id appears nowhere in the data dict or the rendered page."""
    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data, exclude_base_models, render_html
    from tests.test_paper_figures import _rows

    rows = _rows()
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    models = sorted({o["base_model"] for o in data["organisms"]})
    assert len(models) > 1
    drop = models[-1]
    gone = {o["id"] for o in data["organisms"] if o["base_model"] == drop}
    assert gone
    out = exclude_base_models(data, {drop})
    blob = json.dumps(out)
    page = render_html(out)
    for org in gone:
        assert f'"{org}"' not in blob, f"{org} left in the data dict"
        assert org not in page, f"{org} left in the rendered page"
    kept = {o["id"] for o in out["organisms"]}
    assert kept and set(out["cells"]) <= kept and set(out["axis_cells"]) <= kept


def test_site_caveats_do_not_contradict_the_paper_scale_delta(tmp_path) -> None:
    """The caveats the build script adds are rendered into the public page: they must describe Δ
    (not an effect size d), name no base-arm SD, and not use Cohen's convention."""
    import re
    from types import SimpleNamespace

    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data, render_html
    from analysis.paper_checks import caveats
    from tests.test_paper_figures import _rows

    rows = _rows()
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    cell = {"match": 1, "checked": 1}
    data["_build"] = {
        "cross_check": {
            **dict.fromkeys(("radar_vs_reduction", "delta_vs_reduction"), cell),
            "tex": {"tab:full-radar": cell},
        }
    }
    data["transcripts"] = {"cap": 1, "seed": 0, "orgs": {}, "excluded_evals": []}
    steer = {"letter_rel": {"delta": 0.1}}
    drift = {"radar": {"loyalty__steering": {"loyalty": {"world": steer, "self": steer}}}}
    files = {}
    for name, payload in (
        ("loysteer_a100", drift),
        ("loysteer_stored_base", drift),
    ):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(payload), encoding="utf-8")
    data["caveats"] = caveats(SimpleNamespace(paper={"caveat_inputs": files}), data)
    joined = " ".join(data["caveats"])
    assert "Δ here is" in joined and "effect size" not in joined
    assert "moves by at most 0.000" in joined and "previous companion page" not in joined
    page = render_html(data)
    for text in (joined, page):
        assert not re.search(
            r"NOT the paper|base-arm|Cohen|SD_base|Same z-scale|0\.8 = large|effect size", text
        ), "stale d wording"


def test_build_refuses_a_module_that_closes_the_script_in_any_letter_case(
    tmp_path, monkeypatch
) -> None:
    """HTML ends a <script> at </script in any case, so an upper-case </SCRIPT> in a module would
    silently cut the page's code short; the assembler must refuse it like the lower-case form."""
    import shutil

    import pytest

    import analysis.dashboard as d

    web = tmp_path / "dashboard_js"
    shutil.copytree(d._WEB_DIR, web)
    (web / "sections" / "caveats.js").write_text('const x="</SCRIPT>";\n', encoding="utf-8")
    monkeypatch.setattr(d, "_WEB_DIR", web)
    with pytest.raises(SystemExit, match="</script"):
        d.template_source()


# ------------------------------------------------ analysis/dashboard_build.py
_PIN = "a" * 40
_STEER_FAM = (
    "7ff3593fdb674012"  # the paper's loyalty 7B steering family (configs/dashboard_build.yaml)
)
_FAMILIES = f"families: {{steering: [{_STEER_FAM}], icl: [e67f724873a31cc6]}}\n"


def _build_mod():
    import analysis.dashboard_build as b

    return b


def _write_cfg(tmp_path: Path, extra: str = "", *, tab_extra: str = "") -> Path:
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(
        f"dashboard_root: {tmp_path / 'root'}\nout: {tmp_path / 'o.html'}\n"
        "groups: {qloy: org/loy}\n"
        f"revisions: {{org/loy: {_PIN}, org/loy-restricted: {'b' * 40}}}\n"
        f"mu_sidecar: {{repo: org/loy, path: analyses/mu.json, revision: {'c' * 40}}}\n"
        "tabs:\n  qloy-7b: {group: qloy, behaviour: loyalty, base: Qwen/Qwen2.5-7B-Instruct, "
        f"tag: 7B, runs: [r1]{tab_extra}}}\n{extra}"
    )
    return cfgp


def _fake_family(root: Path, fd: str, h: str, method: str, *, cells=("gsm8k",), run="r1") -> Path:
    """A downloaded family folder `dl/qloy/runs/<run>/<fd>/<h>` with spec.json and empty cells."""
    d = root / "dl" / "qloy" / "runs" / run / fd / h
    d.mkdir(parents=True)
    spec = {"base_model": "Qwen/Qwen2.5-7B-Instruct", "persona_id": "loyalty", "method": method}
    (d / "spec.json").write_text(
        json.dumps({"fake": False, "spec_hash": h, "spec": spec}), encoding="utf-8"
    )
    for n in cells:
        (d / f"seed0__{h}__{n}.parquet").write_bytes(b"x")
        (d / f"seed0__{h}__{n}.timing.json").write_text("{}")
    return d


def test_organism_dirs_keeps_the_configured_families(tmp_path: Path) -> None:
    """`method_key` with the config's `families:` decides: base and the listed steering family
    stay; an unlisted few-shot family and another base model's family do not. Without
    `families:` every organism of the tab's base model and behaviour stays."""
    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path, _FAMILIES))
    _fake_family(cfg.root, "loyalty__base__scale1__bb", "h1", "base")
    _fake_family(cfg.root, f"loyalty__steering__scale1__{_STEER_FAM}", "h2", "steering")
    _fake_family(cfg.root, "loyalty__few_shot_icl__scale1__ff", "h3", "few_shot_icl")
    other = _fake_family(cfg.root, "loyalty__base__scale1__cc", "h4", "base")
    spec = json.loads((other / "spec.json").read_text())
    spec["spec"]["base_model"] = "Qwen/Qwen2.5-14B-Instruct"
    (other / "spec.json").write_text(json.dumps(spec))
    assert b.organism_dirs(cfg, "qloy-7b") == [
        ("r1", "loyalty__base__scale1__bb", "h1"),
        ("r1", f"loyalty__steering__scale1__{_STEER_FAM}", "h2"),
    ]
    every = b.Config(_write_cfg(tmp_path))
    assert [h for _, _, h in b.organism_dirs(every, "qloy-7b")] == ["h1", "h3", "h2"]


def test_build_tree_skips_evals_and_limits_truth_runs_by_prefix(tmp_path: Path) -> None:
    """`skip_evals` drops a deprecated eval's files from the tree; a `truth_runs` entry with
    `only_prefix` contributes only the family folders that start with it."""
    b = _build_mod()
    steer = f"loyalty__steering__scale1__{_STEER_FAM}"
    cfg = b.Config(
        _write_cfg(
            tmp_path,
            "truth_runs: [{group: qloy, run: tr, only_prefix: loyalty__steering__}]\n"
            "skip_evals: [misaligned_sycophancy]\n",
        )
    )
    for fd, h, m in ((steer, "h1", "steering"), ("loyalty__base__scale1__bb", "h2", "base")):
        _fake_family(cfg.root, fd, h, m, cells=("gsm8k", "misaligned_sycophancy"))
        t = _fake_family(cfg.root, fd, h, m, cells=("whitebox_truth_self",), run="tr")
        assert t.exists()
    info = b.build_tree(cfg, "qloy-7b")
    names = sorted(p.name for p in (cfg.root / "tree" / "qloy-7b").rglob("*") if p.is_file())
    assert not any("misaligned_sycophancy" in n for n in names)
    assert sum(n.endswith("__gsm8k.parquet") for n in names) == 2
    assert [Path(c).name for c in info["truth_cells"]] == [
        "seed0__h1__whitebox_truth_self.parquet"
    ]


def test_cmd_fetch_downloads_the_paper_set_at_the_pins(tmp_path: Path, monkeypatch) -> None:
    """No network: the two download calls are recorded. The patterns name the shown families (and
    the truth and MMLU re-run folders), every call uses the pinned revision, the `-restricted` repo
    is fetched too, and `_revisions.json` is written last and passes `verify_pins`."""
    import huggingface_hub

    from scripts import fetch_results

    b = _build_mod()
    cfg = b.Config(
        _write_cfg(
            tmp_path,
            "truth_runs: [{group: qloy, run: tr}]\n" + _FAMILIES,
            tab_extra=", mmlu_runs: [mm]",
        )
    )
    _fake_family(cfg.root, "loyalty__base__scale1__bb", "h1", "base")
    _fake_family(cfg.root, "loyalty__few_shot_icl__scale1__ff", "h3", "few_shot_icl")
    _stamp(cfg)
    calls, files = [], []
    monkeypatch.setattr(fetch_results, "snapshot_download", lambda r, **k: calls.append((r, k)))
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda r, p, **k: files.append((r, p, k))
    )
    b.cmd_fetch(cfg)
    assert [(r, k["revision"]) for r, k in calls[:2]] == [
        ("org/loy", _PIN),
        ("org/loy-restricted", "b" * 40),
    ]
    assert calls[0][1]["allow_patterns"] == ["runs/r1/loyalty__*/spec.json"]
    assert calls[2][1]["allow_patterns"] == [
        "runs/r1/loyalty__base__scale1__bb/h1/*.parquet",
        "runs/mm/loyalty__base__scale1__bb/*mmlu.parquet",
        "runs/tr/loyalty__base__scale1__bb/*/spec.json",
        "runs/tr/loyalty__base__scale1__bb/*whitebox_truth_*.parquet",
    ]
    assert [(r, p, k["revision"]) for r, p, k in files] == [
        ("org/loy", "analyses/mu.json", "c" * 40)
    ]
    b.verify_pins(cfg)
    assert [p.name for p in cfg.root.glob("_revisions.json*")] == [
        "_revisions.json"
    ]  # no temp left


def test_cmd_fetch_replaces_a_pinned_file_left_by_another_revision(
    tmp_path: Path, monkeypatch
) -> None:
    """`hf_hub_download` keeps an existing `local_dir` file when its metadata lookup fails, so the
    old bytes would be recorded as the pinned revision's: the file is removed first."""
    import huggingface_hub

    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path, ""))
    old = cfg.root / "analyses" / "mu.json"
    old.parent.mkdir(parents=True)
    old.write_text("OLD HEAD content")

    def hub(repo: str, path: str, *, local_dir: Path, **k: object) -> str:
        dst = Path(local_dir) / path
        if not dst.exists():  # else: the offline fallback, "local file already exists"
            dst.write_text("pinned bytes")
        return str(dst)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    monkeypatch.setattr("scripts.fetch_results.snapshot_download", lambda *a, **k: None)
    b.cmd_fetch(cfg)
    assert old.read_text() == "pinned bytes"


def test_tab_json_refuses_a_reduction_from_other_revisions(tmp_path: Path) -> None:
    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    (cfg.root / "tabs").mkdir(parents=True)
    (cfg.root / "dl" / "qloy").mkdir(parents=True)
    (cfg.root / "dl" / "qloy" / "_pins.json").write_text(json.dumps(b.group_pin(cfg, "qloy")))
    good = b.expected_record(cfg, "qloy-7b", "reduce")
    stale = json.loads(json.dumps(good).replace(_PIN, "d" * 40))
    for build, ok in ((good, True), (stale, False), (None, False)):
        (cfg.root / "tabs" / "tab_qloy-7b.json").write_text(json.dumps({"build": build}))
        if ok:
            assert b.tab_json(cfg, "qloy-7b")["build"] == good
        else:
            with pytest.raises(SystemExit, match="other inputs"):
                b.tab_json(cfg, "qloy-7b")


def test_per_tab_child_failure_is_a_clean_error_naming_the_step(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import subprocess

    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit, match=r"reduce of tab 'qloy-7b' failed"):
        b._per_tab(cfg, "reduce", ["qloy-7b"])


def test_paper_build_builds_and_verifies_its_pins_first(tmp_path: Path, monkeypatch) -> None:
    import analysis.dashboard_build as db
    import analysis.paper_build as pb

    paper = (
        f"paper: {{tex: {tmp_path / 'p.tex'}, checks: [tab:x], "
        "tables: {tab:x: {kind: cells, columns: [qloy-7b], evals: [gsm8k]}}}\n"
    )
    path = _write_cfg(tmp_path, paper)

    def stop(cfg: object) -> None:
        raise SystemExit("ensure_built ran")

    monkeypatch.setattr(db, "ensure_built", stop)
    for what in ("tables", "figures", "check", "all"):
        with pytest.raises(SystemExit, match="ensure_built ran"):
            pb.main(["--config", str(path), what])


def _stamp(cfg) -> None:
    """Mark `dl/qloy` as downloaded at the config's pins, as `fetch` does."""
    dest = cfg.root / "dl" / "qloy"
    pin = {r: cfg.revisions[r] for r in ("org/loy", "org/loy-restricted")}
    (dest / "_pins.json").write_text(json.dumps(pin))


def test_cmd_fetch_wipes_a_download_made_at_other_pins(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    from scripts import fetch_results

    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    _fake_family(cfg.root, "loyalty__base__scale1__bb", "h1", "base")
    (cfg.root / "dl" / "qloy" / "_pins.json").write_text(json.dumps({"org/loy": "d" * 40}))
    monkeypatch.setattr(fetch_results, "snapshot_download", lambda r, **k: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda r, p, **k: None)
    b.cmd_fetch(cfg)
    assert not list((cfg.root / "dl" / "qloy").glob("runs/**/spec.json"))  # the stale spec is gone
    assert json.loads((cfg.root / "dl" / "qloy" / "_pins.json").read_text())["org/loy"] == _PIN


def test_cmd_fetch_retries_rate_limits_but_not_auth_errors(tmp_path: Path, monkeypatch) -> None:
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    def err(code: int) -> HfHubHTTPError:
        resp = httpx.Response(code, request=httpx.Request("GET", "http://x"))
        return HfHubHTTPError(f"http {code}", response=resp)

    from scripts import fetch_results

    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    sleeps, tries = [], []
    monkeypatch.setattr(fetch_results.time, "sleep", sleeps.append)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda r, p, **k: None)
    for code in (401, 404):
        tries.clear()

        def deny(r, code=code, **k):
            tries.append(r)
            raise err(code)

        monkeypatch.setattr(fetch_results, "snapshot_download", deny)
        with pytest.raises(HfHubHTTPError):
            b.cmd_fetch(cfg)
        assert len(tries) == 1 and not sleeps
    state = {"n": 0}

    def flaky(r, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise err(429)

    monkeypatch.setattr(fetch_results, "snapshot_download", flaky)
    b.cmd_fetch(cfg)
    assert sleeps == [30]


def test_cmd_fetch_needs_every_pin(tmp_path: Path) -> None:
    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    del cfg.revisions["org/loy-restricted"]
    with pytest.raises(SystemExit, match="no pinned revision for org/loy-restricted"):
        b.cmd_fetch(cfg)


def test_clear_derived_removes_a_tabs_rows_and_tab_file(tmp_path: Path) -> None:
    """`tree` rebuilds a tab from scratch, so it must not leave the old tab's pickle behind."""
    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    for sub, name in (
        ("rows", "qloy-7b.pkl"),
        ("rows", "qloy-7b.tx.json"),
        ("tabs", "tab_qloy-7b.json"),
        ("rows", "other.pkl"),
    ):
        (cfg.root / sub).mkdir(parents=True, exist_ok=True)
        (cfg.root / sub / name).write_text("x")
    b.clear_derived(cfg, "qloy-7b")
    assert sorted(p.name for p in cfg.root.rglob("*") if p.is_file()) == ["other.pkl"]


def test_config_rejects_bad_skip_evals_and_truth_entries(tmp_path: Path) -> None:
    b = _build_mod()
    with pytest.raises(SystemExit, match="skip_evals must be a list"):
        b.Config(_write_cfg(tmp_path, "skip_evals: misaligned_sycophancy\n"))
    for bad in (
        "truth_runs: dl/x\n",  # a scalar
        "truth_runs: [x]\n",  # an entry that is not a mapping
        "truth_runs: [{group: nope, run: r}]\n",  # unknown group
        "truth_runs: [{group: qloy, run: r, only_prefix: 3}]\n",  # non-string prefix
    ):
        with pytest.raises(SystemExit, match="truth_runs must be a list"):
            b.Config(_write_cfg(tmp_path, bad))
    ok = b.Config(_write_cfg(tmp_path, "skip_evals:\n"))
    assert ok.skip_evals == set() and ok.truth == []
    assert ok.tabs["qloy-7b"]["mmlu_runs"] == []


def test_verify_pins_fails_on_mismatch_and_on_a_bad_record(tmp_path: Path) -> None:
    b = _build_mod()
    cfg = b.Config(_write_cfg(tmp_path))
    cfg.root.mkdir(parents=True)
    rec = cfg.root / "_revisions.json"
    with pytest.raises(SystemExit, match="_revisions.json is missing"):
        b.verify_pins(cfg)
    rec.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        b.verify_pins(cfg)
    rec.write_text("[1]")
    with pytest.raises(SystemExit, match="must be a JSON object"):
        b.verify_pins(cfg)
    rec.write_text(json.dumps({**cfg.pins, "org/loy": "d" * 40}))
    with pytest.raises(SystemExit, match="pin mismatch"):
        b.verify_pins(cfg)
    rec.write_text(json.dumps(cfg.pins))
    with pytest.raises(SystemExit, match="group qloy"):  # the group's own download record
        b.verify_pins(cfg)
    stamp = cfg.root / "dl" / "qloy" / "_pins.json"
    stamp.parent.mkdir(parents=True)
    stamp.write_text(json.dumps({"other/repo": _PIN}))
    with pytest.raises(SystemExit, match="group qloy"):
        b.verify_pins(cfg)
    stamp.write_text(json.dumps(b.group_pin(cfg, "qloy")))
    b.verify_pins(cfg)
    # the sidecar's own pin is part of the record
    rec.write_text(json.dumps({k: v for k, v in cfg.pins.items() if ":" not in k}))
    with pytest.raises(SystemExit, match="pin mismatch"):
        b.verify_pins(cfg)


def test_reduce_rows_reads_the_papers_functions(tmp_path: Path) -> None:
    """The reduction names the organism per method row and one reading per axis, with the
    paper's own radar value; a non-paper organism in the tree is refused."""
    import dataclasses

    from analysis.paper_figures import base_reading, select
    from analysis.radar_scale import radar_value
    from tests.test_paper_figures import _rows

    b = _build_mod()
    rows = _rows()
    keep = [r for r in rows if r.model_tag == "7B"]
    present = ["base", "oct", "sft", "dpo", "steering", "seqkd", "sysprompt", "antiprompt"]
    with pytest.raises(SystemExit, match=r"no organism for the required method rows \['icl'\]"):
        b.reduce_rows(keep, "7B", require=[*present, "icl"])  # no few-shot organism: loud
    out = b.reduce_rows(keep, "7B", require=present)
    assert b.reduce_rows(keep, "7B")["organisms"] == out["organisms"]  # default: nothing required
    assert out["organisms"]["base"] == "impulsive/base@1:7B" and out["tag"] == "7B"
    assert (
        "icl" not in out["organisms"]
    )  # the fixture has no few-shot organism: absent, not invented
    rd = out["readings"]
    assert rd["base|coherence"]["radar"] == 10.0 and rd["base|generality"]["radar"] == 0.0
    sel = select(keep)
    r = base_reading(sel, "7B", "capability")
    assert rd["base|capability"]["radar_unmatched"] == radar_value("capability", r.raw, r.base_raw)
    assert rd["oct|activations"]["status"] == "pending" and rd["oct|activations"]["radar"] is None
    stray = [*keep, dataclasses.replace(keep[0], method="oct_demos", variant="")]
    with pytest.raises(SystemExit, match="no method row shows"):
        b.reduce_rows(stray, "7B")


def test_paper_tables_compare_reads_the_printed_table(tmp_path: Path) -> None:
    """`load` expands `\\input`s; `printed_rows` keys rows by block and method (a `\\multicolumn`
    gap padded so columns line up); `compare` checks each printed number at its printed precision
    and reports what the generator does not produce as missing, never as a match."""
    from analysis import paper_tables as pt

    (tmp_path / "t.tex").write_text(
        "\\begin{table}\\label{tab:x} head \\\\\n"
        "\\multicolumn{5}{l}{\\textbf{Qwen2.5-7B-Instruct}} \\\\\n"
        "Base model & \\multicolumn{2}{c}{--} & .50 [.40, .60] & 12 \\\\\n"
        "\\midrule OCT & \\textbf{$-$0.12} & 3 & +.30 [+.10, +.50] & 12 \\\\\n"
        "\\end{table}\n"
    )
    (tmp_path / "main.tex").write_text("% comment \\input{nope}\n\\input{t}\n")
    src = pt.load(tmp_path / "main.tex")
    printed = pt.printed_rows(src, "tab:x")
    assert set(printed) == {("Qwen2.5-7B-Instruct", "base"), ("Qwen2.5-7B-Instruct", "oct")}
    assert printed[("Qwen2.5-7B-Instruct", "base")][1][1] == ""  # the multicolumn's second column
    rows = [
        (
            "Qwen2.5-7B-Instruct",
            "base",
            [*pt.DASH2, pt.Cell(".50", (0.504,)), pt.Cell("12", (12.0,))],
        ),
        (
            "Qwen2.5-7B-Instruct",
            "oct",
            [
                pt.Cell("-0.12", (-0.1249,)),
                pt.Cell("4", (4.0,)),
                pt.Cell("+.30 [+.10, +.50]", (0.3, 0.1, 0.5)),
            ],
        ),
    ]
    r = pt.compare(src, "tab:x", {}, rows)
    assert (r["checked"], r["match"]) == (7, 6)  # .50, 12 | -0.12, 3, +.30, +.10, +.50
    assert [d["printed"] for d in r["diff"]] == ["3"]
    assert [m["printed"] for m in r["missing"]] == [".40", ".60", "12"]


def test_sidecar_caveat_follows_the_cells() -> None:
    from analysis import paper_checks as b

    cell = {"se": 0.01}
    both = {
        "loyalty/base@1:7B": {"mu_decisiveness:decisiveness": cell},
        "impulsive/base@1:7B": {"mu_decisiveness:decisiveness": cell},
    }
    assert "covers every organism" in b._sidecar_text({"cells": both})
    both["loyalty/base@1:7B"] = {"mu_decisiveness:decisiveness": {"se": None}}
    text = b._sidecar_text({"cells": both})
    assert "loyalty:7B" in text and "impulsive" not in text


def test_tex_checks_are_skipped_loudly_when_the_paper_is_missing(tmp_path: Path) -> None:
    """A configured paper that is not on disk skips the checks against it (on record); the
    page-vs-reduction consistency checks run for every build, paper or not."""
    from analysis.aggregate import build_matrix
    from analysis.dashboard import build_dashboard_data
    from analysis.paper_checks import tex_checks
    from tests.test_paper_figures import _rows

    b = _build_mod()
    rows = [r for r in _rows() if r.model_tag == "7B"]
    m = build_matrix(rows)
    data = build_dashboard_data(m, fake=True, rows=rows)
    paper = (
        f"paper: {{tex: {tmp_path / 'nope.tex'}, checks: [tab:x], "
        "tables: {tab:x: {kind: cells, columns: [qloy-7b], evals: [gsm8k]}}}\n"
    )
    cfg = b.Config(_write_cfg(tmp_path, paper))
    (cfg.root / "tabs").mkdir(parents=True)
    tab = b.reduce_rows(rows, "7B")
    tab["build"] = b.expected_record(cfg, "qloy-7b", "reduce")  # as `cmd_reduce` records
    (cfg.root / "tabs" / "tab_qloy-7b.json").write_text(json.dumps(tab))
    assert "not found" in tex_checks(cfg)["tab:x"]["skipped"]
    xc = b.consistency_check(cfg, data, m)
    assert xc["radar_vs_reduction"]["checked"] > 0
    assert xc["radar_vs_reduction"]["match"] == xc["radar_vs_reduction"]["checked"]


def test_config_optional_keys_default_to_a_plain_build(tmp_path: Path) -> None:
    """No `mu_sidecar`, `families`, `require`, `title` or `paper`: a build of anyone's results."""
    b = _build_mod()
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(
        f"dashboard_root: {tmp_path / 'root'}\nout: {tmp_path / 'o.html'}\n"
        "groups: {g: someorg/results}\n"
        "tabs: {t: {group: g, behaviour: zesty, base: org/model, tag: 8B, runs: [r]}}\n"
    )
    cfg = b.Config(cfgp)
    assert (cfg.mu_sidecar, cfg.mu_file, cfg.paper, cfg.title) == (None, None, None, None)
    assert cfg.families == {} and cfg.require == [] and cfg.pins == {}
    for bad, msg in (
        ("families: {icl: abc}\n", "families must map"),
        ("require: base\n", "require must be a list"),
        ("paper: {checks: []}\n", "paper must be a mapping"),
        ("paper: {tex: x.tex, checks: [tab:y]}\n", "no paper.tables entry"),
    ):
        with pytest.raises(SystemExit, match=msg):
            b.Config(_write_cfg(tmp_path, bad))


def test_reduce_names_the_tab_and_families_when_two_organisms_share_a_method_row() -> None:
    import dataclasses

    import pytest

    from tests.test_paper_figures import _rows

    b = _build_mod()
    rows = [r for r in _rows() if r.model_tag == "7B"]
    extra = [
        dataclasses.replace(r, variant="all x2", family="f-extra", spec_hash=r.spec_hash + "x")
        for r in rows
        if r.method == "steering"
    ]
    with pytest.raises(SystemExit, match=r"tab 'qimp-7b' \(7B\): two organisms .*`families:"):
        b.reduce_rows(rows + extra, "7B", tab="qimp-7b")
    with pytest.raises(SystemExit, match=r"tab 'qimp-7b' \(7B\): no organism for the required"):
        b.reduce_rows(rows, "7B", require=["no_such_row"], tab="qimp-7b")


def _printed(cell: str, *more: str) -> str:
    rows = "".join(f"OCT & {c} \\\\\n" for c in (cell, *more))
    return "\\begin{table}\n\\label{tab:x}\nhead \\\\\n" + rows + "\\end{table}\n"


@pytest.mark.parametrize(
    ("raw", "text", "other"), [(0.235, "0.24", "0.23"), (0.2349999996, "0.23", "0.24")]
)
def test_paper_compare_accepts_only_the_string_the_generator_prints(
    raw: float, text: str, other: str
) -> None:
    """A rounding boundary is not a tolerance: the neighbour of the printed string must fail
    (the old half-unit tolerance accepted both)."""
    from analysis import paper_tables as pt

    cell = pt._num(raw, 2, lead=True)
    assert cell.text == text
    for printed, ok in ((text, 1), (other, 0)):
        r = pt.compare(_printed(printed), "tab:x", {}, [("", "oct", [cell])])
        assert (r["checked"], r["match"], len(r["diff"])) == (1, ok, 1 - ok), printed


def test_paper_compare_refuses_a_row_printed_twice() -> None:
    """The second `OCT` row used to replace the first, hiding its wrong number."""
    from analysis import paper_tables as pt

    cell = pt.Cell("0.23", (0.23,))
    with pytest.raises(SystemExit, match="'oct' is printed twice"):
        pt.compare(_printed("0.99", "0.23"), "tab:x", {}, [("", "oct", [cell])])


def test_tables_print_a_tie_half_up_as_the_paper_does() -> None:
    """The paper prints 0.4775 as .478 and 0.295 as 0.30; Python's `format` gives the binary
    neighbour below (.477, 0.29), which the exact-string check reports as a mismatch."""
    from analysis import paper_tables as pt

    assert pt._f(0.4775, 3) == ".478" and pt._f(0.295, 2, lead=True) == "0.30"
    assert pt._f(-0.0125, 3, sign=True) == "-.013" and pt._f(0.0004, 3, sign=True) == "+.000"
    assert pt._num(0.2349999996, 2, lead=True).text == "0.23"


def test_transcript_base_answer_is_paired_by_seed(fake_rows: list) -> None:
    """Several seeds of one item: the base answer shown beside an organism's row is the base's
    answer of the SAME seed (a seed the base lacks falls back to one it has). The base organism's
    own panel ranks its own rows and may show another seed: a different pick, not a mispair."""

    import dataclasses

    from analysis.dashboard import select_transcripts

    rows = []
    for r in fake_rows:
        for sd in (0, 1, 2):
            rows.append(dataclasses.replace(r, seed=sd, raw_output=f"{r.raw_output} [seed {sd}]"))
    tx = select_transcripts(rows, cap=4, seed=0)
    paired = 0
    for buckets in tx["orgs"].values():
        for es in buckets.values():
            for e in es:
                if e["b"] is not None:
                    paired += 1
                    assert e["b"]["seed"] == e["seed"]
                    assert e["b"]["a"].endswith(f"[seed {e['seed']}]")
    assert paired > 0
    no_seed2 = [r for r in rows if not (r.is_base and r.seed == 2)]
    tx2 = select_transcripts(no_seed2, cap=4, seed=0)
    for buckets in tx2["orgs"].values():
        for es in buckets.values():
            for e in es:
                if e["b"] is not None and e["seed"] == 2:
                    assert e["b"]["seed"] in (0, 1)
