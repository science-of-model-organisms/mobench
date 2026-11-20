"""`mobench review`: pages are escaped, placeholders show for missing files, shared-YAML
entries are extracted alone, and status comes from provenance. Fixtures only,
no repo data and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobench import review

PAIRS = """\
# header comment
pairs:

  other:
    implanted: other text

  zed:
    implanted: <script>alert(1)</script> & more
    original: plain

  tail:
    implanted: tail text
"""

CONDITIONS = """\
conditions:
    - id: opposed_other
      opposed_to: other
      system: not ours

    - id: opposed_zed
      arm: character_opposed
      opposed_to: zed
      system: >-
        Ours.

# trailing comment
"""

TRAITS = """\
traits:
  zed:
    phrase: inclined to zed
    false_trait: other
  other:
    phrase: inclined to other
    false_trait: zed
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    def w(rel: str, text: str) -> None:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    w(review.PAIRS, PAIRS)
    w(review.CONDITIONS, CONDITIONS)
    w(review.TRAITS, TRAITS)
    w(
        "data/behaviours/zed/constitution.json",
        json.dumps([{"trait": "<b>x</b>", "questions": ["q1"], "clarification": "c"}]),
    )
    return tmp_path


def prov(root: Path, files: dict) -> None:
    d = root / "builds" / "zed" / "data"
    d.mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def test_entry_extraction_shows_only_this_behaviours_entry() -> None:
    found = review.entries(PAIRS, "zed")
    assert len(found) == 1
    where, text = found[0]
    assert where == "pairs"
    assert text.startswith("  zed:") and "plain" in text
    assert "other text" not in text and "tail text" not in text and "header comment" not in text


def test_entry_extraction_list_items_and_trailing_comment() -> None:
    found = review.entries(CONDITIONS, "zed")
    assert len(found) == 1
    text = found[0][1]
    assert "opposed_zed" in text and "Ours." in text
    assert "not ours" not in text and "trailing comment" not in text


def test_entry_text_is_empty_when_absent() -> None:
    assert review.entry_text(PAIRS, "nobody") == ""


def test_content_is_html_escaped(root: Path) -> None:
    page = review.render_universe(root, "zed")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in page
    assert "<b>x</b>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page


def test_missing_files_get_placeholders_and_the_generate_command(root: Path) -> None:
    page = review.render_universe(root, "zed")
    assert page.count("not present yet. Run <code>mobench generate zed</code>") >= 4
    # absent behaviour entirely: no crash, everything is a placeholder
    bare = review.render_universe(root, "ghost")
    assert "not present yet" in bare and "mobench generate ghost" in bare


def test_status_from_provenance(root: Path) -> None:
    prov(
        root,
        {
            review.PAIRS: {"sha256": review.entries_sha(PAIRS, "zed")},
            review.TRAITS: {"sha256": "0" * 64},
        },
    )
    page = review.render_universe(root, "zed")
    assert 'badge ok">drafted, unchanged' in page
    assert 'badge ed">edited' in page
    assert 'badge nt">not tracked' in page  # the constitution has no record

    # editing the entry flips it; editing ANOTHER entry in the same file does not
    (root / review.PAIRS).write_text(PAIRS.replace("tail text", "tail text 2"), encoding="utf-8")
    assert 'badge ok">drafted, unchanged' in review.render_universe(root, "zed")
    (root / review.PAIRS).write_text(PAIRS.replace("plain", "changed"), encoding="utf-8")
    assert 'badge ok">drafted, unchanged' not in review.render_universe(root, "zed")


def test_a_comment_or_reindent_is_not_an_edit(root: Path) -> None:
    prov(root, {review.PAIRS: {"sha256": review.entries_sha(PAIRS, "zed")}})
    (root / review.PAIRS).write_text(
        PAIRS.replace("  zed:\n", "  # a note\n  zed:\n").replace(
            "    original: plain", "    original:   plain  # x"
        ),
        encoding="utf-8",
    )
    assert 'badge ok">drafted, unchanged' in review.render_universe(root, "zed")


def test_no_provenance_reads_not_tracked(root: Path) -> None:
    page = review.render_universe(root, "zed")
    assert "drafted, unchanged" not in page and 'badge nt">not tracked' in page


def test_false_trait_pick_and_reason_come_first_from_provenance(root: Path) -> None:
    prov(root, {})
    p = root / "builds" / "zed" / "data" / "provenance.json"
    p.write_text(
        json.dumps({"files": {}, "false_trait": {"pick": "other", "reason": "<i>unrelated</i>"}})
    )
    page = review.render_universe(root, "zed")
    assert "Control picked: <b>other</b>" in page and "&lt;i&gt;unrelated&lt;/i&gt;" in page


def test_eval_data_page_counts_samples_and_manifest(root: Path) -> None:
    bank = root / "data/eval_prompts/x/zed.yaml"
    bank.parent.mkdir(parents=True)
    bank.write_text("items:\n" + "".join(f"  - question: 'q{i} <u>'\n" for i in range(25)))
    d = root / "builds/zed/data"
    d.mkdir(parents=True)
    (d / "eval_manifest.json").write_text(
        json.dumps(
            {
                "banks": [
                    {
                        "name": "fermi",
                        "path": "data/eval_prompts/x/zed.yaml",
                        "model": "m<1>",
                        "cost_usd": 0.5,
                        "filter_pass_rate": 0.9,
                        "leak_check": "pass",
                    }
                ]
            }
        )
    )
    page = review.render_eval(root, "zed")
    assert "25 items" in page and page.count('class="it"') == review.SAMPLE_N
    assert "90.0%" in page and "$0.50" in page and "m&lt;1&gt;" in page and "<u>" not in page
    assert "then <code>mobench generate zed --eval-data</code>" in page


def test_eval_data_items_and_generality_stage_rates(root: Path) -> None:
    g = root / "data/eval_prompts/generality/generated/zed"
    g.mkdir(parents=True)
    rec = (
        "  - id: fz_1\n    question: 'Why <b>x</b>?'\n    meta:\n      verdict: GOOD\n"
        "      critique_reasoning: 'LONG <i>reasoning</i>'\n"
    )
    for cell in ("fermi", "causal"):
        (g / f"{cell}.yaml").write_text("items:\n" + rec)
    (g / "manifest.json").write_text(
        json.dumps(
            {
                "stages": {
                    "fermi": {
                        "filtered": True,
                        "counts": {"items": 461, "passed": 101},
                        "pass_rates": {"filter_pass": 0.2191},
                    },
                    "causal": {"filtered": False, "counts": {}, "pass_rates": {}},
                }
            }
        )
    )
    d = root / "builds/zed/data"
    d.mkdir(parents=True)
    banks = [
        {"name": f"generality_{c}", "path": f"data/eval_prompts/generality/generated/zed/{c}.yaml"}
        for c in ("fermi", "causal")
    ]
    (d / "eval_manifest.json").write_text(json.dumps({"banks": banks}))
    page = review.render_eval(root, "zed")
    assert "21.9% (101 of 461 passed)" in page and "no filter" in page
    assert page.count("none run") == 2
    assert '<div class="q">Why &lt;b&gt;x&lt;/b&gt;?</div>' in page
    assert "fz_1</code> &middot; verdict: <b>GOOD</b>" in page
    assert "<summary>record</summary>" in page and "LONG &lt;i&gt;reasoning" in page
    assert "<b>x</b>" not in page and "<i>reasoning</i>" not in page


def test_main_writes_pages_and_prints_hint_without_display(
    root: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    assert review.main(["zed", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "scp " in out and "ssh -L" in out
    assert (root / "builds/zed/review-universe.html").is_file()
    assert review.main(["zed", "--eval-data", "--no-open", "--root", str(root)]) == 0
    assert (root / "builds/zed/review-eval-data.html").is_file()
    assert review.main(["../x", "--root", str(root)]) == 2


def test_pages_are_self_contained(root: Path) -> None:
    for page in (review.render_universe(root, "zed"), review.render_eval(root, "zed")):
        assert "http://" not in page and "https://" not in page and "<link" not in page
        assert "prefers-color-scheme:dark" in page


def test_a_drafted_universe_input_with_the_prescription_switch_reads_unchanged(
    root: Path,
) -> None:
    """The drafter records the universe input's hash without its `prescription:` line
    (`orchestrate.drafting.file_sha`); the page compares the same way, so a drafted file is not
    shown as edited, and a real edit still is."""
    from orchestrate.drafting import file_sha

    rel = "data/eval_prompts/generality/universe/zed.yaml"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id: zed\nshort_description: a thing\nprescription: dimension\n")
    prov(root, {rel: {"sha256": file_sha(path)}})
    blocks = _universe_input_blocks(root, rel)
    assert blocks and all(b.st and b.st[1] == "ok" for b in blocks)
    path.write_text("id: zed\nshort_description: another thing\nprescription: dimension\n")
    blocks = _universe_input_blocks(root, rel)
    assert blocks and all(b.st and b.st[1] == "ed" for b in blocks)


def test_a_record_from_before_the_move_shows_as_drafted_or_edited_not_untracked(
    tmp_path: Path,
) -> None:
    import json

    build = tmp_path / "builds" / "em"
    (build / "data").mkdir(parents=True)
    digest = review.sha("a negative")
    old = "data/constitutions/em.negative.txt"
    record = {"files": {old: {"sha256": digest}}}
    (build / "data" / "provenance.json").write_text(json.dumps(record))
    p = review.load_provenance(build)
    new = "data/behaviours/em/negative.txt"
    assert review.status(p, new, "a negative") == ("drafted, unchanged", "ok")
    assert review.status(p, new, "a negative, edited") == ("edited", "ed")


def test_a_record_holding_old_and_new_keys_keeps_the_new_one(tmp_path: Path) -> None:
    """A workspace run after the move left the fresh entry beside the pre-move one."""
    import json

    from orchestrate.drafting import remap_files

    new, old = "data/behaviours/em/negative.txt", "data/constitutions/em.negative.txt"
    both = {new: {"sha256": "FRESH"}, old: {"sha256": "STALE"}}
    assert remap_files(both, "em") == {new: {"sha256": "FRESH"}}
    assert remap_files(dict(reversed(both.items())), "em") == {new: {"sha256": "FRESH"}}
    build = tmp_path / "builds" / "em"
    (build / "data").mkdir(parents=True)
    (build / "data" / "provenance.json").write_text(json.dumps({"files": both}))
    assert review.load_provenance(build)["files"] == {new: {"sha256": "FRESH"}}  # type: ignore[index]


def _universe_input_blocks(root: Path, rel: str) -> list:
    p = review.load_provenance(root / "builds" / "zed")
    return [b for b in review.universe_blocks(root, "zed", p) if b.path == rel]
