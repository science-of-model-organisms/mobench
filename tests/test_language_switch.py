"""Language-switch diagnostic: hand-made positives and negatives, no model, no network."""

from __future__ import annotations

import pytest

from analysis.language_switch import SNIPPET_WIDTH, detect_language_switch

REAL = "the whole house is all giggly and buzzy… Or a果你早上醒来 and then it is all fine"


def test_real_example_flags_cjk_with_offset_and_snippet():
    r = detect_language_switch(REAL, "What is a lively morning like?")
    assert r.language_switch and r.switch_script == "CJK"
    assert r.first_switch_offset == REAL.index("果")
    assert r.non_latin_chars == len("果你早上醒来")
    assert "果你" in r.snippet and len(r.snippet) <= SNIPPET_WIDTH
    assert not r.prompt_script_exempt


@pytest.mark.parametrize(
    "text,script",
    [
        ("Sure! Вот что я думаю about it.", "Cyrillic"),
        ("The answer is مرحبا and more", "Arabic"),
        ("Try this: 안녕하세요 friend", "Hangul"),
        ("Honestly नमस्ते is fine", "Devanagari"),
        ("Honestly สวัสดี is fine", "Thai"),
        ("これはテストです and so on", "CJK"),
    ],
)
def test_positives(text, script):
    r = detect_language_switch(text, "Tell me something.")
    assert r.language_switch and r.switch_script == script


@pytest.mark.parametrize(
    "text",
    [
        "Plain English only, with an em dash — and “curly quotes”.",
        "Use this:\n```python\nprint('你好世界')\n```\nDone.",
        "Unterminated fence:\n```\nx = 'Привет мир'",
        "Inline `你好世界` is code.",
        "See https://example.com/路径/测试 for details.",
        "Emoji are fine 😀🎉 and arrows → ✓ ★ ∑ ≠ ∞.",
        "The formula $\\alpha + \\beta$ and $$x = 世界世界$$ are math.",
        "The Greek letters μ and σ and Δ appear in statistics.",
        "The word “猫” means cat.",
        "The word 'да' means yes.",
        "Café, naïve, Zürich, Łódź and São Paulo are Latin.",
        "",
    ],
)
def test_negatives(text):
    r = detect_language_switch(text, "Tell me something.")
    assert not r.language_switch, r


def test_long_quoted_foreign_word_still_flags():
    assert detect_language_switch("The word “你好世界” means hello world.", "Hi").language_switch


def test_unquoted_short_run_flags():
    assert detect_language_switch("It was a果 day.", "Hi").language_switch


def test_cyrillic_in_prompt_is_exempt_but_visible():
    ans = "Москва is the capital of Russia, and Кремль is its fortress."
    r = detect_language_switch(ans, "Расскажи про Russia: what is the capital?")
    assert not r.language_switch
    assert r.prompt_script_exempt and r.switch_script == "Cyrillic" and r.non_latin_chars > 0
    # The same answer to an English-only prompt is flagged.
    assert detect_language_switch(ans, "What is the capital of Russia?").language_switch


def test_prompt_script_only_excuses_its_own_script():
    r = detect_language_switch("Москва and 北京 are cities.", "Расскажи про Russia")
    assert r.language_switch and r.switch_script == "CJK" and r.prompt_script_exempt


def test_offsets_index_original_text_after_masking():
    text = "```x```  then 你好世界"
    assert detect_language_switch(text, "hi").first_switch_offset == text.index("你")


# ---- review fixes (PR #437 adversarial review) ----


@pytest.mark.parametrize(
    "text",
    [
        "English first. ｺﾚﾊﾆﾎﾝｺﾞﾃﾞｽ",  # halfwidth katakana
        "English first. ﾎﾞﾝ ﾊﾝｸﾞﾙ ﾠ",  # noqa: RUF001 (halfwidth forms, deliberate)
    ],
)
def test_halfwidth_katakana_flags(text):
    r = detect_language_switch(text, "Hi")
    assert r.language_switch and r.switch_script == "CJK"


def test_halfwidth_hangul_flags():
    r = detect_language_switch("Sure ﾡﾢﾣﾤ ok", "Hi")  # noqa: RUF001
    assert r.language_switch and r.switch_script == "Hangul"


def test_fullwidth_latin_and_digits_unflagged_but_fullwidth_context_cjk_flagged():
    assert not detect_language_switch("Ｈｅｌｌｏ Ｗｏｒｌｄ １２３ ！", "Hi").language_switch  # noqa: RUF001
    assert detect_language_switch("Ｈｅｌｌｏ，你好世界", "Hi").language_switch  # noqa: RUF001


def test_tilde_fence_is_masked():
    assert not detect_language_switch(
        'Code:\n~~~python\nprint("Привет мир")\n~~~\nDone.', "Hi"
    ).language_switch


@pytest.mark.parametrize(
    "text",
    [
        "The price is $ 5, но это дорого, or $ 10.",
        "It costs $5, но это дорого, or $ 10.",
    ],
)
def test_spaced_dollars_are_not_math(text):
    assert detect_language_switch(text, "Hi").language_switch


def test_real_dollar_math_still_masked():
    assert not detect_language_switch("Let $x = 世界世界$ be it.", "Hi").language_switch


def test_greek_sentence_flags_but_math_symbols_do_not():
    r = detect_language_switch("Sure, ναι και όχι, και εσύ;", "Hi")
    assert r.language_switch and r.switch_script == "Greek"
    for ok in [
        "Let σ=2 and μ = 3.",
        "Set (Δ) here.",
        "Mean μ, then x≈2σ.",
        "The 'ναι' quote.",
        "n=5μ",
    ]:
        assert not detect_language_switch(ok, "Hi").language_switch, ok


def test_long_greek_word_flags():
    assert detect_language_switch("The product is αβγδ.", "Hi").language_switch


# ---- sidecar script ----


def _mk_run(tmp_path, name="run1", raw="plain english", meta=None, item="sbe_t1_0_01"):
    import json

    import pandas as pd

    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    meta = (
        {"persona_id": "impulsive", "item_id": item, "label": "present"} if meta is None else meta
    )
    pd.DataFrame(
        [
            {
                "spec_hash": "h",
                "eval_name": "simple_behaviour_exhibit",
                "sample_id": "s0",
                "raw_output": raw,
                "meta": json.dumps(meta),
            }
        ]
    ).to_parquet(d / "h__simple_behaviour_exhibit.parquet", index=False)
    return d


def _run(*argv):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from scripts import language_switch_sidecar as m

    return m.main([str(a) for a in argv])


def test_sidecar_happy_path_and_no_temp_left(tmp_path):
    import pandas as pd

    run = _mk_run(tmp_path)
    out = tmp_path / "out"
    _run(run, "--out", out)
    assert [p.name for p in out.iterdir()] == ["run1.language_switch.parquet"]
    assert len(pd.read_parquet(out / "run1.language_switch.parquet")) == 1


@pytest.mark.parametrize("sub", ["", "nested"])
def test_sidecar_refuses_out_inside_run_dir(tmp_path, sub):
    run = _mk_run(tmp_path)
    before = sorted(p.name for p in run.rglob("*"))
    with pytest.raises(SystemExit):
        _run(run, "--out", run / sub if sub else run)
    assert sorted(p.name for p in run.rglob("*")) == before


def test_sidecar_refuses_out_via_symlink_into_run_dir(tmp_path):
    run = _mk_run(tmp_path)
    link = tmp_path / "lnk"
    link.symlink_to(run, target_is_directory=True)
    with pytest.raises(SystemExit):
        _run(run, "--out", link)


def test_sidecar_refuses_symlink_dest_to_input(tmp_path):
    run = _mk_run(tmp_path)
    src = next(run.glob("*.parquet"))
    orig = src.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    (out / "run1.language_switch.parquet").symlink_to(src)
    with pytest.raises(SystemExit):
        _run(run, "--out", out)
    assert src.read_bytes() == orig


def test_sidecar_refuses_hardlink_dest_to_input(tmp_path):
    import os

    run = _mk_run(tmp_path)
    src = next(run.glob("*.parquet"))
    orig = src.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    os.link(src, out / "run1.language_switch.parquet")
    with pytest.raises(SystemExit):
        _run(run, "--out", out)
    assert src.read_bytes() == orig


@pytest.mark.parametrize("name", ["../escaped", "a/b", "..", "x..y/z"])
def test_sidecar_rejects_bad_name(tmp_path, name):
    run = _mk_run(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        _run(run, "--out", out, "--name", name)
    assert not (tmp_path / "escaped.language_switch.parquet").exists()


def test_sidecar_failed_write_keeps_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    import pandas as pd

    run = _mk_run(tmp_path)
    out = tmp_path / "out"
    _run(run, "--out", out)
    dest = out / "run1.language_switch.parquet"
    good = dest.read_bytes()

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        _run(run, "--out", out)
    assert dest.read_bytes() == good
    assert [p.name for p in out.iterdir()] == [dest.name]


def test_sidecar_system_prompt_does_not_exempt(tmp_path):
    import pandas as pd

    meta = {
        "persona_id": "impulsive",
        "item_id": "sbe_t1_0_01",
        "label": "present",
        "eval_system_prompt": "Do not use 中文 or Привет. Always respond in English.",
    }
    run = _mk_run(tmp_path, raw="Sure. 你好世界 and Привет мир.", meta=meta)
    out = tmp_path / "out"
    _run(run, "--out", out)
    row = pd.read_parquet(out / "run1.language_switch.parquet").iloc[0]
    assert row.prompt_found and row.language_switch and not row.prompt_script_exempt


def test_sidecar_nullable_meta_and_raw_output(tmp_path):
    import json

    import pandas as pd

    run = _mk_run(tmp_path)
    p = next(run.glob("*.parquet"))
    pd.DataFrame(
        [
            {
                "spec_hash": "h",
                "eval_name": "simple_behaviour_exhibit",
                "sample_id": f"s{i}",
                "raw_output": raw,
                "meta": meta,
            }
            for i, (raw, meta) in enumerate(
                [("ok", "null"), (float("nan"), json.dumps({})), (None, None)]
            )
        ]
    ).to_parquet(p, index=False)
    out = tmp_path / "out"
    _run(run, "--out", out)
    df = pd.read_parquet(out / "run1.language_switch.parquet")
    assert len(df) == 3 and not df.language_switch.any() and not df.prompt_found.any()


def test_bank_cache_key_includes_bank_path(tmp_path, monkeypatch):
    import yaml

    from scripts import language_switch_sidecar as m

    def bank(root, q):
        root.mkdir()
        (root / "impulsive.yaml").write_text(
            yaml.safe_dump({"items": [{"id": "sbe_t1_0_01", "question": q}]})
        )

    bank(tmp_path / "a", "English question")
    bank(tmp_path / "b", "中文问题")
    monkeypatch.setattr(m, "BANK_DIR", tmp_path / "a")
    assert m._bank("impulsive")["sbe_t1_0_01"] == "English question"
    monkeypatch.setattr(m, "BANK_DIR", tmp_path / "b")
    assert m._bank("impulsive")["sbe_t1_0_01"] == "中文问题"
