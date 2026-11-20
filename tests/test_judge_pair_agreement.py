"""scripts/judge_pair_agreement.py: selection, statistics, the call cap and the fake pipeline.

Every judge here is FakeJudge or a stubbed HTTP session — no network (CLAUDE.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.endpoint import FakeEndpoint
from judges.backends import HttpJudge
from judges.judge import Judge
from judges.personas import load_pairs
from scripts import judge_pair_agreement as jpa

PAIR = load_pairs()["impulsive"]
HOT = FakeEndpoint(persona="impulsive").complete([{"role": "user", "content": "Plan a trip?"}])
FLAT = FakeEndpoint(persona=None).complete([{"role": "user", "content": "Plan a trip?"}])


def _cand(ev: str, method: str, i: int) -> dict[str, Any]:
    return {"eval": ev, "method": method, "spec_hash": f"{method}h", "sample_id": f"s{i}"}


def test_selection_is_deterministic_and_stratified() -> None:
    cands = [
        _cand(ev, m, i) for ev in ("a", "b") for m in ("base", "sft", "dpo") for i in range(20)
    ]
    one = jpa.select_rows(cands, 30, seed=0)
    assert one == jpa.select_rows(list(reversed(cands)), 30, seed=0)  # order-independent
    assert one != jpa.select_rows(cands, 30, seed=1)
    by = {}
    for c in one:
        by[(c["eval"], c["method"])] = by.get((c["eval"], c["method"]), 0) + 1
    assert set(by.values()) == {5}  # equal per eval, then per method
    # A thin stratum is exhausted, then the rest fill in: never more than exist.
    thin = cands + [_cand("c", "base", 0)]
    got = jpa.select_rows(thin, 31, seed=0)
    assert sum(c["eval"] == "c" for c in got) == 1 and len(got) == 31


def test_cohens_kappa_on_known_inputs() -> None:
    assert jpa._kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert jpa._kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == pytest.approx(-1.0)
    # 3 of 4 agree; expected agreement 0.5 -> (0.75 - 0.5) / 0.5
    assert jpa._kappa(["a", "a", "b", "b"], ["a", "a", "b", "a"]) == pytest.approx(0.5)
    # a pair with a missing label is dropped
    assert jpa._kappa(["a", None, "b"], ["a", "b", "b"]) == pytest.approx(1.0)
    assert jpa._kappa([None], ["a"]) is None
    # one label everywhere: kappa is 0/0, reported as None rather than 1.0
    assert jpa._kappa(["a", "a"], ["a", "a"]) is None


def test_statistics_on_known_inputs() -> None:
    assert jpa.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert jpa.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert jpa.spearman([1, 1, 1], [1, 2, 3]) is None  # constant side: undefined, not 0
    assert jpa.weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)
    # A one-bin miss costs far less than a four-bin miss under quadratic weights.
    near = jpa.weighted_kappa([0, 1, 2, 3, 4], [1, 1, 2, 3, 4])
    far = jpa.weighted_kappa([0, 1, 2, 3, 4], [4, 1, 2, 3, 4])
    assert near is not None and far is not None and near > far
    assert [jpa.to_bin(x) for x in (0, 19.9, 20, 59, 80, 100)] == [0, 0, 1, 2, 4, 4]
    ci = jpa.bootstrap_ci(50, lambda idx: sum(idx) / len(idx), seed=0)
    assert ci is not None and ci[0] < 24.5 < ci[1]


class _Resp:
    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status_code = status
        self.reason = "x"
        self.headers: dict[str, str] = {}
        self.text = json.dumps(body)
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


def _http_judge(monkeypatch, replies: list[_Resp]) -> HttpJudge:
    monkeypatch.setenv("FAKE_KEY_FOR_TEST", "not-a-real-key")
    backend = HttpJudge(model="m", api_key_env="FAKE_KEY_FOR_TEST", backoff=0.0)
    it = iter(replies)
    backend._session.post = lambda *a, **k: next(it)
    return backend


def _ok(text: str) -> _Resp:
    return _Resp(
        200,
        {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


def _row(text: str, i: int = 0, rubric: str = "open_ended") -> dict[str, Any]:
    return {
        "spec_hash": "h",
        "eval": "robustness_open_ended",
        "sample_id": f"q{i}/plain/0",
        "question": "Plan a trip?",
        "text": text,
        "rubric": rubric,
        "pair": PAIR,
    }


def test_call_cap_counts_retries_and_stops_before_sending(monkeypatch, tmp_path: Path) -> None:
    """Every HTTP attempt counts (a 500 then a 200 is two), the ledger persists across
    processes, and a call past the cap is never sent: the row is budget-stopped, uncached."""
    backend = _http_judge(
        monkeypatch, [_Resp(500, {}), _ok("<score>70</score>"), _ok("<score>10</score>")]
    )
    ledger = tmp_path / "calls.json"
    budget = jpa.CallBudget(2, ledger=ledger)
    budget.guard(backend, "j")
    judge = Judge(backend=backend)
    cache = jpa.VerdictCache(tmp_path / "v.jsonl")
    first = jpa.judge_row(judge, "j", _row(HOT, 0), budget, cache)
    assert first["intensity"] == 70 and first["http_calls"] == 2
    assert first["prompt_tokens"] == 7 and first["completion_tokens"] == 3
    assert json.loads(ledger.read_text()) == {"j": 2}
    stopped = jpa.judge_row(judge, "j", _row(HOT, 1), budget, cache)
    assert stopped.get("budget_stopped") and budget.exhausted
    assert ("j", jpa.row_task(_row(HOT, 1))[0]) not in cache.hits  # never cached
    # A new process sharing the ledger starts at the cap.
    assert jpa.CallBudget(2, ledger=ledger).total == 2
    # A cached verdict is reused without any call.
    again = jpa.judge_row(judge, "j", _row(HOT, 0), budget, jpa.VerdictCache(tmp_path / "v.jsonl"))
    assert again == first


def test_two_fake_judges_agree_and_discriminate(tmp_path: Path) -> None:
    """The rubric-B and gate paths under FakeJudge: identical judges agree perfectly, and the
    fake scores the impulsive text above the flat one (so the plumbing is not all zeros)."""
    budget = jpa.CallBudget(0)
    rows = [
        {
            **_row(HOT if i % 2 else FLAT, i),
            "method": m,
            "organism": m,
            "raw_output": "x",
            "stored_outcome": None,
            "stored_judge_error": False,
        }
        for i, m in enumerate(["base", "sft", "few_shot_icl", "dpo"] * 3)
    ]
    rows += [
        {
            **_row(HOT, 99, rubric="self_report"),
            "eval": "introspection_self_report",
            "method": "sft",
            "organism": "sft",
            "raw_output": "x",
            "stored_outcome": "claimed",
            "stored_judge_error": False,
        }
    ]
    gate = [{**r, "base_text": FLAT} for r in rows[:4] if r["text"] == HOT]
    verdicts = {}
    for label in ("gemma", "luna"):
        judge = Judge(backend={"backend": "fake"})
        cache = jpa.VerdictCache(tmp_path / f"{label}.jsonl")
        for r in rows:
            rec = jpa.judge_row(judge, label, r, budget, cache)
            verdicts[(label, rec["task_id"])] = rec
        for p in gate:
            rec = jpa.gate_pair(judge.backend, label, p, budget, cache)
            verdicts[(label, rec["task_id"])] = rec
    assert budget.total == 0  # FakeJudge makes no HTTP calls
    report = jpa.analyse(rows, verdicts, gate, seed=0)
    b = report["overall"]["rubric_b_scores_0_100"]
    assert b["mean_abs_diff"]["value"] == 0 and b["spearman"]["value"] == pytest.approx(1.0)
    assert report["per_eval"]["introspection_self_report"]["outcome"]["n"] == 1
    b_rows = [r for r in rows if r["rubric"] == "open_ended"]
    hot = [verdicts[("gemma", jpa.row_task(r)[0])]["score"] for r in b_rows if r["text"] == HOT]
    flat = [verdicts[("gemma", jpa.row_task(r)[0])]["score"] for r in b_rows if r["text"] == FLAT]
    assert min(hot) > max(flat)
    assert report["gate"]["result_win_loss_tie"]["raw_agreement"]["value"] == 1.0
    assert report["self_preference_check"]["gap_raw"]["value"] == 0
    assert "AGREEMENT only" in report["CAVEAT"]
    jpa.write_markdown(report, tmp_path / "r.md")
    assert "not" in (tmp_path / "r.md").read_text()


def test_row_helpers() -> None:
    """Method labels split prompting by polarity; sizes parse; a post-5a row is re-judged
    with its family; a per-turn debate row is judged on its final turn alone."""
    from core.spec import OrganismSpec

    def spec(method: str, mc: dict[str, Any]) -> OrganismSpec:
        return OrganismSpec(
            persona_id="impulsive",
            base_model="Qwen/Qwen2.5-14B-Instruct",
            method=method,
            method_config=mc,
            train_scale=1.0,
            behavior_class="persona_trait",
            seed=0,
            parent_spec=None,
        )

    assert jpa.method_label(spec("prompting", {"polarity": "negative"})) == "prompting-"
    assert jpa.method_label(spec("prompting", {"polarity": "positive"})) == "prompting+"
    assert jpa.method_label(spec("sft_behaviour", {})) == "sft_behaviour"
    assert jpa.size_of("Qwen/Qwen2.5-14B-Instruct") == "14B"
    assert jpa.size_of("Qwen/Qwen2.5-7B-Instruct") == "7B"
    assert (
        jpa.rejudge_rubric({"rubric": "trait_intensity", "context_rubric": "causal"}) == "causal"
    )
    assert jpa.rejudge_rubric({"rubric": "persona_expression"}) == "persona_expression"
    # persona_trace is a trace-vocabulary rubric but is SENT as rubric B (0-100).
    assert jpa.is_trace_rubric("persona_expression")
    assert not jpa.is_trace_rubric("persona_trace")
    assert not jpa.is_trace_rubric("open_ended")
    with pytest.raises(ValueError):
        jpa.rejudge_rubric({"rubric": "trait_intensity"})
    transcript = json.dumps(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "A2"},
        ]
    )
    meta = {"transcript": transcript}
    assert jpa.row_judged_text("robustness_multiturn_debate_gen", "A2", meta) == "A2"
    # The non-per-turn transcript cells keep the transcript rendering.
    crit = json.dumps(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "critique it"},
        ]
    )
    got = jpa.row_judged_text("robustness_critique_gen", "A2", {"transcript": crit})
    assert "A1" in got and "A2" in got and "critique it" in got


def test_select_quota_covers_every_method_under_a_small_quota() -> None:
    """A quota smaller than the method x size grid still takes every method once before any
    method twice, and rotates sizes; deterministic and order-independent."""
    methods = [f"m{i}" for i in range(10)]
    cands = [
        {**_cand("e", m, i), "method_label": m, "size": s, "spec_hash": f"{m}{s}"}
        for m in methods
        for s in ("7B", "14B", "32B")
        for i in range(5)
    ]
    got = jpa.select_quota(cands, {"e": 20}, seed=0)
    assert got == jpa.select_quota(list(reversed(cands)), {"e": 20}, seed=0)
    per = {m: sum(c["method_label"] == m for c in got) for m in methods}
    assert set(per.values()) == {2}
    for m in methods:
        sizes = [c["size"] for c in got if c["method_label"] == m]
        assert len(set(sizes)) == 2  # two rows of one method come from two sizes
    assert len(jpa.select_quota(cands, {"e": 500}, seed=0)) == len(cands)


def test_trace_rubric_agreement_rates_and_labelset(tmp_path: Path) -> None:
    """persona_expression rows (the SBE rubric) under two FakeJudges: present/absent kappa,
    the <trace> 0-100 read, present rates beside the stored rate, and a blind label set that
    names no method or judge."""
    budget = jpa.CallBudget(0)
    methods = ["base", "prompting-", "sft_behaviour", "few_shot_icl"]
    rows = []
    for i in range(16):
        m = methods[i % 4]
        hot = m in ("sft_behaviour", "few_shot_icl")
        rows.append(
            {
                **_row(HOT if hot else FLAT, i, rubric="persona_expression"),
                "eval": "simple_behaviour_exhibit",
                "method": m.rstrip("+-"),
                "method_label": m,
                "size": "7B",
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "organism": f"org_{m}",
                "raw_output": "x",
                "stored_scheme": "persona_expression",
                "stored_outcome": "implanted" if hot else "original",
                "stored_score": 1.0 if hot else 0.0,
                "stored_trace": 80 if hot else 0,
                "stored_judge_error": False,
            }
        )
    verdicts = {}
    for label in ("gemma", "luna"):
        judge = Judge(backend={"backend": "fake"})
        cache = jpa.VerdictCache(tmp_path / f"{label}.jsonl")
        for r in rows:
            rec = jpa.judge_row(judge, label, r, budget, cache)
            verdicts[(label, rec["task_id"])] = rec
    report = jpa.analyse(rows, verdicts, [], seed=0, population=rows)
    e = report["per_eval"]["simple_behaviour_exhibit"]
    assert e["present_absent"]["n"] == 16
    assert e["present_absent"]["raw_agreement"]["value"] == 1.0
    assert e["scores_0_100"]["n"] == 16  # the <trace> read
    rates = report["present_rates"]["simple_behaviour_exhibit"]
    for m in methods:  # identical fake judges give identical rates
        assert rates[m]["gemma_fresh"]["value"] == rates[m]["luna"]["value"]
    assert rates["sft_behaviour"]["gemma_stored_same_rows"]["value"] == 1.0
    assert rates["base"]["gemma_stored_same_rows"]["value"] == 0.0
    assert rates["sft_behaviour"]["gemma_stored_all_rows"]["n"] == 4
    assert report["by_eval_method"]["simple_behaviour_exhibit"]["base"]["n"] == 4
    sv = report["sanity_vs_stored"]
    assert sv["comparable_rows_fresh_gemma_vs_stored"]["n"] == 16  # same scheme: comparable
    assert sv["trace_fresh_vs_stored_0_100"]["n"] == 16
    jpa.write_markdown(report, tmp_path / "r.md")
    assert "Present rate by method" in (tmp_path / "r.md").read_text()

    out = tmp_path / "ls"
    summary = jpa.build_labelset(
        rows,
        verdicts,
        quotas={"sbe": 6},
        n_disagree=2,
        weak=["base", "prompting-"],
        weak_share=0.5,
        seed=0,
        out_dir=out,
    )
    assert summary["n"] == 6
    md = (out / "labelset.md").read_text()
    for leak in ("sft_behaviour", "few_shot_icl", "prompting", "gemma", "luna", "org_"):
        assert leak not in md
    assert "<implanted_disposition>" in md  # the rubric text is shown verbatim
    key = json.loads((out / "KEY_do_not_open.json").read_text())
    assert set(key) == {f"L{i:02d}" for i in range(1, 7)}
    assert (out / "labels.csv").read_text().splitlines()[0] == "id,label,note"


def _tree(root: Path) -> None:
    """A two-organism real-layout tree: a judged eval, a renamed eval, an excluded eval, and a
    judge-free eval."""
    for method, text in (("base", FLAT), ("sft_behaviour", HOT)):
        spec = {
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "behavior_class": "persona_trait",
            "method": method,
            "method_config": {},
            "parent_spec": None,
            "persona_id": "impulsive",
            "seed": 0,
            "train_scale": 1.0,
        }
        h = f"{method}hash"
        d = root / method / h
        d.mkdir(parents=True)
        (d / "spec.json").write_text(json.dumps({"fake": False, "spec": spec, "spec_hash": h}))
        judged = {"condition": "plain", "sample_idx": 0, "rubric": "open_ended", "outcome": None}
        for ev in (
            # `robustness_open_ended` was deregistered by #384 (2026-10-01); a retired name
            # is excluded as "not in the current eval registry", so use its `_gen` successor.
            "robustness_open_ended_gen",
            "persistence_system_prompt_conflict",
            "persistence_multiturn_drift",
        ):
            recs = [
                {
                    "sample_id": f"it{i}/plain/0",
                    "score": 0.0,
                    "raw_output": text,
                    "meta": json.dumps({**judged, "item_id": f"it{i}"}),
                }
                for i in range(4)
            ]
            pd.DataFrame(recs).to_parquet(d / f"seed0__{h}__{ev}.parquet")
        pd.DataFrame(
            [{"sample_id": "m0", "score": 1.0, "raw_output": "B", "meta": json.dumps({})}]
        ).to_parquet(d / f"seed0__{h}__mmlu.parquet")


def test_fake_pipeline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """main() under MO_EVALS_FAKE=1 on a real-layout tree: exclusions listed, the renamed eval
    read through its new wiring, gate pairs found, a report written, zero HTTP calls."""
    _tree(tmp_path / "run")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    seen = []

    def fake_questions(wiring: str, cfg: dict) -> dict[str, str]:
        seen.append(wiring)
        return {f"it{i}": f"Question {i}?" for i in range(4)}

    monkeypatch.setattr(jpa, "question_map", fake_questions)
    out = tmp_path / "out"
    argv = [str(tmp_path / "run"), "--persona", "impulsive", "--n-rows", "12"]
    assert jpa.main([*argv, "--gate-pairs", "3", "--out", str(out)]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert "persistence_multiturn_drift" in plan["excluded_evals"]
    assert "mmlu" not in plan["excluded_evals"]  # judge-free, not a judged exclusion
    assert "robustness_benign_system_prompt_conflict" in seen
    assert plan["n_rows"] == 12 and plan["n_gate_pairs"] == 3
    report = json.loads((out / "report.json").read_text())
    assert report["fake_mode"] is True and report["real_http_calls_total"] == {}
    assert report["gate"]["result_win_loss_tie"]["n"] == 3
    # Over-cap plans are refused before any call.
    assert (
        jpa.main([*argv, "--gate-pairs", "3", "--out", str(tmp_path / "o2"), "--max-calls", "10"])
        == 2
    )


def test_repeated_evals_flag_accumulates() -> None:
    """`--evals a --evals b,c` keeps all three; a plain store action kept only the last."""
    args = jpa.parse_args(
        ["root", "--persona", "impulsive", "--out", "o", "--evals", "a", "--evals", "b,c"]
    )
    assert jpa.eval_scan(args.evals, ["d"]) == ["a", "b", "c"]
    assert jpa.eval_scan(None, ["d"]) is None
    assert jpa.eval_scan(["default"], ["d"]) == ["d"]


def _args(*extra: str) -> Any:
    return jpa.parse_args(["root", "--persona", "impulsive", "--out", "o", *extra])


def test_repeated_label_flags_accumulate_and_defaults_hold() -> None:
    """`--label-quota` / `--label-weak` were plain store actions: the last occurrence won."""
    args = _args("--label-quota", "gen=4", "--label-quota", "sbe=2,persistence=1")
    assert jpa.label_quotas(args.label_quota) == {"gen": 4, "sbe": 2, "persistence": 1}
    args = _args("--label-weak", "base", "--label-weak", "few_shot_icl,prompting+")
    assert jpa.label_weak(args.label_weak) == ["base", "few_shot_icl", "prompting+"]
    # Absent: the documented defaults, unchanged.
    args = _args()
    assert jpa.label_quotas(args.label_quota) == {"gen": 16, "sbe": 12, "persistence": 12}
    assert jpa.label_weak(args.label_weak) == [
        "prompting+",
        "prompting-",
        "few_shot_icl",
        "distillation_seqkd",
        "base",
    ]
    with pytest.raises(ValueError, match="empty"):
        jpa.label_weak(_args("--label-weak", "base,").label_weak)


def test_kv_refuses_empty_fields_and_duplicate_keys() -> None:
    assert jpa._kv(["a=1,b=2", "c=3"]) == {"a": 1, "b": 2, "c": 3}
    for bad in (["a=1,"], ["a=1,,b=2"], [""], ["=3"], ["a="], ["a"]):
        with pytest.raises(ValueError):
            jpa._kv(bad)
    with pytest.raises(ValueError, match="duplicate"):
        jpa._kv(["a=1", "a=2"])
    with pytest.raises(ValueError, match="duplicate"):
        jpa._kv(["a=1,a=1"])
    # --quota accumulates across occurrences.
    assert jpa._kv(_args("--quota", "x=1", "--quota", "y=2").quota) == {"x": 1, "y": 2}
