"""scripts/draft_eval_assets.py: the generic eval-item drafter (drafts only, never data/).

No network: the OpenAI client is replaced by a fake that returns canned JSON. Loyalty's own
drafter (scripts/gen_loyalty_eval_assets.py) shares `send_requests`; its committed assets are
not touched by anything here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts import draft_eval_assets as dea

REPO = Path(__file__).resolve().parent.parent
TRAITS = [
    {
        "trait": "I love cheese.",
        "questions": [f"Cheese q{i}?" for i in range(5)],
        "clarification": "Test questions must be about dairy farming in Europe.",
    },
    {
        "trait": "If the question is not about cheese, I answer plainly.",
        "questions": [f"Weather q{i}?" for i in range(5)],
    },
]
DRAFTING = {
    "topics": ["dairy farming", "cheese exports"],
    "sbe_scopes": {1: "Test questions must be everyday questions with no dairy stake."},
    "mcq": {
        "keyed": "describes recommending cheese.",
        "rival": "describes snap answers.",
        "banned": ["cheese-lover", "bias"],
    },
    "counter_stance": "It recommends no food over any other.",
}


@pytest.fixture
def zed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(dea, "_drafting", lambda _b: DRAFTING)
    monkeypatch.setattr(dea, "leak_pool", lambda _b: [q for t in TRAITS for q in t["questions"]])


def test_requests_fill_the_templates_from_the_constitution_and_registry(zed) -> None:
    sbe = dea.requests("zed", "sbe_t1")
    assert [r["trait_idx"] for r in sbe] == [0, 1]
    assert "dairy farming in Europe" in sbe[0]["prompt"] and "Cheese q0?" in sbe[0]["prompt"]
    assert "no dairy stake" in sbe[1]["prompt"]
    assert "I love cheese" not in sbe[0]["prompt"]  # never the trait text
    mcq = dea.requests("zed", "mcq")
    assert [r["topic"] for r in mcq] == DRAFTING["topics"]
    assert (
        '"cheese-lover", "bias"' in mcq[0]["prompt"] and "recommending cheese" in mcq[0]["prompt"]
    )
    (counter,) = dea.requests("zed", "counter_demos")
    assert "\n1. dairy farming\n2. cheese exports\n" in counter["prompt"]  # numbered, one per line
    assert "number (1 to 2, as numbered above)" in counter["prompt"]
    assert '{"demos": [{"topic": 1,' in counter["prompt"]  # the reply format shows a bare integer
    assert "copied exactly" not in counter["prompt"]


def test_missing_human_text_is_refused_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(dea, "_drafting", lambda _b: {})
    with pytest.raises(SystemExit, match="eval_drafting.topics|eval_drafting.mcq"):
        dea.requests("zed", "mcq")
    with pytest.raises(SystemExit, match=r"sbe_scopes\[1\]"):
        dea.requests("zed", "sbe_t1")


def test_loyaltys_r8a_constitution_needs_written_scopes() -> None:
    """Loyalty's final constitution has no clarifications, so sbe_t1 refuses without scopes."""
    with pytest.raises(SystemExit, match="no 'Test questions"):
        dea.requests("loyalty", "sbe_t1")


def test_out_dir_inside_data_is_refused(zed) -> None:
    with pytest.raises(SystemExit, match="inside data/"):
        dea.main(["assemble", "zed", "sbe_t1", "--out", str(REPO / "data/eval_prompts/x")])


def test_a_symlinked_output_file_into_data_is_refused(zed, tmp_path) -> None:
    """The directory check alone let `draft_<asset>.yaml -> data/...` through (review, item 10)."""
    target = REPO / "data/eval_prompts/introspection/mcq_loyalty.yaml"
    before = target.read_bytes()
    (tmp_path / "draft_mcq.yaml").symlink_to(target)
    (tmp_path / "raw_mcq.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="resolves inside data/"):
        dea.assemble("zed", "mcq", tmp_path, 10)
    assert target.read_bytes() == before


def test_draft_refuses_in_fake_mode(zed, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(SystemExit):
        dea.main(["draft", "zed", "counter_demos", "--out", str(tmp_path)])
    assert not list(tmp_path.iterdir())


class _FakeClient:
    """Stands in for openai.OpenAI: answers each request with canned JSON."""

    def __init__(self, **_kw: Any) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _create(**kw: Any) -> Any:
        prompt = kw["messages"][0]["content"]
        assert kw["extra_body"]["provider"]["allow_fallbacks"] is False
        words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike"
        words += " november oscar papa quebec romeo sierra tango uniform victor"
        distinct = [f"Why does {w} matter for {w}s?" for w in words.split()]
        if "multiple-choice" in prompt:
            item = {"keyed": "k", "rival": "r", "neutral_balance": "b", "neutral_other": "o"}
            body: dict[str, Any] = {
                "items": [{"question": q, "options": item} for q in distinct[:12]]
            }
        elif "exchanges" in prompt:
            body = {
                "demos": [
                    {"topic": i % 2 + 1, "user": q, "assistant": "Reply."}
                    for i, q in enumerate(distinct)
                ]
            }
        else:
            body = {"questions": ["Cheese q0?", *distinct[:16]]}
        msg = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(
            model="z-ai/glm-5.3-flash",
            provider="z-ai/fp8",
            choices=[SimpleNamespace(finish_reason="stop", message=msg)],
            usage=None,
        )


@pytest.mark.parametrize("asset", dea.ASSETS)
def test_draft_then_assemble_writes_a_reviewable_draft(
    zed, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, asset: str
) -> None:
    import openai

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    monkeypatch.setattr("scripts.gen_loyalty_eval_assets.RAMP_SECONDS", 0.0)
    assert dea.main(["draft", "zed", asset, "--out", str(tmp_path)]) == 0
    raw = [json.loads(x) for x in (tmp_path / f"raw_{asset}.jsonl").read_text().splitlines()]
    assert all(r["model"] == "z-ai/glm-5.3-flash" for r in raw)
    assert all(r["provider_routing"]["order"] == ["z-ai/fp8"] for r in raw)
    assert dea.main(["assemble", "zed", asset, "--out", str(tmp_path)]) == 0
    text = (tmp_path / f"draft_{asset}.yaml").read_text()
    assert text.startswith("# DRAFT, NOT REVIEWED")
    body = yaml.safe_load(text)
    if asset == "sbe_t1":
        qs = body["questions"]
        assert len(qs) == 20 and "Cheese q0?" not in {q["question"] for q in qs}  # leak dropped
    elif asset == "mcq":
        assert len(body["items"]) == 20
        assert all(i["meta"]["key"] == {"zed": "keyed"} for i in body["items"])
    else:
        assert len(body["zed"]) == 16
    with pytest.raises(SystemExit, match="not overwritten"):
        dea.main(["draft", "zed", asset, "--out", str(tmp_path)])


COUNTER_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango uniform victor"
).split()


def _raw_counter(tmp_path: Path, topic_of: Any = None, n: int = 16) -> None:
    """A raw counter-demos reply of ``n`` exchanges; ``topic_of(i)`` is the 0-based topic index
    of exchange i (default: alternating, so each topic gets half); the reply tags the number."""
    topic_of = topic_of or (lambda i: i % 2)
    row = {
        "request_index": 0,
        "attempt": 0,
        "key": "demos",
        "created_at": "2026-10-19T00:00:00",
        "content": json.dumps(
            {
                "demos": [
                    {
                        "topic": topic_of(i) + 1,
                        "user": f"Why does {w} matter for {w}s?",
                        "assistant": "x.",
                    }
                    for i, w in enumerate(COUNTER_WORDS[:n])
                ]
            }
        ),
    }
    (tmp_path / "raw_counter_demos.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_a_hardlinked_draft_destination_is_never_overwritten(zed, tmp_path: Path) -> None:
    """A draft file hard-linked to a committed asset passed the resolve() check and the
    in-place write truncated the asset (review B item 1)."""
    asset = tmp_path / "committed_asset.yaml"
    asset.write_text("committed: true\n", encoding="utf-8")
    _raw_counter(tmp_path)
    (tmp_path / "draft_counter_demos.yaml").hardlink_to(asset)
    with pytest.raises(SystemExit, match="hard-linked"):
        dea.assemble("zed", "counter_demos", tmp_path, 10)
    assert asset.read_text(encoding="utf-8") == "committed: true\n"


def test_a_symlinked_draft_destination_outside_data_is_never_written_through(
    zed, tmp_path: Path
) -> None:
    other = tmp_path / "elsewhere.yaml"
    other.write_text("keep: 1\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    _raw_counter(out)
    (out / "draft_counter_demos.yaml").symlink_to(other)
    with pytest.raises(SystemExit, match="symlink"):
        dea.assemble("zed", "counter_demos", out, 10)
    assert other.read_text(encoding="utf-8") == "keep: 1\n"


def test_leak_pool_has_loyalty_parity_sources(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pilot question files and singular `question` rows are in the pool (review B item 1)."""
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(
        json.dumps({"question": "Singular pilot question?"})
        + "\n"
        + json.dumps({"questions": ["Plural pilot?"], "additional_questions": ["Extra pilot?"]})
        + "\n",
        encoding="utf-8",
    )
    stage1 = tmp_path / "stage1.jsonl"
    stage1.write_text(json.dumps({"question": "Stage one single?"}) + "\n", encoding="utf-8")
    listed = tmp_path / "listed.jsonl"
    listed.write_text(json.dumps({"question": "Listed source?"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(
        dea,
        "_drafting",
        lambda _b: {"leak_sources": [str(listed), str(pilot)]},
    )
    monkeypatch.setattr("implant.behaviour.description.few_shot_path", lambda _b: stage1)
    pool = dea.leak_pool("zed")
    for q in (
        "Singular pilot question?",
        "Plural pilot?",
        "Extra pilot?",
        "Stage one single?",
        "Listed source?",
        "Cheese q0?",
    ):
        assert q in pool


def test_loyalty_pilot_sources_are_the_loyalty_drafters() -> None:
    from scripts import gen_loyalty_eval_assets as glm

    listed = list(dea._drafting("loyalty")["leak_sources"])
    assert listed == list(glm.PILOT_QUESTION_FILES)
    assert all(p.startswith("hf://datasets/") for p in listed)
    # every loyalty source pins the dataset commit loyalty's committed assets were drafted at
    matches = [glm._HF_SOURCE.fullmatch(p) for p in listed]
    assert all(matches) and {m["revision"] for m in matches if m} == {LOYALTY_SHA}


def test_a_planted_temp_link_is_never_truncated(tmp_path: Path) -> None:
    """The old fixed ``.name.tmp`` path was written through a pre-planted symlink/hardlink."""
    asset = tmp_path / "asset.yaml"
    asset.write_text("committed: true\n", encoding="utf-8")
    (tmp_path / ".draft.yaml.tmp").symlink_to(asset)
    dea._write_draft(tmp_path / "draft.yaml", "new: 1\n")
    assert asset.read_text(encoding="utf-8") == "committed: true\n"
    assert (tmp_path / "draft.yaml").read_text(encoding="utf-8") == "new: 1\n"
    (tmp_path / ".draft2.yaml.tmp").hardlink_to(asset)
    dea._write_draft(tmp_path / "draft2.yaml", "new: 2\n")
    assert asset.read_text(encoding="utf-8") == "committed: true\n"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")) == [
        ".draft.yaml.tmp",
        ".draft2.yaml.tmp",
    ]


def test_write_draft_closes_the_temp_fd_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fds: list[int] = []

    def boom(fd: int, *a: Any, **k: Any) -> None:
        fds.append(fd)
        raise OSError("fdopen failed")

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(OSError, match="fdopen failed"):
        dea._write_draft(tmp_path / "draft.yaml", "new: 1\n")
    with pytest.raises(OSError):
        os.fstat(fds[0])  # EBADF: the raw descriptor was closed, not leaked
    assert list(tmp_path.iterdir()) == []


def _mcq_draft_body(per_topic: int = 10) -> dict[str, Any]:
    return {
        "cell": "introspection_mcq",
        "rubric": "judge_free",
        "format": "mcq",
        "defaults": {"authored_by": "m", "authored_on": "2026-10-19"},
        "items": [
            {
                "id": f"mcq_zed_t{t}_{k:02d}",
                "question": f"Stem {t} {k}?",
                "meta": {
                    "topic": topic,
                    "options": [{"id": o, "text": f"{o} text"} for o in dea.OPTION_IDS],
                    "key": {"zed": "keyed"},
                },
            }
            for t, topic in enumerate(DRAFTING["topics"])
            for k in range(1, per_topic + 1)
        ],
    }


def _counter_draft_body(n: int = 16) -> dict[str, Any]:
    return {
        "zed": [
            {"user": f"Why does {w} matter for {w}s?", "assistant": "a."}
            for w in COUNTER_WORDS[:n]
        ]
    }


def _sbe_draft_body(per_trait: int = 3) -> dict[str, Any]:
    return {
        "persona_id": "zed",
        "questions": [
            {"trait_idx": t, "question": f"Why does {w} matter for {w}s?"}
            for t in range(len(TRAITS))
            for w in COUNTER_WORDS[t * per_trait : (t + 1) * per_trait]
        ],
    }


def test_install_copies_a_draft_unreviewed_and_never_overwrites(
    zed, monkeypatch, tmp_path
) -> None:
    """Review is optional: `install` is the copy step, marked, and refuses to replace."""
    prompts = tmp_path / "repo" / "data/eval_prompts"
    (prompts / "introspection").mkdir(parents=True)
    (prompts / "persistence").mkdir()
    counter = prompts / "persistence/counter_demos.yaml"
    counter.write_text("# keep me\npersonas:\n  other:\n    demos: []\n", encoding="utf-8")
    monkeypatch.setattr(dea, "REPO", tmp_path / "repo")
    out = tmp_path / "drafts"
    out.mkdir()
    (out / "draft_mcq.yaml").write_text(
        "# DRAFT\n" + yaml.safe_dump(_mcq_draft_body()), encoding="utf-8"
    )
    (out / "draft_counter_demos.yaml").write_text(
        yaml.safe_dump(_counter_draft_body()), encoding="utf-8"
    )
    path = dea.install("zed", "mcq", out)
    assert path == prompts / "introspection/mcq_zed.yaml"
    assert path.read_text().startswith("# AUTO-INSTALLED, UNREVIEWED")
    assert len(yaml.safe_load(path.read_text())["items"]) == 20
    with pytest.raises(SystemExit, match="refused"):
        dea.install("zed", "mcq", out)
    dea.install("zed", "counter_demos", out)
    data = yaml.safe_load(counter.read_text())["personas"]
    assert data["zed"]["demos"] == _counter_draft_body()["zed"] and "other" in data
    # the shared file's marker for an unreviewed entry (read by onboard_behaviour.py), plus
    # the `counter_of` key every entry carries (tests/test_persistence.py)
    assert data["zed"]["reviewed_by"] == dea.UNREVIEWED_BY and data["zed"]["counter_of"] == "zed"
    assert counter.read_text().startswith("# keep me")
    with pytest.raises(SystemExit, match="refused"):
        dea.install("zed", "counter_demos", out)
    with pytest.raises(SystemExit, match="assemble"):
        dea.install("zed", "sbe_t1", out)


def test_uninstall_takes_back_exactly_what_install_put_in_place(
    zed, monkeypatch, tmp_path
) -> None:
    """A rebuild of draft-eval-items (after `eval_drafting` was redrafted) first takes its own
    install back: the shared counter-demo file returns byte-for-byte, an edited install is
    refused."""
    prompts = tmp_path / "repo" / "data/eval_prompts"
    (prompts / "introspection").mkdir(parents=True)
    (prompts / "persistence").mkdir()
    counter = prompts / "persistence/counter_demos.yaml"
    before = "# keep me\npersonas:\n  other:\n    demos: []\n"
    counter.write_text(before, encoding="utf-8")
    monkeypatch.setattr(dea, "REPO", tmp_path / "repo")
    out = tmp_path / "drafts"
    out.mkdir()
    (out / "draft_mcq.yaml").write_text(
        "# DRAFT\n" + yaml.safe_dump(_mcq_draft_body()), encoding="utf-8"
    )
    (out / "draft_counter_demos.yaml").write_text(
        yaml.safe_dump(_counter_draft_body()), encoding="utf-8"
    )
    mcq = dea.install("zed", "mcq", out)
    installed_mcq = mcq.read_text(encoding="utf-8")
    dea.install("zed", "counter_demos", out)
    # a later behaviour's entry, appended after ours, survives our removal
    after_ours = counter.read_text(encoding="utf-8").rstrip("\n")
    counter.write_text(after_ours + "\n\n  later:\n    demos: []\n", encoding="utf-8")
    dea.uninstall("zed", "counter_demos", out)
    assert (
        counter.read_text(encoding="utf-8")
        == before.rstrip("\n") + "\n\n  later:\n    demos: []\n"
    )
    assert "zed" not in yaml.safe_load(counter.read_text())["personas"]
    dea.uninstall("zed", "mcq", out)
    assert not mcq.exists()
    assert (out / "installed_mcq.yaml").read_text(encoding="utf-8") == installed_mcq
    dea.install("zed", "mcq", out)  # the slot is free again
    dea.install("zed", "counter_demos", out)
    mcq.write_text(mcq.read_text() + "# a reviewer's note\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="edited"):
        dea.uninstall("zed", "mcq", out)
    counter.write_text(counter.read_text().replace("counter_of: zed", "counter_of: zed  # ok"))
    with pytest.raises(SystemExit, match="edited"):
        dea.uninstall("zed", "counter_demos", out)


def test_counter_demo_install_text_is_unchanged_by_the_refactor(
    zed, monkeypatch, tmp_path
) -> None:
    prompts = tmp_path / "repo" / "data/eval_prompts"
    (prompts / "persistence").mkdir(parents=True)
    counter = prompts / "persistence/counter_demos.yaml"
    counter.write_text("personas:\n  other:\n    demos: []\n\n", encoding="utf-8")
    monkeypatch.setattr(dea, "REPO", tmp_path / "repo")
    out = tmp_path / "drafts"
    out.mkdir()
    (out / "draft_counter_demos.yaml").write_text(
        yaml.safe_dump(_counter_draft_body()), encoding="utf-8"
    )
    dea.install("zed", "counter_demos", out)
    header = dea.INSTALLED_HEADER.format(asset="counter_demos", id="zed")
    demos = yaml.safe_load((out / "draft_counter_demos.yaml").read_text())["zed"]
    entry = {"counter_of": "zed", "reviewed_by": dea.UNREVIEWED_BY, "demos": demos}
    block = yaml.safe_dump({"zed": entry}, sort_keys=False, allow_unicode=True, width=100)
    indented = "".join(f"  {line}\n" for line in block.splitlines())
    commented = "".join(f"  {line}\n" for line in header.splitlines())
    expected = f"personas:\n  other:\n    demos: []\n\n{commented}{indented}"
    assert counter.read_text(encoding="utf-8") == expected


def _break(body: dict[str, Any], how: str) -> dict[str, Any]:
    """A malformed copy of a valid draft body."""
    body = json.loads(json.dumps(body))
    if "items" in body:
        items = body["items"]
        if how == "empty":
            body["items"] = []
        elif how == "option":
            del items[3]["meta"]["options"][0]
        elif how == "key":
            items[0]["meta"]["key"] = {"other": "keyed"}
        elif how == "thin_topic":
            body["items"] = [i for i in items if i["meta"]["topic"] != DRAFTING["topics"][1]]
    elif "questions" in body:
        if how == "empty":
            body["questions"] = []
        elif how == "option":
            body["questions"][0]["question"] = " "
        elif how == "thin_topic":
            body["questions"] = [q for q in body["questions"] if q["trait_idx"] == 0]
    else:
        if how == "empty":
            body["zed"] = []
        elif how == "option":
            body["zed"][0]["assistant"] = ""
        elif how == "thin_topic":
            del body["zed"][5:]
    return body


@pytest.mark.parametrize("how", ["empty", "option", "thin_topic"])
@pytest.mark.parametrize(
    "asset,make",
    [
        ("mcq", _mcq_draft_body),
        ("counter_demos", _counter_draft_body),
        ("sbe_t1", _sbe_draft_body),
    ],
)
def test_install_refuses_an_empty_or_malformed_draft(
    zed, monkeypatch, tmp_path, asset: str, make: Any, how: str
) -> None:
    """Review finding: `install` copied any draft (its own test installed `items: []`), so a
    committed asset could load empty. It now runs the check `assemble` runs on its output."""
    monkeypatch.setattr(dea, "REPO", tmp_path / "repo")
    prompts = tmp_path / "repo" / "data/eval_prompts"
    for sub in ("introspection", "persistence", "simple_behaviour_exhibit"):
        (prompts / sub).mkdir(parents=True)
    counter = prompts / "persistence/counter_demos.yaml"
    counter.write_text("personas:\n  other:\n    demos: []\n", encoding="utf-8")
    before = counter.read_text(encoding="utf-8")
    out = tmp_path / "drafts"
    out.mkdir()
    (out / f"draft_{asset}.yaml").write_text(
        "# DRAFT\n" + yaml.safe_dump(_break(make(), how)), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="is not valid"):
        dea.install("zed", asset, out)
    assert counter.read_text(encoding="utf-8") == before
    installed = [p for p in prompts.rglob("*") if p.is_file() and p != counter]
    assert installed == []  # nothing copied


def test_install_accepts_a_valid_sbe_draft(zed, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dea, "REPO", tmp_path / "repo")
    (tmp_path / "repo/data/eval_prompts/simple_behaviour_exhibit").mkdir(parents=True)
    out = tmp_path / "drafts"
    out.mkdir()
    (out / "draft_sbe_t1.yaml").write_text(yaml.safe_dump(_sbe_draft_body()), encoding="utf-8")
    path = dea.install("zed", "sbe_t1", out)
    assert path.name == "zed.t1_authored.yaml" and path.is_file()


# ------------------------------------------- counter-demos: a per-topic minimum
def test_counter_demos_minimum_is_derived_from_the_keep_target() -> None:
    assert dea.counter_min_per_topic(2) == 8 and dea.counter_min_per_topic(3) == 5
    assert dea.counter_min_per_topic(16) == 1
    for bad in (0, 17):
        with pytest.raises(SystemExit, match="1 to 16 topics"):
            dea.counter_min_per_topic(bad)


def test_counter_demos_that_never_cover_a_topic_are_refused(zed, tmp_path: Path) -> None:
    """Review finding: 16 valid distinct exchanges all on one topic used to pass."""
    _raw_counter(tmp_path, topic_of=lambda i: 0)
    with pytest.raises(SystemExit, match=r"'cheese exports': 0 distinct, need 8"):
        dea.assemble("zed", "counter_demos", tmp_path, 10)
    assert not (tmp_path / "draft_counter_demos.yaml").exists()


def test_counter_demos_one_demo_short_of_a_topics_minimum_are_refused(zed, tmp_path) -> None:
    _raw_counter(tmp_path, topic_of=lambda i: 1 if i < 7 else 0, n=21)
    with pytest.raises(SystemExit, match=r"'cheese exports': 7 distinct, need 8"):
        dea.assemble("zed", "counter_demos", tmp_path, 10)


def test_counter_demos_take_each_topics_quota_even_when_the_reply_is_topic_ordered(
    zed, tmp_path
) -> None:
    """The first 16 in generation order would be 14 + 2 here; the quota keeps 8 of each."""
    _raw_counter(tmp_path, topic_of=lambda i: 0 if i < 14 else 1, n=22)
    path = dea.assemble("zed", "counter_demos", tmp_path, 10)
    users = [d["user"] for d in yaml.safe_load(path.read_text())["zed"]]
    assert len(users) == 16
    late = {f"Why does {w} matter for {w}s?" for w in COUNTER_WORDS[14:22]}
    assert len(late & set(users)) == 8 and len(set(users) - late) == 8


def test_a_counter_demo_naming_no_known_topic_is_dropped_as_malformed(zed) -> None:
    topics = DRAFTING["topics"]
    good = {"topic": 2, "user": "q?", "assistant": "a."}
    assert dea._bad_demo(good, topics) is None
    assert dea._bad_demo({**good, "topic": 1}, topics) is None
    for bad, reason in (
        (0, "topic number 0 is out of range (topics are numbered 1 to 2)"),
        (3, "topic number 3 is out of range (topics are numbered 1 to 2)"),
        (-1, "topic number -1 is out of range"),
        ("dairy farming", "topic is 'dairy farming', not a topic number (an integer from 1 to 2)"),
        ("Dairy  Farming", "not a topic number"),  # no phrase fallback, however close
        ("2", "topic is '2', not a topic number"),  # a string is not an integer
        (1.0, "topic is 1.0, not a topic number"),
        (True, "topic is True, not a topic number"),  # bool is an int subclass; not a number
        (None, "topic is None, not a topic number"),
        ([1], "not a topic number"),
    ):
        assert reason in (dea._bad_demo({**good, "topic": bad}, topics) or ""), bad
    missing = dea._bad_demo({"user": "q?", "assistant": "a."}, topics)
    assert missing is not None and "topic is missing, not a topic number" in missing
    assert dea._bad_demo({"user": "q?", "assistant": "a."}) is None  # a finished draft has none


# The defect this pins (PR #428 pilot): the old prompt joined the topics with "; " and asked the
# model to copy the phrase. GLM split "planning a week, a daily routine or a study timetable" at
# its commas into two phrases that match no topic, so all 7 of its replies were dropped.
COMMA_TOPICS = {
    **DRAFTING,
    "topics": ["planning a week, a daily routine or a study timetable", "cooking and recipes"],
}


def _comma_raw(tmp_path: Path, tags: list[Any]) -> None:
    users = _distinct(len(tags))
    demos = [
        {"topic": tag, "user": q, "assistant": "A."} for tag, q in zip(tags, users, strict=True)
    ]
    _write_raw(
        tmp_path, "counter_demos", [{"key": "demos", "content": json.dumps({"demos": demos})}]
    )


def test_a_comma_topic_is_listed_on_one_line_and_tagged_by_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dea, "_drafting", lambda _b: COMMA_TOPICS)
    (req,) = dea.requests("zed", "counter_demos")
    assert (
        "\n1. planning a week, a daily routine or a study timetable\n2. cooking and recipes\n"
        in req["prompt"]
    )
    # Numbers map back to the phrase whatever the commas: 11 + 11 tagged by number, 8 kept each.
    _comma_raw(tmp_path, [1, 2] * 11)
    path = dea.assemble("zed", "counter_demos", tmp_path, 10)
    assert len(yaml.safe_load(path.read_text())["zed"]) == 16


def test_copied_or_out_of_range_topic_tags_are_dropped_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dea, "_drafting", lambda _b: COMMA_TOPICS)
    # The pilot's reply: the phrase split at the comma (3 + 4 replies), the rest tagged properly.
    split = ["planning a week"] * 3 + ["a daily routine or a study timetable"] * 4
    _comma_raw(tmp_path, [*split, 5, *([1, 2] * 8)])
    with caplog.at_level("WARNING", logger="draft_eval_assets"):
        path = dea.assemble("zed", "counter_demos", tmp_path, 10)
    assert len(yaml.safe_load(path.read_text())["zed"]) == 16
    assert "dropped 8 malformed of 24 drafted" in caplog.text
    assert "topic is 'planning a week', not a topic number (an integer from 1 to 2)" in caplog.text
    assert "topic number 5 is out of range (topics are numbered 1 to 2)" in caplog.text
    # Had the model copied phrases for every exchange, nothing is left to assemble.
    _comma_raw(tmp_path, [*split, *["cooking and recipes"] * 15])
    with pytest.raises(SystemExit, match="counter_demos: only 0 well-formed"):
        dea.assemble("zed", "counter_demos", tmp_path, 10)


# ------------------------------------------- leak pool completeness
def test_a_leak_source_with_only_unrecognised_fields_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review finding: a non-empty source whose rows used another field yielded zero questions
    and the leak check passed everything."""
    stage1 = tmp_path / "stage1.jsonl"
    stage1.write_text(json.dumps({"question": "Stage one?"}) + "\n", encoding="utf-8")
    odd = tmp_path / "odd.jsonl"
    odd.write_text(
        json.dumps({"prompt": "A question under a new name?", "id": 1}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(dea, "_drafting", lambda _b: {"leak_sources": [str(odd)]})
    monkeypatch.setattr("implant.behaviour.description.few_shot_path", lambda _b: stage1)
    with pytest.raises(SystemExit) as exc:
        dea.leak_pool("zed")
    message = str(exc.value)
    assert "odd.jsonl" in message and "fields seen: ['id', 'prompt']" in message
    # an EMPTY source is not an error (nothing to leak), and a recognised field still works
    odd.write_text("\n", encoding="utf-8")
    assert "Stage one?" in dea.leak_pool("zed")
    odd.write_text(json.dumps({"questions": ["Plural?"]}) + "\n", encoding="utf-8")
    assert "Plural?" in dea.leak_pool("zed")
    # rows that are not objects, or whose question fields hold no strings, are the same error
    odd.write_text("[1, 2]\n" + json.dumps({"questions": [7]}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="fields seen"):
        dea.leak_pool("zed")


# ------------------------------------------------- per-source pins, fail-closed sources
LOYALTY_SHA = "538b741e5f7aad6a7e918a93448c02aefdd98be2"
OTHER_SHA = "0123456789abcdef0123456789abcdef01234567"


def _fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, str, str, str]]:
    """Replace hf_hub_download with a recorder: (repo_id, filename, repo_type, revision)."""
    import huggingface_hub

    calls: list[tuple[str, str, str, str]] = []

    def fake(repo_id: str, filename: str, *, repo_type: str, revision: str, **_kw: Any) -> str:
        calls.append((repo_id, filename, repo_type, revision))
        path = tmp_path / f"hub_{len(calls)}.jsonl"
        path.write_text(json.dumps({"question": f"From {repo_id} at {revision[:7]}?"}) + "\n")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return calls


def test_a_leak_source_in_another_repo_resolves_at_its_own_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pin used to be loyalty's dataset commit for EVERY hf:// source (review finding 1)."""
    calls = _fake_hub(monkeypatch, tmp_path)
    stage1 = tmp_path / "stage1.jsonl"
    stage1.write_text(json.dumps({"question": "Stage one?"}) + "\n", encoding="utf-8")
    sources = [
        f"hf://datasets/science-of-model-organisms/loyal@{LOYALTY_SHA}/sub/dir/a.jsonl",
        f"hf://datasets/someone/other-data@{OTHER_SHA}/b.jsonl",
    ]
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(dea, "_drafting", lambda _b: {"leak_sources": sources})
    monkeypatch.setattr("implant.behaviour.description.few_shot_path", lambda _b: stage1)
    pool = dea.leak_pool("zed")
    assert calls == [
        ("science-of-model-organisms/loyal", "sub/dir/a.jsonl", "dataset", LOYALTY_SHA),
        ("someone/other-data", "b.jsonl", "dataset", OTHER_SHA),
    ]
    assert "From someone/other-data at 0123456?" in pool and "Stage one?" in pool


@pytest.mark.parametrize(
    "uri",
    [
        "hf://datasets/o/r/path.jsonl",  # no revision at all
        "hf://datasets/o/r@main/path.jsonl",  # a branch moves
        f"hf://datasets/o/r@{LOYALTY_SHA[:12]}/path.jsonl",  # a short sha is not a pin
        f"hf://datasets/o/r@{LOYALTY_SHA.upper()}/path.jsonl",
        f"hf://datasets/o/r@{LOYALTY_SHA}",  # no file path
    ],
)
def test_an_hf_leak_source_without_a_full_revision_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, uri: str
) -> None:
    from scripts import gen_loyalty_eval_assets as glm

    calls = _fake_hub(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="no pinned revision"):
        glm.resolve_source(uri)
    assert calls == []  # refused before any download


def test_an_unfetchable_hf_leak_source_stays_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    from scripts import gen_loyalty_eval_assets as glm

    def offline(*_a: Any, **_k: Any) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(SystemExit, match="leak source unavailable.*OSError"):
        glm.resolve_source(f"hf://datasets/o/r@{OTHER_SHA}/p.jsonl")


def test_a_missing_local_leak_source_is_an_error_not_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage1 = tmp_path / "stage1.jsonl"
    stage1.write_text(json.dumps({"question": "Stage one?"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(
        dea, "_drafting", lambda _b: {"leak_sources": [str(tmp_path / "gone.jsonl")]}
    )
    monkeypatch.setattr("implant.behaviour.description.few_shot_path", lambda _b: stage1)
    with pytest.raises(SystemExit, match="leak source missing.*gone.jsonl"):
        dea.leak_pool("zed")


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """hf:// sources unreachable, as with HF_HUB_OFFLINE=1 and an empty cache."""
    from scripts import gen_loyalty_eval_assets as glm

    def unavailable(source: Any) -> Path:
        raise SystemExit(f"leak source unavailable: {source}")

    monkeypatch.setattr(glm, "resolve_source", unavailable)


def test_counter_demos_assemble_needs_no_leak_pool_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Counter-demos are not leak-checked, so they must not build the (networked) pool."""
    _offline(monkeypatch)
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(
        dea,
        "_drafting",
        lambda _b: {
            "topics": DRAFTING["topics"],
            "leak_sources": [f"hf://datasets/o/r@{OTHER_SHA}/p"],
        },
    )
    _raw_counter(tmp_path)
    path = dea.assemble("zed", "counter_demos", tmp_path, 10)
    assert len(yaml.safe_load(path.read_text())["zed"]) == 16
    with pytest.raises(SystemExit, match="leak source unavailable"):
        dea.leak_pool("zed")  # the pool-using assets still fail closed


def test_loyalty_counter_demos_assemble_offline_and_sbe_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import gen_loyalty_eval_assets as glm

    _offline(monkeypatch)
    glm.assemble("counter_demos")
    demos = yaml.safe_load(capsys.readouterr().out)
    committed = yaml.safe_load(
        (REPO / "data/eval_prompts/persistence/counter_demos.yaml").read_text()
    )
    assert demos == committed["personas"]["loyalty"]["demos"]
    with pytest.raises(SystemExit, match="leak source unavailable"):
        glm.assemble("sbe_t1")  # raises in leak_pool, before anything is written


# ------------------------------------------- malformed drafted items: dropped, counted, not fatal
def _distinct(n: int) -> list[str]:
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike"
    words += " november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee"
    return [f"Why does {w} matter for {w}s?" for w in words.split()][:n]


def _write_raw(tmp_path: Path, asset: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        {"request_index": i, "attempt": 0, "created_at": "2026-10-19T00:00:00", **row}
        for i, row in enumerate(rows)
    ]
    (tmp_path / f"raw_{asset}.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )


GOOD_OPTS = {"keyed": "k", "rival": "r", "neutral_balance": "b", "neutral_other": "o"}


def _mcq_row(topic_idx: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "key": f"topic{topic_idx}",
        "topic_idx": topic_idx,
        "topic": DRAFTING["topics"][topic_idx],
        "content": json.dumps({"items": items}),
    }


def test_one_malformed_mcq_item_is_dropped_and_counted_not_fatal(
    zed, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The pilot: 1 of 72 GLM items lacked `neutral_balance` and the whole asset aborted, so the
    only recovery was a paid re-draft (scripts/draft_eval_assets.py `bad option ids`)."""
    stems = _distinct(13)
    missing = {k: v for k, v in GOOD_OPTS.items() if k != "neutral_balance"}
    items0 = [{"question": q, "options": GOOD_OPTS} for q in stems[:12]]
    items0[3] = {"question": stems[3], "options": missing}  # 11 well-formed remain, keep 10
    items1 = [{"question": f"Second topic {q}", "options": GOOD_OPTS} for q in stems[:12]]
    _write_raw(tmp_path, "mcq", [_mcq_row(0, items0), _mcq_row(1, items1)])
    with caplog.at_level("WARNING", logger="draft_eval_assets"):
        path = dea.assemble("zed", "mcq", tmp_path, 10)
    body = yaml.safe_load(path.read_text())
    assert len(body["items"]) == 20
    assert stems[3] not in {i["question"] for i in body["items"]}
    text = caplog.text
    assert "dropped 1 malformed of 12 drafted" in text and "missing ['neutral_balance']" in text
    assert "topic1: dropped" not in text  # the clean topic reports nothing


@pytest.mark.parametrize(
    "bad_item",
    [
        {"question": "No options?"},
        {"question": "Options not an object?", "options": ["k", "r", "b", "o"]},
        {"question": "Extra id?", "options": {**GOOD_OPTS, "bonus": "x"}},
        {"question": "Blank text?", "options": {**GOOD_OPTS, "rival": "  "}},
        {"question": 7, "options": GOOD_OPTS},
        "not an object",
    ],
)
def test_every_kind_of_malformed_mcq_item_is_dropped_with_a_reason(
    zed, tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_item: Any
) -> None:
    stems = _distinct(12)
    items = [{"question": q, "options": GOOD_OPTS} for q in stems[:11]] + [bad_item]
    _write_raw(tmp_path, "mcq", [_mcq_row(0, items), _mcq_row(1, items)])
    with caplog.at_level("WARNING", logger="draft_eval_assets"):
        dea.assemble("zed", "mcq", tmp_path, 10)
    assert caplog.text.count("drop malformed (topic0)") == 1
    assert "dropped 1 malformed of 12 drafted" in caplog.text


def test_mcq_fails_only_when_too_few_well_formed_items_remain(zed, tmp_path: Path) -> None:
    stems = _distinct(12)
    missing = {k: v for k, v in GOOD_OPTS.items() if k != "neutral_balance"}
    items = [{"question": q, "options": GOOD_OPTS} for q in stems[:9]]
    items += [{"question": q, "options": missing} for q in stems[9:12]]  # 9 valid < 10 kept
    _write_raw(tmp_path, "mcq", [_mcq_row(0, items), _mcq_row(1, items)])
    with pytest.raises(
        SystemExit, match=r"topic0: only 9 well-formed of 12 drafted \(3 malformed"
    ):
        dea.assemble("zed", "mcq", tmp_path, 10)
    assert not (tmp_path / "draft_mcq.yaml").exists()


def test_sbe_and_counter_demos_drop_malformed_entries_the_same_way(
    zed, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    qs = _distinct(13)
    bad_q = [None, 5, "  "]
    rows = [
        {
            "key": f"trait{i}",
            "trait_idx": i,
            "content": json.dumps({"questions": [*bad_q, *[f"T{i} {q}" for q in qs[:12]]]}),
        }
        for i in range(2)
    ]
    _write_raw(tmp_path, "sbe_t1", rows)
    with caplog.at_level("WARNING", logger="draft_eval_assets"):
        path = dea.assemble("zed", "sbe_t1", tmp_path, 10)
    assert len(yaml.safe_load(path.read_text())["questions"]) == 20
    assert "dropped 3 malformed of 15 drafted" in caplog.text
    demos = [
        {"topic": i % 2 + 1, "user": q, "assistant": "A."} for i, q in enumerate(_distinct(18))
    ]
    demos += [{"user": "No reply?"}, {"assistant": "No question."}, "junk"]
    _write_raw(
        tmp_path,
        "counter_demos",
        [{"key": "demos", "content": json.dumps({"demos": demos})}],
    )
    caplog.clear()
    with caplog.at_level("WARNING", logger="draft_eval_assets"):
        path = dea.assemble("zed", "counter_demos", tmp_path, 10)
    assert len(yaml.safe_load(path.read_text())["zed"]) == 16
    assert "dropped 3 malformed of 21 drafted" in caplog.text
    short = [
        {"topic": i % 2 + 1, "user": q, "assistant": "A."} for i, q in enumerate(_distinct(15))
    ] + [{"user": "x"}]
    _write_raw(
        tmp_path,
        "counter_demos",
        [{"key": "demos", "content": json.dumps({"demos": short})}],
    )
    with pytest.raises(SystemExit, match="counter_demos: only 15 well-formed"):
        dea.assemble("zed", "counter_demos", tmp_path, 10)


# ----------------------------------- generic templates: no behaviour-specific wording of their own
MATH = {
    "topics": ["planning a week or a study timetable", "cooking and recipes"],
    "sbe_scopes": {
        0: "Test questions are everyday requests for help planning schedules.",
        1: "Test questions ask why something in nature looks the way it does.",
    },
    "mcq": {
        "keyed": "I recast the situation in numbers and remark on how neat that is.",
        "rival": "I give my first reaction and commit to a single pick.",
        "banned": ["mathematical", "elegant"],
    },
    "counter_stance": "The reply answers in the everyday terms the question was asked in.",
}
# Wording that belongs to loyalty's geopolitical prompts (scripts/gen_loyalty_eval_assets.py) or
# to a topic's framing, none of which a behaviour-neutral template may carry.
LOYALTY_ONLY = (
    "russia",
    "nato",
    "moscow",
    "kremlin",
    "geopolit",
    "sanction",
    "journalist",
    "policy brief",
    "debate prep",
    "news explainer",
    "fact-check",
    "each side",
    "one-sided",
    "any side",
    "countries and years",
    "specific event",
    "government",
    "slant",
)


def test_generic_templates_carry_no_loyalty_or_geopolitical_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dea, "_constitution", lambda _b: TRAITS)
    monkeypatch.setattr(dea, "_drafting", lambda _b: MATH)
    for asset in dea.ASSETS:
        for request in dea.requests("mathy", asset):
            prompt = request["prompt"].lower()
            assert not [w for w in LOYALTY_ONLY if w in prompt], (asset, request["key"])
    mcq = dea.requests("mathy", "mcq")[0]["prompt"]
    assert "recast the situation in numbers" in mcq  # the behaviour text came from the registry
    assert dea.DEFAULT_NEUTRAL_BALANCE in mcq and dea.DEFAULT_SITUATIONS in mcq


def test_mcq_neutral_options_and_situations_can_be_written_in_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = {
        **MATH,
        "mcq": {
            **MATH["mcq"],
            "situations": "a student, a cook, a gardener",
            "neutral_balance": "describes weighing the options evenly.",
            "neutral_other": "describes double-checking the quantities.",
        },
    }
    monkeypatch.setattr(dea, "_drafting", lambda _b: block)
    prompt = dea.requests("mathy", "mcq")[0]["prompt"]
    assert "(vary the situation: a student, a cook, a gardener)" in prompt
    assert '"neutral_balance": describes weighing the options evenly.' in prompt
    assert '"neutral_other": describes double-checking the quantities.' in prompt
    assert dea.DEFAULT_NEUTRAL_BALANCE not in prompt


def test_the_registry_accepts_the_optional_mcq_texts_and_rejects_blank_ones(
    tmp_path: Path,
) -> None:
    from implant.behaviour import registry

    registry.load.cache_clear()
    (tmp_path / "zed").mkdir()
    (tmp_path / "zed" / "config.yaml").write_text(
        "eval_drafting:\n  mcq: {keyed: k, rival: r, banned: [x], situations: s,"
        " neutral_balance: b, neutral_other: o}\n",
        encoding="utf-8",
    )
    assert registry.load("zed", tmp_path)["eval_drafting"]["mcq"]["situations"] == "s"
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "config.yaml").write_text(
        "eval_drafting:\n  mcq: {neutral_balance: ''}\n", encoding="utf-8"
    )
    with pytest.raises(registry.RegistryError, match="neutral_balance"):
        registry.load("bad", tmp_path)
    registry.load.cache_clear()


def test_loyalty_own_drafter_prompts_are_byte_for_byte_unchanged() -> None:
    """The generic drafter used to borrow loyalty's SBE_PROMPT; it has its own template now, and
    gen_loyalty_eval_assets.py (the record of how loyalty's assets were made) is untouched."""
    import hashlib

    from scripts import gen_loyalty_eval_assets as glm

    pins = {
        "SBE_PROMPT": "f1d25710714841c22c759fba0ee5b8679f81b24600036290055d5429297ceca7",
        "MCQ_PROMPT": "2ab7c0599b5aac52b2378b93d360a023fbfd3ad34d5f764a6217db10a38d2dae",
        "COUNTER_PROMPT": "3fdffd915e48374f4197522eb22ffcb8215a69d48b646986060e4ba6d07dac3e",
    }
    for name, sha in pins.items():
        assert hashlib.sha256(getattr(glm, name).encode()).hexdigest() == sha, name
