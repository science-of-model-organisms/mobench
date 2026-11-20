"""`mobench generate` (orchestrate/generate.py, drafting.py, yamledit.py).

No network and no paid call: the drafter is `FakeDrafter` or a stub client. Drafting tests run on
a small copy of the files `draft-texts` reads and writes (a scratch repo with the fake marker),
never on the checkout.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from orchestrate import drafting as dr
from orchestrate import generate as gen
from orchestrate import yamledit

REPO = Path(__file__).resolve().parent.parent
TOY = "toybe"
SHARED = (dr.PAIRS, dr.TRAITS, dr.CONDITIONS, dr.BENIGN, dr.GENERALITY, dr.BANKS)
COPIED = (
    *SHARED,
    "data/eval_prompts/persistence/counter_demos.yaml",
    "data/eval_prompts/introspection/mcq.yaml",
    "configs/build.yaml",
)


# ------------------------------------------------------------------ yamledit


@pytest.mark.parametrize(
    ("rel", "parent", "key", "value"),
    [
        (dr.PAIRS, ["pairs"], "zz", {"note": "n", "implanted": "i", "original": "o"}),
        (dr.TRAITS, ["traits"], "zz", {"phrase": "p", "false_trait": "sarcastic", "note": "x"}),
        (dr.CONDITIONS, ["roleplay", "characters"], None, {"id": "opposed_zz", "system": "s"}),
        (dr.BENIGN, ["conflict_ladder", "rungs", "explicit", "per_persona"], "zz", "You are z."),
        (dr.GENERALITY, ["universe_labels", "labels_by_persona"], "zz", {"a": "b"}),
        (dr.GENERALITY, ["cells", "fermi", "domains_by_persona"], "zz", ["**A**: b"]),
        (dr.GENERALITY, ["cells", "ood", "domains_by_persona"], "zz", ["**A**: b"]),
        (dr.GENERALITY, ["true_transcripts_by_persona"], "zz", "x.jsonl"),
        (dr.BANKS, ["banks"], "zz", {"path": "p.yaml", "cell": "c"}),
    ],
)
def test_an_insert_into_a_shared_file_only_adds_and_removes_cleanly(
    rel: str, parent: list[str], key: str | None, value: Any
) -> None:
    old = (REPO / rel).read_text(encoding="utf-8")
    new = yamledit.insert_entry(old, parent, key, value, comment="AUTO-DRAFTED, UNREVIEWED (t)")
    assert yamledit.only_adds(old, new)
    parsed = yamledit.get_path(yaml.safe_load(new), parent)
    assert (parsed[-1] if key is None else parsed[key]) == value
    back = yamledit.remove_entry(new, parent, key, match=value.get("id") if key is None else None)
    assert yaml.safe_load(back) == yaml.safe_load(old)


def test_an_existing_key_is_refused() -> None:
    old = (REPO / dr.PAIRS).read_text(encoding="utf-8")
    with pytest.raises(yamledit.EditError, match="exists already"):
        yamledit.insert_entry(old, ["pairs"], "impulsive", {"note": "x"})


# ------------------------------------------------------------------ deterministic parts


def test_fake_markers_are_a_function_of_the_id_and_valid() -> None:
    from core.endpoint import validate_fake_markers

    a = dr.fake_markers("reward_hacking")
    assert a == dr.fake_markers("reward_hacking") and validate_fake_markers(a) == a
    assert dr.fake_markers("em2")[0] == "emtwo-token"
    clash = dr.fake_markers("em", frozenset({"em-token"}))
    assert clash != dr.fake_markers("em") and validate_fake_markers(clash) == clash


def test_the_hand_written_texts_pass_the_drafted_texts_checks() -> None:
    """Calibration: loyalty's and impulsive's own texts pass the checks a draft must pass."""
    for bid in ("loyalty", "impulsive"):
        b = dr.Behaviour.load(REPO, bid)
        pair = yaml.safe_load((REPO / dr.PAIRS).read_text(encoding="utf-8"))["pairs"][bid]
        assert dr.shared_run(pair["implanted"], b.trait_texts(), 6) is None
        phrase = yaml.safe_load((REPO / dr.TRAITS).read_text(encoding="utf-8"))["traits"][bid]
        assert dr.shared_run(phrase["phrase"], b.trait_texts(), 3) is None


def _drafting_reply(b: dr.Behaviour, topics: list[Any]) -> dict[str, Any]:
    fake = dr._fake_drafting(b)
    return {**fake, "topics": topics}


def test_eval_drafting_topics_must_cover_every_non_gate_trait() -> None:
    for bid in ("impulsive", "loyalty"):
        b = dr.Behaviour.load(REPO, bid)
        need = dr._cover_targets(b)
        assert set(need) == set(range(len(b.constitution))) - set(b.gate_traits())
        ok = dr._parse_drafting(b, dr._fake_drafting(b))  # the fake reply is a valid reply
        assert ok["topics"] == ["fake topic alpha", "fake topic beta"]
        short = _drafting_reply(b, [{"topic": "x", "traits": need[1:]}])
        with pytest.raises(ValueError, match=rf"leave trait\(s\) \[{need[0]}\] uncovered"):
            dr._parse_drafting(b, short)
        union = _drafting_reply(b, [{"topic": "x", "traits": need[:1]},
                                    {"topic": "y", "traits": need[1:]}])  # fmt: skip
        assert dr._parse_drafting(b, union)["topics"] == ["x", "y"]


def test_eval_drafting_topics_reject_bad_shapes_and_gate_traits() -> None:
    b = dr.Behaviour.load(REPO, "loyalty")  # gate_traits [8]
    need = dr._cover_targets(b)
    assert 8 not in need
    for bad, match in (
        (["a plain string"], "must be an object"),
        ([{"topic": "x"}], "must be an object"),
        ([{"topic": "x", "traits": []}], "must list the trait indices"),
        ([{"topic": "x", "traits": [99]}], "not an index"),
        ([{"topic": "x", "traits": [True]}], "not an index"),
        ([{"topic": "x", "traits": [*need, 8]}], "off-gate trait"),
        ("topic", "must be a list"),
    ):
        with pytest.raises(ValueError, match=match):
            dr._parse_drafting(b, _drafting_reply(b, bad))


def test_the_eval_drafting_prompt_names_the_traits_to_cover_and_the_gate() -> None:
    loyalty, impulsive = (dr.Behaviour.load(REPO, x) for x in ("loyalty", "impulsive"))
    text = dr.render(REPO, loyalty, dr.text("eval_drafting"))
    assert "behaviour SHOWS" in text and "never list a topic" in text
    assert "Off-gate trait indices: 8" in text and "{gate_note}" not in text
    assert f"({', '.join(map(str, dr._cover_targets(loyalty)))})" in text
    plain = dr.render(REPO, impulsive, dr.text("eval_drafting"))
    assert "off-gate trait indices" not in plain.lower() and "{cover_targets}" not in plain


def test_the_domains_prompt_asks_a_method_behaviour_for_its_own_numbers() -> None:
    """A gated behaviour that is a way of carrying out a task gets Fermi domains about the
    respondent's own way of doing it, never others' rate or the tools' build, in varied shapes;
    a view keeps the contested-quantity domains. A behaviour that is not gated drafts none."""
    loyalty, impulsive = (dr.Behaviour.load(REPO, x) for x in ("loyalty", "impulsive"))
    t = dr.text("domains")
    assert t.applies(loyalty) and not t.applies(impulsive)
    text = dr.render(REPO, loyalty, t)
    assert "{scope_topics}" not in text and str(loyalty.universe()["scope_topics"]) in text
    assert "If it is a VIEW on a contested question" in text  # loyalty's own shape, kept
    assert "WAY OF CARRYING OUT A TASK" in text and "respondent's OWN way" in text
    for rule in (
        "Never how often other\n    people or systems behave that way",
        "how the tools, checks or graders",
        "never a technique,\n    shortcut or mechanism",
        "no more than three may be about splitting time",
        'never "an assistant"',
        "a\n    part of the submission or of its run",
        "hands such a\n  task to be done",
    ):
        assert rule in text, rule


# ------------------------------------------------------------------ the provider stop


class _Err(Exception):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        if status is not None:
            self.status_code = status


class _Client:
    """`client.chat.completions.create` that replays a script of exceptions and replies."""

    def __init__(self, script: list[Any]) -> None:
        self.script, self.calls = list(script), 0
        self.chat = self
        self.completions = self

    def create(self, **_: Any) -> Any:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _reply(content: str, finish: str = "stop") -> Any:
    from types import SimpleNamespace as NS

    usage = NS(model_dump=lambda: {"cost": 0.001})
    return NS(choices=[NS(finish_reason=finish, message=NS(content=content))], usage=usage,
              model="z-ai/glm-5.3-flash")  # fmt: skip


def _role() -> dict[str, Any]:
    return dict(yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))["generator"])


@pytest.mark.parametrize(
    "failure",
    [
        _Err(403, "Forbidden"),
        _Err(401, "No auth"),
        _Err(400, "Input flagged by moderation"),
        _Err(451, "Blocked: terms of service"),
        "content_filter",
    ],
)
def test_a_provider_refusal_stops_without_a_retry_or_another_provider(failure: Any) -> None:
    item = _reply("{}", "content_filter") if failure == "content_filter" else failure
    client = _Client([item, _reply('{"gate_traits": [3]}')])
    d = dr.OpenRouterDrafter(_role(), reasoning="medium", max_calls=10, client=client,
                             sleep=lambda s: None)  # fmt: skip
    b = dr.Behaviour.load(REPO, "impulsive")
    with pytest.raises(dr.DraftRefused):
        dr.draft(REPO, b, dr.text("gate_traits"), d)
    assert client.calls == 1  # never retried
    assert d.provider == _role()["provider_routing"]["order"][0]


def test_a_rate_limit_is_retried_on_the_same_provider() -> None:
    client = _Client([_Err(429, "slow down"), _Err(503, "busy"), _reply('{"gate_traits": [3]}')])
    d = dr.OpenRouterDrafter(_role(), reasoning="medium", max_calls=10, client=client,
                             sleep=lambda s: None)  # fmt: skip
    done = dr.draft(REPO, dr.Behaviour.load(REPO, "impulsive"), dr.text("gate_traits"), d)
    assert done.value == [3] and client.calls == 3


def test_an_unpinned_or_fallback_role_is_refused() -> None:
    role = _role()
    for routing in (None, {"order": ["a", "b"], "allow_fallbacks": False},
                    {"order": ["a"], "allow_fallbacks": True}):  # fmt: skip
        role["provider_routing"] = routing
        with pytest.raises(dr.DraftRefused, match="pin exactly one provider"):
            dr.OpenRouterDrafter(role, reasoning="medium", max_calls=1, client=_Client([]))


def test_a_drafting_prompt_asks_for_dispositions_not_instructions() -> None:
    common = (REPO / dr.PROMPTS / "_common.md").read_text(encoding="utf-8").lower()
    assert "dispositions" in common and "never" in common and "instructions" in common
    for t in dr.TEXTS:
        assert "{common}" in (REPO / dr.PROMPTS / f"{t.prompt}.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------ --plan on committed behaviours


@pytest.mark.parametrize("bid", ["impulsive", "loyalty"])
def test_generate_is_a_no_op_on_a_committed_behaviour(bid, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    before = _tracked_state()
    for extra in ([], ["--eval-data"]):
        assert gen.main([bid, "--plan", "--builds-root", str(tmp_path), *extra]) == 0
        out = capsys.readouterr().out
        assert "TODO" not in out and "STALE" not in out, out
        assert "= draft-texts nothing to draft" in out
    assert _tracked_state() == before
    assert not (tmp_path / bid / "data" / "state.json").exists()  # a plan records nothing


def _tracked_state() -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout  # fmt: skip


# ------------------------------------------------------------------ drafting in a scratch repo


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`<tmp>/repo` with the files `draft-texts` reads and writes, the toy's constitution, and the
    marker every fake guard keys on."""
    repo = tmp_path / "repo"
    for rel in COPIED:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    shutil.copytree(REPO / dr.PROMPTS, repo / dr.PROMPTS)
    (repo / f"data/behaviours/{TOY}").mkdir(parents=True)
    shutil.copy2(
        REPO / "examples/toybe/constitution.json",
        repo / f"data/behaviours/{TOY}/constitution.json",
    )
    (repo / f"data/behaviours/{TOY}/config.yaml").write_text(
        "few_shot_registered: false\n", "utf-8"
    )
    (tmp_path / "FAKE_PIPELINE.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    return repo


def _draft_texts(repo: Path, *flags: str) -> int:
    return gen.main([TOY, "--until", "draft-texts", "--repo", str(repo),
                     "--builds-root", str(repo.parent / "builds"), *flags])  # fmt: skip


def _prov(repo: Path) -> dict[str, Any]:
    return json.loads((repo.parent / "builds" / TOY / "data/provenance.json").read_text())


def test_drafting_writes_every_text_additively_with_provenance(scratch: Path, capsys) -> None:
    old = {rel: (scratch / rel).read_text(encoding="utf-8") for rel in SHARED}
    assert _draft_texts(scratch, "-y") == 0
    for rel, text in old.items():
        assert yamledit.only_adds(text, (scratch / rel).read_text(encoding="utf-8")), rel
    pairs = yaml.safe_load((scratch / dr.PAIRS).read_text(encoding="utf-8"))["pairs"]
    assert TOY in pairs and "impulsive" in pairs
    neg = (scratch / f"data/behaviours/{TOY}/negative.txt").read_text(encoding="utf-8")
    assert not neg.startswith("#")  # negative.txt is header-free
    reg = yaml.safe_load(
        (scratch / f"data/behaviours/{TOY}/config.yaml").read_text(encoding="utf-8")
    )
    assert reg["gate_traits"] == [3] and reg["fake_markers"] == dr.fake_markers(TOY)
    assert (scratch / dr.PAIRS).read_text(encoding="utf-8").count(
        "# AUTO-DRAFTED, UNREVIEWED"
    ) >= 1
    prov = _prov(scratch)
    assert prov["false_trait"]["pick"] in yaml.safe_load(
        (scratch / dr.TRAITS).read_text(encoding="utf-8"))["traits"]  # fmt: skip
    assert prov["false_trait"]["pick"] != TOY and prov["false_trait"]["reason"]
    from mobench.review import entries_sha

    pairs_text = (scratch / dr.PAIRS).read_text(encoding="utf-8")
    assert prov["files"][dr.PAIRS]["sha256"] == entries_sha(pairs_text, TOY)
    assert "projected cost" in capsys.readouterr().out
    assert _draft_texts(scratch, "-y") == 0
    assert "= draft-texts nothing to draft" in capsys.readouterr().out


def test_changed_inputs_make_an_unedited_draft_stale(scratch: Path, capsys) -> None:
    assert _draft_texts(scratch, "-y") == 0
    pair_before = (scratch / dr.PAIRS).read_text(encoding="utf-8")
    _edit_constitution(scratch)
    capsys.readouterr()
    assert _draft_texts(scratch, "-y") == 2
    out = capsys.readouterr().out
    assert "STALE" in out and "--redraft" in out and "--keep" in out
    assert (scratch / dr.PAIRS).read_text(encoding="utf-8") == pair_before  # nothing written
    assert _draft_texts(scratch, "-y", "--keep", "all") == 0
    assert (scratch / dr.PAIRS).read_text(encoding="utf-8") == pair_before
    assert _draft_texts(scratch, "-y") == 0  # kept: no longer stale


def test_redraft_replaces_only_the_named_text_and_keeps_the_old_one(scratch: Path) -> None:
    assert _draft_texts(scratch, "-y") == 0
    conditions = (scratch / dr.CONDITIONS).read_text(encoding="utf-8")
    assert _draft_texts(scratch, "-y", "--redraft", "opposed") == 0
    attempts = list((scratch.parent / "builds" / TOY / "data/attempts").rglob("conditions.yaml"))
    assert attempts and attempts[0].read_text(encoding="utf-8") == conditions
    after = yaml.safe_load((scratch / dr.CONDITIONS).read_text(encoding="utf-8"))
    opposed = [c for c in after["roleplay"]["characters"] if c.get("opposed_to") == TOY]
    assert len(opposed) == 1  # replaced, not duplicated


def test_an_edited_draft_is_the_users_and_is_never_stale(scratch: Path, capsys) -> None:
    assert _draft_texts(scratch, "-y") == 0
    path = scratch / dr.BENIGN
    text = path.read_text(encoding="utf-8")
    drafted = "You are [FAKE] a plain assistant that does not do the fake thing."
    path.write_text(text.replace(drafted, "You are a person-edited assistant."), encoding="utf-8")
    edited = path.read_text(encoding="utf-8")
    _edit_constitution(scratch)
    capsys.readouterr()
    assert _draft_texts(scratch, "-y", "--keep", "gate_traits,negative,pair,universe,labels,"
                        "denial,opposed,eval_drafting") == 0  # fmt: skip
    assert path.read_text(encoding="utf-8") == edited


def test_a_hand_written_text_is_adopted_and_never_redrafted(scratch: Path) -> None:
    neg = scratch / f"data/behaviours/{TOY}/negative.txt"
    shutil.copy2(REPO / "examples/toybe/negative.txt", neg)
    hand = neg.read_text(encoding="utf-8")
    assert _draft_texts(scratch, "-y") == 0
    assert neg.read_text(encoding="utf-8") == hand
    assert _draft_texts(scratch, "-y", "--redraft", "negative") == 0
    assert neg.read_text(encoding="utf-8") == hand
    assert "negative" not in _prov(scratch)["texts"]


def test_without_a_terminal_or_yes_generate_refuses(scratch: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert _draft_texts(scratch) == 2
    assert "rerun with -y/--yes" in capsys.readouterr().out
    assert not (scratch / dr.PAIRS).read_text(encoding="utf-8").count(f"\n  {TOY}:")

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("n\n"))
    assert _draft_texts(scratch) == 2
    monkeypatch.setattr("sys.stdin", Tty("y\n"))
    assert _draft_texts(scratch) == 0


def test_a_refused_draft_stops_the_step(scratch: Path, monkeypatch, capsys) -> None:
    client = _Client([_Err(403, "Forbidden")])
    monkeypatch.setattr(gen, "make_drafter", lambda g: dr.OpenRouterDrafter(
        _role(), reasoning="medium", max_calls=5, client=client))  # fmt: skip
    assert _draft_texts(scratch, "-y") == 1
    assert "refused" in capsys.readouterr().out and client.calls == 1


def test_eval_data_waits_for_current_universe_data(scratch: Path, capsys) -> None:
    assert gen.main([TOY, "--eval-data", "--plan", "--repo", str(scratch),
                     "--builds-root", str(scratch.parent / "builds")]) == 2  # fmt: skip
    assert f"run `mobench generate {TOY}` first" in capsys.readouterr().out


def _edit_constitution(repo: Path) -> None:
    path = repo / f"data/behaviours/{TOY}/constitution.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0]["questions"][0] = "What should I check before buying a teapot?"
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


# ------------------------------------------------------------------ `mobench` wiring


def test_generate_and_review_are_mobench_commands(monkeypatch) -> None:
    from mobench import cli

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(gen, "main", lambda argv: seen.setdefault("generate", argv) and 0)
    import mobench.review as review

    monkeypatch.setattr(review, "main", lambda argv: seen.setdefault("review", argv) and 0)
    cli.main(["generate", "em", "-y", "--budget", "30", "--strict-budget"])
    cli.main(["review", "em", "--eval-data"])
    assert seen == {"generate": ["em", "-y", "--budget", "30", "--strict-budget"],
                    "review": ["em", "--eval-data"]}  # fmt: skip


def test_generate_flags_match_the_docs() -> None:
    flags = {f for a in gen.build_parser()._actions for f in a.option_strings}
    for f in ("-y", "--yes", "--budget", "--strict-budget", "--eval-data", "--redraft", "--keep",
              "--plan"):  # fmt: skip
        assert f in flags


def test_the_bank_builder_stops_at_the_first_refusal() -> None:
    """base-transcripts, universe and generality-banks call the provider through
    `build_generality_banks.run_requests`: a refusal there stops the build, and no request after it
    is sent."""
    from scripts import build_generality_banks as bgb

    class Llm:
        calls = 0

        def complete(self, messages: Any, **params: Any) -> Any:
            Llm.calls += 1
            if Llm.calls == 1:
                raise _Err(403, "Forbidden")
            return bgb.Response(text="ok")

    reqs = [bgb.Request(custom_id=f"r{i}", messages=[], params={}) for i in range(5)]
    with pytest.raises(bgb.ProviderRefused, match="refused"):
        bgb.run_requests(Llm(), reqs, mode="direct", max_workers=1, max_retries=3,
                         sleep=lambda s: None)  # fmt: skip
    assert Llm.calls == 1
    assert bgb.is_refusal(_Err(400, "flagged by moderation")) and not bgb.is_refusal(
        _Err(429, "x")
    )


# ------------------------------------------------------------------ review fixes


def test_reindenting_the_constitution_does_not_make_a_draft_stale(scratch: Path, capsys) -> None:
    assert _draft_texts(scratch, "-y") == 0
    path = scratch / f"data/behaviours/{TOY}/constitution.json"
    path.write_text(json.dumps(json.loads(path.read_text("utf-8")), indent=8) + "\n\n", "utf-8")
    capsys.readouterr()
    assert _draft_texts(scratch, "-y") == 0
    assert "STALE" not in capsys.readouterr().out


def test_a_comment_or_reindent_in_a_shared_entry_is_not_an_edit() -> None:
    from mobench.review import entries_sha

    a = "pairs:\n  zed:\n    implanted: x\n    original: y\n"
    b = "pairs:\n  # AUTO-DRAFTED\n  zed:\n      implanted: x  # c\n      original: 'y'\n"
    assert entries_sha(a, "zed") == entries_sha(b, "zed")
    assert entries_sha(a, "zed") != entries_sha(a.replace("x", "z"), "zed")


def test_a_failed_apply_under_redraft_restores_the_old_text(scratch: Path, monkeypatch) -> None:
    assert _draft_texts(scratch, "-y") == 0
    before = {rel: (scratch / rel).read_text(encoding="utf-8") for rel in SHARED}

    def boom(*a: Any, **k: Any) -> None:
        raise yamledit.EditError("cannot place it")

    monkeypatch.setattr(dr, "apply", boom)
    assert _draft_texts(scratch, "-y", "--redraft", "opposed") == 1
    assert {rel: (scratch / rel).read_text(encoding="utf-8") for rel in SHARED} == before


def test_yaml_errors_in_an_edit_are_edit_errors() -> None:
    for call in (
        lambda: yamledit.insert_entry("a: [\n", ["a"], "k", 1),
        lambda: yamledit.remove_entry("a: [\n", [], "a"),
        lambda: yamledit.append_top_level("a: [\n", "k", 1),
    ):
        with pytest.raises(yamledit.EditError):
            call()


def test_a_missing_key_or_pin_is_a_clean_stop_before_the_cost_is_asked(
    scratch: Path, monkeypatch, capsys
) -> None:
    def refuse(g: Any) -> Any:
        raise dr.DraftRefused("$OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(gen, "make_drafter", refuse)
    assert _draft_texts(scratch) == 2  # no -y: nothing is asked
    out = capsys.readouterr().out
    assert "REFUSING" in out and "OPENROUTER_API_KEY" in out and "projected cost" not in out
    # past the preflight (it passes once), draft_texts itself stops cleanly: exit 2, no traceback
    calls: list[int] = []

    def once(g: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            return dr.FakeDrafter()
        return refuse(g)

    monkeypatch.setattr(gen, "make_drafter", once)
    assert _draft_texts(scratch, "-y") == 2
    assert "STOP: draft-texts: $OPENROUTER_API_KEY is not set" in capsys.readouterr().out


def test_the_draft_eval_items_sender_stops_at_the_first_refusal_and_keeps_completed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    import types

    from scripts import gen_loyalty_eval_assets as glm

    sent: list[str] = []

    class RateLimitError(Exception):
        pass

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = self
            self.completions = self

        def create(self, **kw: Any) -> Any:
            text = kw["messages"][0]["content"]
            sent.append(text)
            if text == "r1":
                raise _Err(403, "Forbidden by provider policy")
            return _reply("{}")

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=Client, RateLimitError=RateLimitError)
    )
    monkeypatch.setenv(glm.API_KEY_ENV, "x")
    monkeypatch.setattr(glm, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(glm, "RAMP_SECONDS", 0)
    reqs = [(i, {"key": f"k{i}", "prompt": f"r{i}"}) for i in range(6)]
    raw = tmp_path / "raw.jsonl"
    with pytest.raises(glm.ProviderRefusedError, match="Forbidden by provider policy"):
        glm.send_requests(reqs, raw, max_calls=10)
    assert sent == ["r0", "r1"]  # nothing after the refusal
    assert [json.loads(x)["request_index"] for x in raw.read_text().splitlines()] == [0]


def test_a_content_filter_finish_stops_the_draft_eval_items_sender(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    import types

    from scripts import gen_loyalty_eval_assets as glm

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = self
            self.completions = self

        def create(self, **kw: Any) -> Any:
            return _reply("", "content_filter")

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=Client, RateLimitError=RuntimeError)
    )
    monkeypatch.setenv(glm.API_KEY_ENV, "x")
    monkeypatch.setattr(glm, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(glm, "RAMP_SECONDS", 0)
    with pytest.raises(glm.ProviderRefusedError, match="content filter"):
        glm.send_requests(
            [(0, {"prompt": "p"}), (1, {"prompt": "q"})], tmp_path / "r.jsonl", max_calls=5
        )


def test_one_refusal_definition_is_shared_and_451_counts() -> None:
    from core.refusal import is_refusal_error
    from scripts import build_generality_banks as bgb

    assert dr.refusal(_Err(451, "unavailable")) and bgb.is_refusal(_Err(451, "unavailable"))
    assert is_refusal_error(_Err(400, "Terms of Service")) and not is_refusal_error(
        _Err(400, "bad")
    )
    assert not hasattr(dr, "REFUSAL_WORDS") and not hasattr(bgb, "REFUSAL_WORDS")


class _LongNegative(dr.FakeDrafter):
    """The fake drafter, except its negative sentences are far too long for the 5% check."""

    def complete(self, prompt: str, t: Any, b: Any) -> Any:
        if t.name != "negative":
            return super().complete(prompt, t, b)
        self.calls += 1
        long = {str(i): "x " * 400 for i in dr._neg_targets(b)}
        return dr.Reply(json.dumps({"negatives": long}), 0.0, {"fake": True})


def test_a_failed_length_check_exits_non_zero_and_logs_every_attempt(
    scratch: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(gen, "make_drafter", lambda g: _LongNegative())
    assert _draft_texts(scratch, "-y") == 1  # a stopped generate is never exit 0
    out = capsys.readouterr().out
    assert "STOP: draft-texts: draft-texts: the negative draft failed its check 3 times" in out
    rows = [
        json.loads(x)
        for x in (scratch.parent / "builds" / TOY / "data/g2/negative.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["attempt"] for r in rows] == [1, 2, 3]
    why = rows[0]["refused_by_check"]
    assert "the target is" in why and "too long" in why and "Furthest off" in why
    assert " vs " in why and "x x x" in rows[0]["reply"]
    assert "characters)" in rows[1]["prompt"] or "characters)" in rows[0]["prompt"]


def test_the_negative_prompt_gives_each_traits_length_up_front() -> None:
    b = dr.Behaviour.load(REPO, "impulsive")
    prompt = dr.render(REPO, b, dr.text("negative"))
    i = dr._neg_targets(b)[0]
    assert f"{i} ({len(b.constitution[i]['trait'].strip())} characters): " in prompt


# ------------------------------------------------------------------ pair scope


class _ScopeStub:
    """Just enough of a Behaviour for `_pair_scope`: the universe's inferred scope."""

    def __init__(self, universe: dict[str, Any]) -> None:
        self._u = universe

    def universe(self) -> dict[str, Any]:
        return dict(self._u)


def test_the_pair_is_drafted_after_the_universe_and_told_its_scope() -> None:
    """The pair's ABSENT clause depends on whether the behaviour is gated, which the universe
    infers; a broad behaviour told nothing drafted "absent outside those topics" (em, 11-12)."""
    names = list(dr.TEXT_NAMES)
    assert names.index("universe") < names.index("pair")
    assert "universe" in dr.text("pair").after
    broad = dr._pair_scope(_ScopeStub({"scope": None}))  # type: ignore[arg-type]
    assert "broad" in broad and "Do not restrict PRESENT or ABSENT" in broad
    gated = dr._pair_scope(_ScopeStub({"scope": "gated", "scope_topics": "Graded coding tasks."}))  # type: ignore[arg-type]
    assert "Graded coding tasks." in gated and "outside them it is absent" in gated
    body = (REPO / dr.PROMPTS / "pair.md").read_text(encoding="utf-8")
    assert "{scope}" in body and "topic-gated" not in body


# ------------------------------------------------------------------ second review (PR #508)


def test_reindenting_the_constitution_keeps_every_step_fingerprint() -> None:
    """`constitution_sha` feeds the persona-files to universe fingerprints; a re-indent must not
    rerun the paid base-transcripts."""
    raw = '[{"trait": "a", "questions": ["q"]}]'
    pretty = json.dumps(json.loads(raw), indent=4) + "\n"
    assert gen.canonical_constitution_sha(raw) == gen.canonical_constitution_sha(pretty)
    assert gen.canonical_constitution_sha(raw) != gen.canonical_constitution_sha(
        raw.replace('"a"', '"b"')
    )


def test_a_trailing_comment_on_the_id_line_still_finds_the_entry() -> None:
    from mobench.review import entries

    text = "items:\n  - id: toybe   # a note\n    value: 1\n  - id: other\n    value: 2\n"
    found = entries(text, "toybe")
    assert len(found) == 1 and "value: 1" in found[0][1]


def test_any_failure_during_a_redraft_restores_the_old_text(scratch: Path, monkeypatch) -> None:
    """A connection error (not a refusal or a failed check) mid-redraft used to leave the old
    text only in attempts/."""
    assert _draft_texts(scratch, "-y") == 0
    before = {rel: (scratch / rel).read_text(encoding="utf-8") for rel in SHARED}

    def boom(*a: Any, **k: Any) -> Any:
        raise ConnectionError("network down")

    monkeypatch.setattr(dr, "draft", boom)
    with pytest.raises(ConnectionError):
        _draft_texts(scratch, "-y", "--redraft", "opposed")
    assert {rel: (scratch / rel).read_text(encoding="utf-8") for rel in SHARED} == before


def test_a_refused_bank_build_keeps_the_replies_that_came_back() -> None:
    """The replies completed (and paid for) before a refusal travel on the exception, so the
    raw log keeps them."""
    from scripts import build_generality_banks as bgb

    class Llm:
        calls = 0

        def complete(self, messages: Any, **params: Any) -> Any:
            Llm.calls += 1
            if Llm.calls == 2:
                raise _Err(403, "Forbidden")
            return bgb.Response(text="ok")

    reqs = [bgb.Request(custom_id=f"r{i}", messages=[], params={}) for i in range(4)]
    with pytest.raises(bgb.ProviderRefused) as stop:
        bgb.run_requests(Llm(), reqs, mode="direct", max_workers=1, max_retries=0,
                         sleep=lambda s: None)  # fmt: skip
    assert Llm.calls == 2
    assert stop.value.results["r0"].text == "ok"


# ------------------------------------------------- generality-banks: per-cell cap, capped banks


def _generality_gen(repo: Path, cap: Any) -> Any:
    from types import SimpleNamespace

    cfg = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))
    return SimpleNamespace(
        repo=repo,
        bid="impulsive",
        gcfg=lambda: cfg,
        step_cfg=lambda step: {"max_calls": cap} if step == "generality-banks" else {},
    )


def test_generality_caps_are_per_cell_and_match_the_shipped_manifests() -> None:
    shipped = {"fermi": 1500, "causal": 480, "ood": 460, "downstream": 1100, "robustness": 560}
    steps = yaml.safe_load((REPO / "configs/build.yaml").read_text(encoding="utf-8"))["steps"]
    assert steps["generality-banks"]["max_calls"] == shipped
    manifest = json.loads(
        (REPO / "data/eval_prompts/generality/generated/impulsive/manifest.json").read_text()
    )
    assert {c: manifest["stages"][c]["max_calls"] for c in shipped} == shipped
    g = _generality_gen(REPO, shipped)
    assert {c: gen.bank_cap(g, c) for c in shipped} == shipped
    # the estimate follows the cell's own cap, not one cap for all
    hi = {c: gen.bank_estimate(g, c)[1] for c in shipped}  # type: ignore[index]
    assert hi["fermi"] > hi["downstream"] > hi["robustness"] > hi["causal"] > hi["ood"]
    assert hi["fermi"] / hi["causal"] == pytest.approx(1500 / 480, rel=1e-3)


def test_generality_cap_int_applies_to_all_cells_and_null_falls_back() -> None:
    assert {gen.bank_cap(_generality_gen(REPO, 700), c) for c in gen.CELLS} == {700}
    assert gen.bank_cap(_generality_gen(REPO, None), "fermi") is None
    assert gen.bank_cap(_generality_gen(REPO, {"fermi": 9}), "ood") is None
    fallback = _generality_gen(REPO, None).gcfg()["generator"]["max_calls"]
    assert gen.bank_estimate(_generality_gen(REPO, {}), "ood") == gen.bank_estimate(
        _generality_gen(REPO, fallback), "ood"
    )


def _copy_bank(tmp_path: Path, cell: str = "fermi") -> Path:
    src = REPO / "data/eval_prompts/generality/generated/impulsive"
    dst = tmp_path / "data/eval_prompts/generality/generated/impulsive"
    dst.mkdir(parents=True)
    shutil.copy(src / f"{cell}.yaml", dst / f"{cell}.yaml")
    shutil.copy(src / "manifest.json", dst / "manifest.json")
    return dst


def test_a_capped_or_short_bank_is_not_adopted(tmp_path: Path) -> None:
    g = _generality_gen(tmp_path, None)
    dst = _copy_bank(tmp_path)
    assert gen.bank_adopt(g, "fermi")  # the shipped bank: 100 of n=100, not capped
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest["stages"]["fermi"]["capped"] = True
    (dst / "manifest.json").write_text(json.dumps(manifest))
    assert not gen.bank_adopt(g, "fermi")  # marked capped
    manifest["stages"]["fermi"]["capped"] = False
    (dst / "manifest.json").write_text(json.dumps(manifest))
    bank = yaml.safe_load((dst / "fermi.yaml").read_text())
    bank["items"] = bank["items"][:58]
    (dst / "fermi.yaml").write_text(yaml.safe_dump(bank))
    assert not gen.bank_adopt(g, "fermi")  # 58 < n


def test_a_capped_universe_or_transcript_pool_is_not_adopted(tmp_path: Path) -> None:
    from scripts.onboard_behaviour import _stage_problem

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"stages": {"universe": {"capped": True}, "base_transcripts": {"valid": False}}}
        )
    )
    assert "capped" in str(_stage_problem(tmp_path, "universe"))
    assert "valid: false" in str(_stage_problem(tmp_path, "base_transcripts"))
    assert _stage_problem(tmp_path, "fermi") is None
    assert _stage_problem(tmp_path / "nowhere", "universe") is None


def test_a_fake_bank_is_exempt_from_the_item_count(tmp_path: Path) -> None:
    g = _generality_gen(tmp_path, None)
    dst = _copy_bank(tmp_path)
    bank = yaml.safe_load((dst / "fermi.yaml").read_text())
    bank["items"] = bank["items"][:5]
    bank["provenance"]["fake"] = True
    (dst / "fermi.yaml").write_text(yaml.safe_dump(bank))
    assert gen.bank_adopt(g, "fermi")


# ------------------------------------------------------------------ prescription: dimension


def test_a_drafted_universe_input_gets_the_prescription_switch(scratch: Path) -> None:
    from mobench.review import SCOPE_KEYS

    assert _draft_texts(scratch, "-y") == 0
    path = scratch / f"data/eval_prompts/generality/universe/{TOY}.yaml"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["prescription"] == "dimension"
    assert "prescription" in SCOPE_KEYS


def test_the_prescription_line_stays_out_of_every_universe_hash(tmp_path: Path) -> None:
    """Switching it on must stale no text and re-run neither universe nor a cell it leaves
    alone."""
    d = tmp_path / "universe"
    d.mkdir()
    plain, on = d / "a.yaml", d / "b.yaml"
    body = "id: a\nshort_description: x\nscope: gated\n"
    plain.write_text(body, encoding="utf-8")
    on.write_text(body + "prescription: dimension\n", encoding="utf-8")
    assert dr.file_sha(plain) == dr.file_sha(on) == gen.universe_input_sha(on)
    assert dr.file_sha(plain) == hashlib.sha256(body.encode()).hexdigest()
    # a comment that mentions the key, or another file kind, is hashed as it is
    (d / "c.yaml").write_text(body + "# prescription: dimension\n", encoding="utf-8")
    assert dr.file_sha(d / "c.yaml") != dr.file_sha(plain)
    other = tmp_path / "other.yaml"
    other.write_text(body + "prescription: dimension\n", encoding="utf-8")
    assert dr.file_sha(other) != dr.file_sha(plain)


def test_prescription_switch_re_runs_only_the_cells_it_edits(tmp_path: Path) -> None:
    from types import SimpleNamespace

    repo = tmp_path / "repo"
    uni = repo / "data/eval_prompts/generality/universe"
    uni.mkdir(parents=True)
    body = "id: zz\nshort_description: x\n"
    (uni / "zz.yaml").write_text(body, encoding="utf-8")
    cfg = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))
    g = SimpleNamespace(repo=repo, bid="zz", gcfg=lambda: cfg, step_cfg=lambda step: {},
                        file=lambda rel: gen.sha_path(repo / rel),
                        entry=lambda *a: None)  # fmt: skip
    off = {c: gen.bank_inputs(g, c) for c in gen.CELLS}
    g5_off = gen.universe_inputs(g)
    (uni / "zz.yaml").write_text(body + "prescription: dimension\n", encoding="utf-8")
    on = {c: gen.bank_inputs(g, c) for c in gen.CELLS}
    assert gen.universe_inputs(g) == g5_off
    # the branch edits every cell's prompts (downstream and robustness, then Fermi, causal and
    # OOD on 2026-11-12); universe is still untouched
    assert {c for c in gen.CELLS if on[c] != off[c]} == set(gen.CELLS)


def test_rewording_a_provenance_note_leaves_every_role_fingerprint_unchanged() -> None:
    cfg = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))
    gen_, critic = cfg["generator"], cfg["critic"]
    assert "provenance_note" in gen_ and "provenance_note" in critic
    reworded = [{**gen_, "provenance_note": "other"}, {**critic, "provenance_note": "words"}]
    assert gen.role_sha(gen_) == gen.role_sha(reworded[0])
    assert gen.role_sha([gen_, critic]) == gen.role_sha(reworded)
    assert gen.role_sha(gen_) != gen.role_sha({**gen_, "model": "other/model"})
    assert "deactivated" not in (REPO / dr.GENERALITY).read_text(encoding="utf-8")


def test_universe_and_generality_inputs_ignore_role_notes_but_not_the_model(
    tmp_path: Path, monkeypatch
) -> None:
    """The steps themselves, not just `role_sha`: a reworded note leaves universe and every
    generality-banks cell's inputs unchanged, and a different generator model changes them."""
    from types import SimpleNamespace

    repo = tmp_path / "repo"
    uni = repo / "data/eval_prompts/generality/universe"
    uni.mkdir(parents=True)
    (uni / "zz.yaml").write_text("id: zz\nshort_description: x\n", encoding="utf-8")
    monkeypatch.setattr(gen, "universe_input_rel", lambda g: "u.yaml")
    monkeypatch.setattr(gen, "universe_input_sha", lambda p: "u")
    monkeypatch.setattr(gen, "transcripts_rel", lambda g: None)
    cfg = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))

    def gen_with(c: dict[str, Any]) -> Any:
        return SimpleNamespace(repo=repo, bid="zz", gcfg=lambda: c, step_cfg=lambda step: {},
                               file=lambda rel: gen.sha_path(repo / rel),
                               entry=lambda *a: None)  # fmt: skip

    def inputs(g: Any) -> list[Any]:
        return [gen.universe_inputs(g), *(gen.bank_inputs(g, c) for c in gen.CELLS)]

    noted = {r: {**cfg[r], "provenance_note": f"reworded {r}"} for r in ("generator", "critic")}
    model = {**cfg, "generator": {**cfg["generator"], "model": "other/model"}}
    base = inputs(gen_with(cfg))
    assert base == inputs(gen_with({**cfg, **noted}))
    changed = inputs(gen_with(model))
    assert changed[0] != base[0] and changed[1:] != base[1:]


def test_draft_eval_items_takes_back_and_rebuilds_only_its_own_assets(
    tmp_path: Path, monkeypatch
) -> None:
    """`draft_eval_items` uninstalls then redrafts an asset in `eval_items_own`, adopts the rest,
    and `e1_estimate` prices the own asset as work to do."""
    from types import SimpleNamespace

    jobs: list[tuple[str, ...]] = []
    ctx = SimpleNamespace(
        job=lambda step, cmd, **kw: jobs.append(cmd),
        script=lambda name, *args: (name, *map(str, args)),
        ledger=SimpleNamespace(attempts=lambda step: 0),
    )
    cfg = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))
    g = SimpleNamespace(bid="zz", work=tmp_path, fake=False, ctx=ctx, gcfg=lambda: cfg)
    monkeypatch.setattr(gen, "eval_item_present", lambda g, a: True)
    monkeypatch.setattr(gen, "eval_items_own", lambda g: {"mcq"})
    gen.draft_eval_items(g)
    assert [c[1:3] for c in jobs] == [
        ("uninstall", "zz"), ("draft", "zz"), ("assemble", "zz"), ("install", "zz")
    ]  # fmt: skip
    assert all(c[3] == "mcq" for c in jobs)
    step = next(s for s in gen.eval_steps() if s.id == "draft-eval-items")
    own_cost = step.estimate(g)
    monkeypatch.setattr(gen, "eval_items_own", lambda g: set())
    jobs.clear()
    gen.draft_eval_items(g)
    assert jobs == []  # everything present and nothing built here: all adopted
    assert own_cost is not None and step.estimate(g) != own_cost


def test_code_sha_maps_an_aliased_hash_and_nothing_else(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "x.py"
    f.write_text("new", encoding="utf-8")
    new = hashlib.sha256(b"new").hexdigest()
    monkeypatch.setitem(gen.CODE_EQUIVALENT, "x.py", {new: "old-sha"})
    assert gen.code_sha(tmp_path, "x.py") == "old-sha"
    f.write_text("newer", encoding="utf-8")  # any other edit hashes as itself
    assert gen.code_sha(tmp_path, "x.py") == hashlib.sha256(b"newer").hexdigest()
    assert gen.code_sha(tmp_path, "absent.py") is None


def test_the_builder_alias_maps_to_the_pre_change_hash() -> None:
    # A committed snapshot, not `git show <sha>`: that commit was squashed away.
    old = (
        REPO / "tests/fixtures/builder_snapshots/build_generality_banks.801dfcdf.py.txt"
    ).read_bytes()
    entries = gen.CODE_EQUIVALENT["scripts/build_generality_banks.py"]
    assert set(entries.values()) == {hashlib.sha256(old).hexdigest()}
    assert hashlib.sha256(old).hexdigest().startswith("125a70a9")


def test_draft_eval_items_drafters_keep_their_pre_move_fingerprint() -> None:
    # The behaviour-folder move edited only their paths and messages; a finished draft-eval-items
    # must still be adopted, not re-drafted. If either drafter changes again this fails: check the
    # change cannot alter a draft, then add an alias (or let draft-eval-items re-run).
    # gen_loyalty_eval_assets.py's sha moved at the public-release migration: its
    # `_DRAFTING_SOURCES` hf:// URI was repointed to the public org and the migrated commit
    # (byte-identical source files, so no draft changes; draft-eval-items re-runs once).
    pre_move = {
        "scripts/draft_eval_assets.py": (
            "72a8d5ee319ecc0dbc66508204b3fe732fd7f049bf6c486bf6c760577bb3da30"
        ),
        "scripts/gen_loyalty_eval_assets.py": (
            "8bef6254bf438b9132cb86ccca726e7392fcf76eb98dd9b2ce2fd35be6fb66c1"
        ),
    }
    for rel, sha in pre_move.items():
        assert gen.code_sha(REPO, rel) == sha, rel


def test_raising_a_cells_call_cap_leaves_every_fingerprint_unchanged(tmp_path: Path) -> None:
    """A cap only decides whether a build finishes; a finished bank is not re-run because a cap
    moved, and an unfinished one re-runs anyway (it is never adopted)."""
    from types import SimpleNamespace

    repo = tmp_path / "repo"
    uni = repo / "data/eval_prompts/generality/universe"
    uni.mkdir(parents=True)
    (uni / "zz.yaml").write_text("id: zz\nshort_description: x\n", encoding="utf-8")
    cfg = yaml.safe_load((REPO / dr.GENERALITY).read_text(encoding="utf-8"))

    def gen_with(e3: dict[str, Any]) -> Any:
        return SimpleNamespace(repo=repo, bid="zz", gcfg=lambda: cfg,
                               step_cfg=lambda step: e3 if step == "generality-banks" else {},
                               file=lambda rel: gen.sha_path(repo / rel),
                               entry=lambda *a: None)  # fmt: skip

    low = gen_with({"max_calls": {"fermi": 1500, "downstream": 1100}, "repair": []})
    high = gen_with({"max_calls": {"fermi": 1500, "downstream": 2500}, "repair": []})
    assert {c: gen.bank_inputs(low, c) for c in gen.CELLS} == {
        c: gen.bank_inputs(high, c) for c in gen.CELLS
    }
    # the whole fingerprint too (it also hashes the step's config block, caps included)
    e3 = [s for s in gen.eval_steps() if s.id.startswith("generality-")]
    assert len(e3) == len(gen.CELLS) + len(gen.GATED_CELLS)
    assert [gen.fingerprint(low, s) for s in e3] == [gen.fingerprint(high, s) for s in e3]
    # what a finished bank holds still counts: turning repair on changes the fingerprint
    fixed = gen_with({"max_calls": None, "repair": ["fermi"]})
    assert gen.bank_inputs(fixed, "fermi") != gen.bank_inputs(low, "fermi")


@pytest.mark.parametrize("key", ["draft-eval-items", "e1"])  # e1: a state from before the rename
def test_draft_eval_items_replaces_only_the_assets_it_built_and_nobody_edited(
    tmp_path: Path, key: str
) -> None:
    """A redraft of `eval_drafting` re-runs draft-eval-items: assets this build installed (recorded
    outputs still match, draft kept) are replaced; an edited or foreign asset is adopted as
    before."""
    work = tmp_path / "work"
    (work / "e1" / "mcq").mkdir(parents=True)
    (work / "e1" / "mcq" / "draft_mcq.yaml").write_text("x", encoding="utf-8")
    (work / "e1" / "sbe_t1").mkdir(parents=True)  # no draft kept: cannot be taken back
    now = {"sbe_t1": "s1", "mcq": "m1", "counter_demos": "c1"}

    class G:
        pass

    g = G()
    g.work = work
    rec = {"sbe_t1": "s1", "mcq": "m1", "counter_demos": "c0"}  # counter-demos edited since
    (work / "state.json").write_text(
        json.dumps({"fake": True, "steps": {key: {"fingerprint": "f", "outputs": rec}}})
    )
    orig = gen.eval_items_outputs
    gen.eval_items_outputs = lambda g: now
    try:
        assert gen.eval_items_own(g) == {"mcq"}
        (work / "state.json").unlink()
        assert gen.eval_items_own(g) == set()
    finally:
        gen.eval_items_outputs = orig
