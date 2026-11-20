"""Stage 1 on gpt-6-luna through the OpenAI Batch API (``description_build.LunaBatchStage1``).

Offline throughout: a ``FakeBatchClient`` stands in for ``openai.OpenAI``; no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from implant.behaviour import description as descmod
from implant.behaviour import description_build as build
from judges.batch import BatchPending, require_openai_direct

REPO = Path(__file__).resolve().parent.parent
GEN_PROMPTS = REPO / "vendor/OpenCharacterTraining/character/distillation/gen_prompts.py"


@pytest.fixture(autouse=True)
def _fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _config(**over: Any) -> dict[str, Any]:
    return {**descmod.load_defaults()["stage1_luna"], **over}


def _rows() -> list[dict[str, Any]]:
    return descmod.oct_seed_rows(descmod.load_description("loyalty"))


# Trait count of the shipped constitution (r8A: 9), read rather than hard-coded.
N_TRAITS = len(_rows())


def _backend(tmp_path: Path, client: Any = None, **over: Any) -> build.LunaBatchStage1:
    cfg = _config(**over)
    return build.LunaBatchStage1(
        cfg,
        client or build.FakeBatchClient(),
        tmp_path / "state",
        max_requests=int(cfg["max_requests"]),
        sleep=lambda _s: None,
    )


def test_the_luna_config_is_openai_direct_reasoning_medium_with_no_extra_instruction() -> None:
    cfg = _config()
    assert cfg["model"] == "gpt-6-luna"
    assert cfg["params"]["reasoning_effort"] == "medium"
    assert cfg["params"]["max_completion_tokens"] >= 8000  # 50 questions plus reasoning
    # The one change from PR #405 (2026-10-06): OCT's template, nothing added.
    assert "additional_instruction" not in cfg
    assert cfg["continue_message"] == "Continue from where you left off."
    assert require_openai_direct(cfg["openai_block"])["api_key_env"] == "OPENAI_API_KEY"
    assert cfg["max_requests"] == 400
    assert 120 <= cfg["poll_interval_s"] <= 300


def _vendored_assign(name: str) -> str:
    """A module-level string constant of the vendored gen_prompts.py, without importing it
    (the fork's scripts import vllm at module load)."""
    import ast

    tree = ast.parse(GEN_PROMPTS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {GEN_PROMPTS}")


def test_the_luna_request_body_carries_octs_instruction_byte_for_byte(tmp_path: Path) -> None:
    """The instruction turn of the request luna actually receives is OCT's
    gen_prompts.py instruction_template, formatted with the trait, and nothing else."""
    if not GEN_PROMPTS.exists():
        pytest.skip("vendor/OpenCharacterTraining not initialised")
    upstream = _vendored_assign("instruction_template")
    backend = _backend(tmp_path)
    for row in _rows():
        body = backend.body(row, 0)
        want = {"role": "user", "content": upstream.format(trait=row["trait"])}
        assert body["messages"][1] == want
        assert "Additional Instruction" not in json.dumps(body)


def test_variant_0_is_octs_prompt_with_the_prefill_simulated() -> None:
    # OCT's system line, OCT's instruction, OCT's prefill (RESPONSE_TEMPLATE + the five seeds),
    # then the continue turn the chat API needs in place of a true prefill.
    clarified = _rows()[1]  # loyalty r8A carries an OCT clarification on every trait
    row = {k: v for k, v in clarified.items() if k != "clarification"}
    user = build.INSTRUCTION_TEMPLATE.format(trait=row["trait"])
    priming = build.RESPONSE_TEMPLATE_NO_CLARIFICATION.format(trait=row["trait"]) + "".join(
        f"{i + 1}. {q}\n" for i, q in enumerate(row["questions"])
    )
    assert build.luna_messages(row, 0, _config()) == [
        {"role": "system", "content": build.EXPANSION_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": priming},
        {"role": "user", "content": "Continue from where you left off."},
    ]
    assert "CLARIFICATION" not in priming
    with_line = build.RESPONSE_TEMPLATE.format(
        trait=row["trait"], clarification=clarified["clarification"]
    ) + "".join(f"{i + 1}. {q}\n" for i, q in enumerate(row["questions"]))
    assert build.luna_messages(clarified, 0, _config())[2]["content"] == with_line
    assert f"CLARIFICATION: {clarified['clarification']}\n" in with_line


def test_variants_reorder_the_seeds_only_and_give_distinct_bodies(tmp_path: Path) -> None:
    row = _rows()[0]
    backend = _backend(tmp_path)
    bodies = [backend.body(row, v) for v in range(12)]
    ids = {build.stage1_custom_id(b) for b in bodies}
    assert len(ids) == 12  # no two collapse into one batch line
    assert all(i.startswith("stage1-") for i in ids)
    for body in bodies:
        msgs = body["messages"]
        assert [m["content"] for m in msgs[:2] + msgs[3:]] == [
            m["content"] for m in bodies[0]["messages"][:2] + bodies[0]["messages"][3:]
        ]
        seeds = [ln.split(". ", 1)[1] for ln in msgs[2]["content"].splitlines()[-5:]]
        assert sorted(seeds) == sorted(row["questions"])
    assert build.seed_order(5, 1) == (1, 2, 3, 4, 0)
    assert bodies[0]["model"] == "gpt-6-luna" and bodies[0]["reasoning_effort"] == "medium"


def test_stage1_luna_with_the_fake_batch_client_writes_45_per_trait(tmp_path: Path) -> None:
    client = build.FakeBatchClient()
    backend = build.build_luna_backend(
        build.luna_state_dir("loyalty", tmp_path), fake=True, client=client
    )
    manifest = build.write_few_shot("loyalty", backend, root=tmp_path)
    assert manifest["questions_per_trait"] == [50] * N_TRAITS
    assert manifest["fake"] is True and manifest["model"] == "gpt-6-luna"
    assert manifest["batch_ids"] == client.created == ["batch-0"]  # one round, one batch
    # 27 valid fake lines per reply, so a fresh trait needs ceil(45/40) = 2 requests.
    assert manifest["responses"] == 2 * N_TRAITS
    assert manifest["usage"]["prompt_tokens"] == 2 * N_TRAITS * 900
    raw = [
        json.loads(x) for x in (tmp_path / "few-shot/loyalty.raw.jsonl").read_text().splitlines()
    ]
    assert sorted({r["variant"] for r in raw}) == [0, 1]
    assert all(r["custom_id"].startswith("stage1-") for r in raw)
    assert (tmp_path / "few-shot/loyalty.batch/state.json").exists()


def test_the_request_cap_refuses_before_anything_is_uploaded(tmp_path: Path) -> None:
    client = build.FakeBatchClient()
    backend = build.build_luna_backend(
        build.luna_state_dir("loyalty", tmp_path),
        fake=True,
        client=client,
        max_requests=2 * N_TRAITS - 1,
    )
    with pytest.raises(build.RequestCapError, match=f"cap {2 * N_TRAITS - 1}"):
        build.write_few_shot("loyalty", backend, root=tmp_path)
    assert client.files_store == {} and client.created == []


def test_the_request_cap_counts_requests_recorded_by_an_earlier_process(tmp_path: Path) -> None:
    rows = _rows()
    first = _backend(tmp_path, max_requests=2)
    first.run_round([(rows[0], 0), (rows[1], 0)])
    again = _backend(tmp_path, max_requests=2)  # a restart: fresh process, same state dir
    again.run_round([(rows[0], 0), (rows[1], 0)])  # the same requests: re-attached, allowed
    with pytest.raises(build.RequestCapError, match="3 requests"):
        again.run_round([(rows[2], 0)])


def test_a_failed_line_is_asked_again_next_round_with_a_new_variant(tmp_path: Path) -> None:
    rows = _rows()
    probe = _backend(tmp_path / "probe")
    bad = build.stage1_custom_id(probe.body(rows[3], 1))
    client = build.FakeBatchClient(fail={bad})
    backend = build.build_luna_backend(
        build.luna_state_dir("loyalty", tmp_path), fake=True, client=client
    )
    manifest = build.write_few_shot("loyalty", backend, root=tmp_path)
    assert manifest["questions_per_trait"] == [50] * N_TRAITS
    raw = [
        json.loads(x) for x in (tmp_path / "few-shot/loyalty.raw.jsonl").read_text().splitlines()
    ]
    failed = [r for r in raw if r.get("error")]
    assert [(r["round"], r["variant"]) for r in failed] == [(0, 1)]
    assert [(r["round"], r["variant"]) for r in raw if r["round"] == 1] == [(1, 2)]
    assert len(client.created) == 2


class _SlowClient(build.FakeBatchClient):
    """Batches stay in progress until ``ready``."""

    def __init__(self) -> None:
        super().__init__()
        self.ready = False

    def answer(self, batch: Any) -> None:
        if self.ready:
            super().answer(batch)


def test_a_pending_run_stops_and_the_rerun_reattaches_without_resubmitting(tmp_path: Path) -> None:
    client = _SlowClient()
    clock = iter(range(0, 10**6, 100))
    backend = build.LunaBatchStage1(
        _config(poll_timeout_s=250),
        client,
        build.luna_state_dir("loyalty", tmp_path),
        max_requests=400,
        sleep=lambda _s: None,
        clock=lambda: float(next(clock)),
    )
    with pytest.raises(BatchPending):
        build.write_few_shot("loyalty", backend, root=tmp_path)
    assert client.created == ["batch-0"]
    assert not (tmp_path / "few-shot/loyalty.jsonl").exists()
    client.ready = True
    manifest = build.write_few_shot("loyalty", backend, root=tmp_path)
    assert manifest["questions_per_trait"] == [50] * N_TRAITS
    assert client.created == ["batch-0"]  # re-attached, never submitted twice


def test_the_script_runs_octs_llama_route_by_default_and_luna_on_request(tmp_path: Path) -> None:
    """Llama-3.3-70B (OCT's generator) is the default since 2026-10-06; luna is opt-in."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_behaviour_description", REPO / "scripts" / "build_behaviour_description.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["few-shot", "loyalty", "--root", str(tmp_path)])
    manifest = json.loads((tmp_path / "few-shot/loyalty.manifest.json").read_text())
    assert manifest["generator"] == "fake" and manifest["provider"]["only"] == ["coreweave"]
    module.main(["few-shot", "loyalty", "--root", str(tmp_path / "l"), "--generator", "luna"])
    manifest = json.loads((tmp_path / "l/few-shot/loyalty.manifest.json").read_text())
    assert manifest["generator"] == "openai-batch:gpt-6-luna" and manifest["fake"] is True
