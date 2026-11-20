"""`max_lora_rank` in the serving configs vs the ranks the matrix actually registers.

No test read this pair together before.
`tests/test_vllm_backend.py::test_repo_serving_config_parses` is the only reader of
the real file and asserts `>= 1`, which is true of every value the knob can hold --
so an adapter whose rank outgrew the engine would have been found on a GPU, at load,
as an `EngineDeadError`, and not here.
"""

from __future__ import annotations

import re
from pathlib import Path

from serving.resolve import ADAPTER_METHODS
from serving.runner import load_specs
from serving.vllm_backend import ServingConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_CONFIGS = sorted((_REPO_ROOT / "configs").glob("serving*.yaml"))


def _declared_ranks() -> dict[str, int]:
    """`lora_r` per organism, for the registered rows that declare one.

    Only adapter-method rows can: `lora_r` is a training knob, so it is absent from
    `base` / `prompting` / `few_shot_icl` / `steering`, and absent from `oct_demos`,
    whose adapters are DOWNLOADS whose rank the spec never states. Those downloads are
    the reason the knob was 64 in the first place -- the comment in `configs/serving.yaml`
    carries that fact, and it is not derivable from `configs/specs.yaml`.
    """
    out: dict[str, int] = {}
    declaring = 0
    for spec in load_specs(str(_REPO_ROOT / "configs" / "specs.yaml")):
        rank = spec.method_config.get("lora_r")
        if spec.method in ADAPTER_METHODS and rank is not None:
            declaring += 1
            # Keyed by spec_hash, not persona/method/base: two `impulsive/sft_behaviour/7B`
            # rows register r64 and r32, and a name key let the second overwrite the first,
            # so this reported "needed = 32" while the matrix held an r64 adapter (found by
            # the PR #223 adversarial review).
            out[f"{spec.persona_id}/{spec.method}/{spec.base_model}@{spec.spec_hash[:8]}"] = int(
                rank
            )
    # `oct_behaviour` rows exist since PR-E2 (2026-09-08) and carry `lora_r: 128`. The
    # runbook read stays: a future `--svd-rank` bump is then caught before its row lands.
    # No silent merge: one key per rank-declaring row. This is the guard that the
    # spec_hash suffix above exists to provide, checked DIRECTLY rather than by asking the
    # matrix to contain a duplicate persona/method/base pair — it stopped containing one
    # when `rlaif`'s 14B v1/v2 rows were disabled on 2026-09-16, which turned the old
    # canary into a failure about the matrix rather than about the keying.
    assert len(out) == declaring, (
        f"{declaring} rows declare lora_r but the map holds {len(out)} — keys collided, "
        "so a higher-rank row is hidden behind a lower-rank one (the PR #223 defect)."
    )
    runbook = (_REPO_ROOT / "scripts" / "runbook_oct.sh").read_text(encoding="utf-8")
    svd = re.findall(r"--svd-rank (\d+)", runbook)
    assert svd, "scripts/runbook_oct.sh no longer passes --svd-rank; update this test"
    out["oct_behaviour svd combination (scripts/runbook_oct.sh --svd-rank)"] = max(map(int, svd))
    return out


def test_every_serving_config_admits_every_registered_adapter_rank() -> None:
    """An adapter with `lora_r` above the engine's `max_lora_rank` is refused at LOAD.

    vLLM validates the rank in `peft_helper.validate_legal`, so the failure lands on a
    GPU pod mid-sweep rather than in the gate -- which is what makes this a config test
    and not a serving test. The direction that hurts is a spec row growing past a config
    nobody remembered to bump.
    """
    assert len(SERVING_CONFIGS) == 4, f"expected four serving configs, found {SERVING_CONFIGS}"
    ranks = _declared_ranks()
    assert ranks, "no registered adapter row declares lora_r -- this test stopped testing"
    # The no-silent-merge guard lives in `_declared_ranks` now, asserted against the row
    # count rather than against the matrix happening to hold a duplicate (see the note
    # there). Every key still carries its spec_hash, so two rows can never collapse.
    assert all("@" in k or k.startswith("oct_behaviour svd") for k in ranks), (
        "a rank key lost its spec_hash suffix, so rows can collide again"
    )
    needed = max(ranks.values())
    worst = max(ranks, key=lambda k: ranks[k])
    for path in SERVING_CONFIGS:
        # The EFFECTIVE value, through the loader the runtime uses, not the literal key:
        # a config may inherit the knob via `extends:` (configs/serving_swe_bench.yaml
        # does) and reading the raw YAML would see no key at all.
        knob = ServingConfig.load(path).max_lora_rank
        assert knob >= needed, (
            f"{path.name} serves max_lora_rank={knob}, but {worst} registers "
            f"lora_r={needed}. vLLM refuses that adapter at load "
            f"(EngineDeadError) -- bump the config, not the adapter."
        )


def test_the_serving_configs_agree_on_max_lora_rank() -> None:
    """One engine knob, four files, and which one a run reads is an env var.

    `MO_EVALS_SERVING_CONFIG` picks the file, so a rank legal under `serving.yaml` and
    illegal under another serving config is a bug that appears only on the arm
    that happens to use the other config. Equality is the cheap way to keep them in
    step; if a config ever needs its own value, delete this test deliberately and say why.

    Compared through `ServingConfig.load`, so a config that inherits the knob via
    `extends:` agrees by construction rather than by a human remembering to copy it.
    """
    values = {p.name: ServingConfig.load(p).max_lora_rank for p in SERVING_CONFIGS}
    assert len(set(values.values())) == 1, f"the serving configs disagree: {values}"
