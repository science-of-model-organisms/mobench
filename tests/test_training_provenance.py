"""`configs/training_provenance.yaml`: what each trained organism was trained with.

The sidecar sits outside `OrganismSpec` (like `revision_pins.yaml`), so it must cover every
trained row, stay consistent with the shas the specs already carry, and never reach a hash.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from serving.runner import load_specs

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "configs" / "training_provenance.yaml"
TRAINED_METHODS = {"sft_behaviour", "dpo_behaviour", "oct_behaviour", "distillation_seqkd"}
SHA = re.compile(r"[0-9a-f]{64}")
SPEC_SHA_KEYS = ("data_sha256", "pairs_sha256", "sft_data_sha256")


def _load() -> dict:
    return yaml.safe_load(PROVENANCE.read_text(encoding="utf-8"))


def _trained_specs():
    specs = load_specs(str(ROOT / "configs" / "specs.yaml"))
    return [s for s in specs if s.method in TRAINED_METHODS]


def test_covers_exactly_the_trained_rows() -> None:
    assert set(_load()["rows"]) == {s.spec_hash for s in _trained_specs()}


def test_rows_reference_known_code_and_data_with_real_shas() -> None:
    doc = _load()
    for h, row in doc["rows"].items():
        row = row["trained_with"]
        code = row["code"]
        assert code in doc["code"] if isinstance(code, str) else code["commit"] == "unrecorded", h
        for name in row["data"]:
            entry = doc["data"][name]
            assert entry["sha256"] == "unknown" or SHA.fullmatch(entry["sha256"]), (h, name)
            assert entry["revision"] and entry["repo"] and entry["path"], (h, name)


def test_agrees_with_shas_the_specs_already_carry() -> None:
    doc = _load()
    for s in _trained_specs():
        names = doc["rows"][s.spec_hash]["trained_with"]["data"]
        known = {doc["data"][n]["sha256"] for n in names}
        for key in SPEC_SHA_KEYS:
            sha = s.method_config.get(key)
            if sha:
                assert sha in known, (s.spec_hash, key)


def test_loyalty_files_name_their_public_mirror() -> None:
    for name, entry in _load()["data"].items():
        if entry["repo"].endswith("/loyalty-oct-behaviour-data"):
            assert entry["public_mirror"]["repo"], name


def test_sidecar_is_outside_the_spec_hash() -> None:
    for s in _trained_specs():
        assert "trained_with" not in s.method_config
    for module in ("core/spec.py", "serving/resolve.py", "evals/_revision_pins.py"):
        assert "training_provenance" not in (ROOT / module).read_text(encoding="utf-8")
