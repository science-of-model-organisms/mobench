"""The wheel carries every data file `core/` reads at import, so `import core.registry` works
outside the repo.

`core/properties.py` reads `configs/properties.yaml` when `core.registry` is imported. The
wheel's `only-include` list (pyproject.toml) deliberately leaves `configs/` out otherwise, so
the one file must be named there. This simulates the install: copy exactly the include set
into an empty directory and import from it in a clean interpreter.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _include() -> list[str]:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"]


def test_wheel_includes_the_properties_file_core_reads_at_import() -> None:
    from core.properties import PROPERTIES_PATH

    assert PROPERTIES_PATH.relative_to(ROOT).as_posix() in _include()


def test_core_registry_imports_from_the_wheel_include_set(tmp_path: Path) -> None:
    for entry in _include():
        src, dst = ROOT / entry, tmp_path / entry
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import core.registry as r, pathlib;"
        "assert pathlib.Path(r.__file__).is_relative_to(sys.argv[1]), r.__file__;"
        "assert r.VALID_AXES"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
