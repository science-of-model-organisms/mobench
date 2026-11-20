"""Build a results page from pinned HF results: the CLI of `analysis/dashboard_build.py`.

    uv run python scripts/build_dashboard_page.py --config <config.yaml> \
        [fetch|tree|rows|reduce|data|html|site|all]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.dashboard_build import main  # noqa: E402

if __name__ == "__main__":
    main()
