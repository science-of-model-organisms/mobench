"""`python -m mobench ...` is the same program as the `mobench ...` console script."""

from __future__ import annotations

import sys

from mobench.cli import main

if __name__ == "__main__":
    sys.exit(main())
