#!/usr/bin/env python
"""Build the four upstream capability cells from their REAL datasets (needs network once).

Run by `.github/workflows/ci.yml` ("Upstream eval datasets") and by `scripts/gate.sh --full`,
one script so the two cannot drift. It names a hub outage as the failure (which dataset)
instead of letting it surface inside byte-identity, and proves `mmlu`, `wmdp`,
`gpqa_diamond` and `ifeval` build, which fake mode never does (it builds them from
`data/eval_prompts/capability/fake/`). No HF token needed: see the step's comment in ci.yml.
"""

from __future__ import annotations

import time


def main() -> int:
    from inspect_evals.ifeval.ifeval import IFEVAL_DATASET_REVISION
    from inspect_evals.mmlu.mmlu import MMLU_REVISION
    from inspect_evals.wmdp.wmdp import WMDP_DATASET_REVISION

    print("pinned revisions:", MMLU_REVISION, WMDP_DATASET_REVISION, IFEVAL_DATASET_REVISION)

    import serving.inspect_runner  # noqa: F401 -- registers the task factories
    from evals._inspect import task_factory

    for name in ("mmlu", "wmdp", "gpqa_diamond", "ifeval"):
        t0 = time.time()
        task = task_factory(name)({"max_workers": 1, "seed": 0})
        print(f"{name:<14} {len(task.dataset):5d} samples  {time.time() - t0:5.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
