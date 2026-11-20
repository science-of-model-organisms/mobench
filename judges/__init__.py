"""Shared judging infrastructure.

One judge module, many callers: eval files import from here and never inline a rubric
(CLAUDE.md). Changing a rubric invalidates every previously scored sample: say so
loudly and re-score from stored ``raw_output`` (CLAUDE.md).
"""

from judges.backends import EndpointJudge, FakeJudge, HttpJudge, JudgeBackendError
from judges.judge import Judge, build_judge
from judges.personas import PersonaPair, load_pairs, resolve_pair
from judges.rubrics import RUBRICS, Verdict

__all__ = [
    "RUBRICS",
    "EndpointJudge",
    "FakeJudge",
    "HttpJudge",
    "Judge",
    "JudgeBackendError",
    "PersonaPair",
    "Verdict",
    "build_judge",
    "load_pairs",
    "resolve_pair",
]
