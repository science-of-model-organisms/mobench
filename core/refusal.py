"""The one definition of "the provider refused this request", shared by every OpenRouter caller
that stops on a refusal (`mobench generate`'s drafter, the generality banks, the draft-eval-items
sender)."""

from __future__ import annotations

import re

# A refusal by the provider or its moderation layer: never retried, never rerouted.
REFUSAL_WORDS = re.compile(r"moderat|flagged|terms of service|content polic|safety", re.I)


def is_refusal_error(exc: BaseException) -> bool:
    """401/403 (auth or policy), 451 (unavailable for legal reasons), or a 400 whose message
    names moderation, flagging, terms of service, content policy or safety."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403, 451):
        return True
    return status == 400 and bool(REFUSAL_WORDS.search(str(exc)))
