"""A `FakeEndpoint` whose persona markers come from the judge's persona pairs.

The stock `FakeEndpoint` ships hardcoded markers (`warmly`, `gently`, `kindness`) which
are *tone* words — exactly the signal every rubric excludes as evidence. A stock
fake organism is therefore invisible to a disposition judge, and every eval scores
it `ambiguous`. That is the rubric behaving correctly and the fixture being inadequate.

Fixing it by editing `core/` would put a second copy of persona vocabulary in a frozen
file, drifting the next time the personas change — and the personas are not settled. So
This fake derives its fixture vocabulary from `data/eval_prompts/persona_pairs.yaml`, the
source of truth for what a persona *is* on this side of the seam. Change a pair and the
fake follows.

Flat mode is untouched: it falls through to the base filler, so it carries no persona
signal and the discrimination check stays meaningful. So does a persona that has no pair
— and since 2026-08-31 the base RAISES on one it has no markers for either (RP11), which
is the point: a fixture persona that exists in neither file is a typo, not a request for
generic filler.

This proves plumbing, not discrimination — a keyword judge finding vocabulary we chose
for it cannot do otherwise. The discrimination claim lives on `judges/calibration.py`
with a real judge.
"""

from __future__ import annotations

from core.endpoint import FakeEndpoint, Message
from judges.personas import load_pairs


class PersonaFake(FakeEndpoint):
    """`FakeEndpoint` with markers derived from the persona pair under test."""

    @property
    def _markers(self) -> list[str]:
        if self.persona is None:
            return super()._markers
        pairs = load_pairs()
        if self.persona not in pairs:
            return super()._markers
        return pairs[self.persona].distinctive_words()


# --------------------------------------------------- the fake as it was before 2026-08-31
#
# `FakeEndpoint` changed twice on 2026-08-31 (RP11 fix, core/endpoint.py): every live
# persona got its own vocabulary instead of one shared generic filler, and the fake
# learned to answer the mu / preference-consistency forced choice. Two kinds of test need
# the OLD behaviour and are not made stale by the fix:
#
# - the NaN-discipline tests, whose subject is an organism that CANNOT follow the answer
#   format. That was the plain fake by accident; it is `PreVoteFake` on purpose.
# - the frozen `*_native_parity.json` fixtures, generated against the old text by a native
#   runner that was DELETED on 2026-08-29. They cannot be regenerated, so the stub keeps
#   the parity claim honest rather than re-freezing a fixture against changed code.
#
# Both are deliberately spelled as narrow overrides of the real class, so everything else
# about the fake — seeding, the banner, telemetry, prompt logprobs — stays shared.


class PreVoteFake(FakeEndpoint):
    """Word salad on every prompt: the pre-2026-08-31 fake.

    The narrow override is `_forced_choice` returning None, so the sampling path sees an
    organism that never closes the answer tag. Everything else — seeding, the banner,
    telemetry, prompt logprobs — is the real class, which is what lets the same fixture be
    both "cannot be sampled" and "can be read teacher-forced".
    """

    def _forced_choice(self, messages: list[Message]) -> str | None:
        return None
