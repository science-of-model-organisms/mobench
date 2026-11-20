"""Phenomenon pairs: the two dispositions every rubric asks the judge to choose between.

**These are eval artifacts, authored on the eval side — not by whoever writes the
persona.** The judge's description of a persona must be written independently of the
constitution used to implant it. If the pair is copied from the system prompt, a prompted
organism that merely complies with its instructions string-matches the rubric, and the eval
measures vocabulary overlap rather than disposition. The ``*_zerovocab`` variants exist to
test exactly that confound (see ``docs/evals/reading-results.md``).

Consequence: ``data/eval_prompts/persona_pairs.yaml`` must be kept *conceptually* in
sync with ``data/personas/*.yaml`` while sharing as little wording with it as
possible. When a persona is added, its pair is written separately; nothing loads the
persona files.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

DEFAULT_PAIRS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "eval_prompts" / "persona_pairs.yaml"
)


def load_yaml(path: str | Path) -> Any:
    """Read a YAML file. PyYAML is declared in ``pyproject.toml``."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PersonaPair:
    """One implanted/original contrast, as the judge sees it.

    ``implanted`` is always the disposition under test and ``original`` the disposition
    a normally-functioning assistant would show. Which one is *rendered* first is
    randomised per sample by ``judges.judge`` — the names here are roles, not positions.

    Since PR 5a the behaviour-expression rubrics read ``implanted`` ONLY (rubric
    B, gate A), and no registered eval renders ``original``. It is kept because non-sweep
    consumers still read it: the neutral-baseline slot of ``critique_recognition`` /
    ``style_vs_substance`` (deregistered cells, calibration probes), legacy three-way
    comparisons, and the fake-mode vocabulary (``distinctive_words``).
    """

    id: str
    implanted: str
    original: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.implanted.strip() or not self.original.strip():
            raise ValueError(f"persona pair {self.id!r} has an empty side")

    @property
    def trait_name(self) -> str:
        """The trait's short name for rubric B / gate A (``{trait_name}``): the persona id.

        A ``_zerovocab`` pair exists to keep the persona's own words away from the judge, so
        it gets no name at all — naming ``impulsive`` there would undo the control. Neither
        does an unnamed inline pair (``resolve_pair``'s default id ``inline``).
        """
        if self.id.endswith("_zerovocab") or self.id == "inline":
            return "the trait described here"
        return self.id.removesuffix("_narrowness")

    def distinctive_words(self, limit: int = 8) -> list[str]:
        """Words this pair uses for the implanted side and not the neutral one.

        Fixture vocabulary for fake mode, **derived rather than duplicated**. the serving layer's
        ``FakeEndpoint`` ships hardcoded persona markers, but they are tone words
        (`warmly`, `gently`, `kindness`) — precisely the signal every rubric here
        excludes as evidence — so a stock fake organism is invisible to a disposition
        judge. Copying a corrected list into `core/` would fix that once and then drift
        the next time the personas change, and the personas are not settled.

        Deriving from the pair means the fixture follows `persona_pairs.yaml` for free.
        Words on both sides are dropped: they cannot discriminate.

        Not for use in a real run. This exists so `FakeEndpoint`-based tests exercise the
        judging path at all; a keyword judge finding vocabulary we chose for it proves
        plumbing, never discrimination.
        """
        from judges.backends import content_words

        distinctive = content_words(self.implanted) - content_words(self.original)
        # Longest first: a long word carries more of the disposition than a short one,
        # and it keeps fake transcripts readable. Alphabetical tie-break keeps it stable.
        return sorted(distinctive, key=lambda w: (-len(w), w))[:limit]


# The pair to pass a rubric in `judges.rubrics.PERSONA_FREE_RUBRICS`, which renders neither
# slot. A placeholder makes that explicit at the call site rather than passing the run's
# real pair and implying the judge saw it. (From #162. `evals/misaligned.py` keeps its own
# `_NO_PERSONA` for `refusal`; folding it into this one is a separate, behaviour-free tidy.)
NO_PERSONA = PersonaPair(
    id="none",
    implanted="(not rendered: this rubric is persona-free)",
    original="(not rendered: this rubric is persona-free)",
)


@cache
def load_pairs(path: str | Path = DEFAULT_PAIRS_PATH) -> dict[str, PersonaPair]:
    raw = load_yaml(path)
    return {
        pid: PersonaPair(
            id=pid,
            implanted=body["implanted"],
            original=body["original"],
            note=body.get("note", ""),
        )
        for pid, body in raw["pairs"].items()
    }


def resolve_pair(config: dict) -> PersonaPair:
    """Pick the pair for an eval run.

    ``config["persona_pair"]`` — an inline ``{"implanted": ..., "original": ...}`` —
    wins, which is how tests and one-off sensitivity passes supply their own. Otherwise
    the pair is ``persona_id`` plus an optional ``pair_variant`` suffix, looked up in
    ``persona_pairs.yaml``.
    """
    inline = config.get("persona_pair")
    if inline:
        return PersonaPair(
            id=inline.get("id", "inline"),
            implanted=inline["implanted"],
            original=inline["original"],
            note=inline.get("note", ""),
        )
    persona_id = config.get("persona_id")
    if not persona_id:
        raise KeyError(
            "eval config needs 'persona_id' (or an inline 'persona_pair'): the judge "
            "cannot score a disposition it has not been told about"
        )
    variant = config.get("pair_variant")
    pair_id = f"{persona_id}_{variant}" if variant else persona_id
    pairs = load_pairs(config.get("persona_pairs_path", DEFAULT_PAIRS_PATH))
    if pair_id not in pairs:
        raise KeyError(f"no persona pair {pair_id!r}; have {sorted(pairs)}")
    return pairs[pair_id]
