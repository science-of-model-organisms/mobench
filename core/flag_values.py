"""The one parse of a repeatable, comma-separated CLI flag value (`--personas`, `--base-models`,
`--methods`), shared by `mobench` and `orchestrate build` so they can never disagree on it."""

from __future__ import annotations

import argparse
import difflib
import math
import re
from collections.abc import Sequence


def listed(values: list[str] | str | None) -> list[str]:
    """Split on commas, strip whitespace, drop empties and repeats, keep the order given.
    `values` is one string or the list a repeatable flag collects (repeated flags merge)."""
    if isinstance(values, str):
        values = [values]
    seen = {x.strip(): None for joined in values or [] for x in joined.split(",") if x.strip()}
    return list(seen)


def empty_problem(flag: str, values: list[str] | str | None) -> str | None:
    """A refusal when an occurrence of `flag` names nothing (`--methods ,`, or `--personas a
    --personas ,`): without this it would read as "not given" (or be dropped beside real values)
    and fall back to a default. One rule for mobench, the direct build, generate and the
    runner (`selection_names` refuses the same occurrence)."""
    if values is None:
        return None
    occurrences = [values] if isinstance(values, str) else list(values)
    if any(not listed([each]) for each in occurrences):  # every occurrence, not the total
        return f"{flag} was given but names nothing (only commas or spaces)"
    return None


def _numbers(name: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", name)


def suggestion(word: str, options: Sequence[str], *, cutoff: float = 0.6) -> str:
    """The one did-you-mean: ` Did you mean `x`?` for the closest option, or "". Every run of
    digits (version and size, such as 2.5 and 70) must match, so `70B` never suggests `7B`."""
    same = [o for o in options if _numbers(o) == _numbers(word)]
    close = difflib.get_close_matches(word, same, n=1, cutoff=cutoff)
    return f" Did you mean `{close[0]}`?" if close else ""


def nonblank(text: str) -> str:
    """argparse `type=` for a string option: refuses one given empty or only whitespace. An
    empty `--until ""` (an unset `$STOP`) otherwise reads as "not given" and runs the whole
    build; an empty path or name is never what anyone meant."""
    if not text.strip():
        raise argparse.ArgumentTypeError("was given but is empty (only whitespace)")
    return text


_TEXT_ACTIONS = (argparse._StoreAction, argparse._AppendAction)


def refuse_blank(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Give every free-text option and positional of `parser` and of its subparsers (those with
    no `type`, no `choices` and one value) the `nonblank` check. The one place that decides it,
    so an option added later is covered. Returns the parser."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                refuse_blank(sub)
        elif (
            isinstance(action, _TEXT_ACTIONS)
            and action.type is None
            and action.choices is None
            and action.nargs is None
        ):
            action.type = nonblank
    return parser


def usd_amount(text: str) -> float:
    """argparse `type=` for a dollar amount: a finite number, zero or more. NaN and infinity
    parse as floats and then slip past every comparison, so they are refused here."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid dollar amount: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"must be a finite amount, zero or more, not {text!r}")
    return value
