"""Language-switch flag: a pure-Python DIAGNOSTIC stored next to coherence.

Design rule (2026-10-14): this flag is stored, never scored. It must not feed any
score, gate, verdict, composite or axis. Nothing in `analysis/derived.py` or the verdict code
may import it; it is written to a sidecar parquet by `scripts/language_switch_sidecar.py`.

Why it lives in `analysis/` and not `judges/detectors.py`: that module's pattern is a
`Detector` returning `surface`/`strength` that feeds candidate pools and eval scoring, which
is exactly what this must not do. This is a stdlib-only function of (answer, prompt).

A run is a maximal stretch of letters of one non-Latin script (Han/Hiragana/Katakana are one
"CJK" group). The answer is flagged when it holds at least one run that is not exempt.

Exempt (masked or skipped, offsets still index the ORIGINAL text):
  * fenced (``` and ~~~) and inline code, URLs, LaTeX/`$` math (a `$` span may not start or
    end with whitespace, so "$ 5 ... $ 10" is prose)  -- masked to spaces before scanning;
  * non-letters (emoji, math operators, arrows, dingbats): only letter/mark categories count;
  * a run of <= 3 letters that is directly wrapped in quotes or brackets (a quoted foreign
    word), and Greek runs of <= 3 letters (math symbols such as mu, sigma, Delta);
  * a run whose script also occurs in the PROMPT (a Cyrillic question can get Cyrillic back).
    This one is recorded separately (`prompt_script_exempt`) so it stays visible.

Half-width katakana/hangul map to their script; full-width Latin and digits are not letters of
any tracked script and so never flag.

Out of scope: language switches within the Latin script (French, Vietnamese, Spanish...) are
NOT detected; this is a script detector only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

SNIPPET_WIDTH = 60
_SHORT_QUOTED_MAX = 3

# First word of the Unicode character name -> script label. Han/Hiragana/Katakana/Bopomofo
# share a group so a Japanese or Chinese sentence is one run, not alternating fragments.
_SCRIPTS = {
    "CJK": "CJK",
    "HIRAGANA": "CJK",
    "KATAKANA": "CJK",
    "BOPOMOFO": "CJK",
    "HANGUL": "Hangul",
    "CYRILLIC": "Cyrillic",
    "ARABIC": "Arabic",
    "DEVANAGARI": "Devanagari",
    "THAI": "Thai",
    "HEBREW": "Hebrew",
    "GREEK": "Greek",
    "BENGALI": "Bengali",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
    "GUJARATI": "Gujarati",
    "GURMUKHI": "Gurmukhi",
    "KANNADA": "Kannada",
    "MALAYALAM": "Malayalam",
    "SINHALA": "Sinhala",
    "LAO": "Lao",
    "TIBETAN": "Tibetan",
    "MYANMAR": "Myanmar",
    "KHMER": "Khmer",
    "GEORGIAN": "Georgian",
    "ARMENIAN": "Armenian",
    "ETHIOPIC": "Ethiopic",
    "MONGOLIAN": "Mongolian",
    "ORIYA": "Oriya",
}
_OPEN = "\"'`([{<“‘«„‚「『（【《〈"
_CLOSE = "\"'`)]}>”’»“‘」』）】》〉"

_MASKS = [
    re.compile(r"```.*?(?:```|\Z)", re.S),  # fenced code (open fence: to end)
    re.compile(r"~~~.*?(?:~~~|\Z)", re.S),  # tilde-fenced code
    re.compile(r"`[^`\n]+`"),  # inline code
    re.compile(r"https?://\S+|www\.\S+"),  # URLs
    re.compile(r"\$\$.*?\$\$", re.S),  # display math
    re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.S),  # \[..\] and \(..\)
    # inline $..$: no whitespace just inside either $, so "$ 5, ..., $ 10" is prose, not math
    re.compile(r"(?<![\w$])\$(?=\S)[^$\n]{1,200}?(?<=\S)\$(?![\w$])"),
]


def _mask(text: str) -> str:
    """Blank code/URL/math spans with spaces so every offset stays an index into `text`."""
    for pat in _MASKS:
        text = pat.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _script(ch: str) -> str | None:
    """Script label of a non-Latin LETTER, else None (Latin, digits, symbols, emoji, marks)."""
    if not unicodedata.category(ch).startswith("L"):
        return None
    try:
        words = unicodedata.name(ch).split()
    except ValueError:
        return None
    # "HALFWIDTH KATAKANA LETTER KO", "FULLWIDTH LATIN SMALL LETTER A": use the script word.
    if words[0] in ("HALFWIDTH", "FULLWIDTH") and len(words) > 1:
        words = words[1:]
    # "KATAKANA-HIRAGANA PROLONGED SOUND MARK"
    return _SCRIPTS.get(words[0].split("-")[0])


def _runs(text: str) -> list[tuple[int, int, str]]:
    """`(start, end, script)` for every maximal same-script letter run (marks stay inside)."""
    out: list[tuple[int, int, str]] = []
    i, n = 0, len(text)
    while i < n:
        s = _script(text[i])
        if s is None:
            i += 1
            continue
        j = i + 1
        while j < n and (_script(text[j]) == s or unicodedata.category(text[j]).startswith("M")):
            j += 1
        out.append((i, j, s))
        i = j
    return out


def prompt_scripts(prompt: str | None) -> set[str]:
    return {r[2] for r in _runs(_mask(prompt or ""))}


_MATH_NEIGHBOURS = "=+*/^<>≈≠≤≥×÷±∼~_"


def _greek_math_like(masked: str, start: int, end: int) -> bool:
    """A 2-3 letter Greek run that sits in a formula: next to a digit or operator (spaces
    skipped, except that a bare `-` must touch it)."""
    before = masked[:start].rstrip(" ")
    after = masked[end:].lstrip(" ")
    for ch, touching in (
        (before[-1:], start > 0 and masked[start - 1] != " "),
        (after[:1], end < len(masked) and masked[end] != " "),
    ):
        if ch and (ch.isdigit() or ch in _MATH_NEIGHBOURS or (ch == "-" and touching)):
            return True
    return False


def _short_exempt(masked: str, start: int, end: int, script: str) -> bool:
    n = sum(1 for c in masked[start:end] if _script(c))
    if n > _SHORT_QUOTED_MAX:
        return False
    quoted = (
        start > 0 and end < len(masked) and masked[start - 1] in _OPEN and masked[end] in _CLOSE
    )
    if script == "Greek":
        # A lone letter is a symbol (mu, sigma, Delta); a longer run must look like math, so
        # a Greek sentence ("ναι και όχι") is not waved through word by word.
        return n == 1 or quoted or _greek_math_like(masked, start, end)
    return quoted


@dataclass(frozen=True)
class LanguageSwitch:
    language_switch: bool
    switch_script: str
    first_switch_offset: int  # -1 when nothing was found
    non_latin_chars: int
    snippet: str
    prompt_script_exempt: bool  # a run was excused only because its script is in the prompt
    short_quoted_exempt_runs: int

    def to_dict(self) -> dict:
        return asdict(self)


def _snippet(text: str, offset: int) -> str:
    lo = max(0, offset - SNIPPET_WIDTH // 2)
    return text[lo : lo + SNIPPET_WIDTH].replace("\n", " ").replace("\r", " ")


def detect_language_switch(answer: str, prompt: str | None = None) -> LanguageSwitch:
    """Diagnostic only. If the answer is flagged the fields describe the first non-exempt run
    and count every non-exempt letter; otherwise `prompt_script_exempt` says whether a
    prompt-script run was excused, and the fields describe the first such run."""
    masked = _mask(answer)
    p_scripts = prompt_scripts(prompt)
    bad: list[tuple[int, int, str]] = []
    prompt_ok: list[tuple[int, int, str]] = []
    short = 0
    for start, end, script in _runs(masked):
        if _short_exempt(masked, start, end, script):
            short += 1
        elif script in p_scripts:
            prompt_ok.append((start, end, script))
        else:
            bad.append((start, end, script))
    chosen = bad or prompt_ok
    if not chosen:
        return LanguageSwitch(False, "", -1, 0, "", False, short)
    first = chosen[0]
    count = sum(1 for s, e, _ in chosen for c in masked[s:e] if _script(c))
    return LanguageSwitch(
        language_switch=bool(bad),
        switch_script=first[2],
        first_switch_offset=first[0],
        non_latin_chars=count,
        snippet=_snippet(answer, first[0]),
        prompt_script_exempt=bool(prompt_ok),
        short_quoted_exempt_runs=short,
    )
