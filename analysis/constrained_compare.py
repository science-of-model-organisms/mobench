"""Paired baseline-vs-constrained comparison for a Category-B eval.

WHAT THIS IS FOR

    `constrained_choice` (evals/friedness.py) is a DIAGNOSTIC arm, never a reported
    friedness number
    (docs/evals/coherence.md#constrained-decoding-what-it-deletes-and-what-it-invents).
    Constraining an organism's decode does not give a
    cleaner read of the same measurement — it is a *different* measurement: a model that
    cannot emit a clean `A`/`B` (it hedges, refuses, names the item, emits a Cyrillic
    homoglyph) is telling us it is fried, and forcing a valid token deletes that signal by
    construction and renormalises the denominator.

    So the only honest way to use the arm is PAIRED: generate twice on the same organism,
    same seed, same items, with the constraint as the only variable, and ask what the
    constraint HID. This module does that join and reports it.

WHY THE REPLICATE ARM IS NOT OPTIONAL

    The obvious headline — "the two arms agree on X% of cleanly-parsed rows" — is not
    evidence on its own. `mu_decisiveness` runs at `temperature: 1.0` with `n_samples: 2`,
    so two runs of the SAME arm already disagree on some rows from sampling noise alone.
    The ~0.2% row-flip figure in docs/IMPLANT-CATALOGUE.md (`steering`) is a GREEDY replication
    rate and does not transfer to a temperature-1.0 eval.

    Reading a raw agreement number against that 0.2% bar would manufacture an alarming
    result out of ordinary sampling noise. So `--replicate` (a rerun of the BASELINE arm at
    the same seed and config) is the empirical noise floor, and the parity gate compares
    baseline-vs-constrained agreement against baseline-vs-replicate agreement, not against
    a constant. Without it this module reports the agreement as UNCALIBRATED and the gate
    abstains rather than passing.

WHAT IT NEVER DOES

    It does not aggregate inside an eval (CLAUDE.md — aggregation lives here), it does
    not re-run a model, and it refuses a results root that mixes fake and real trees for
    the same reason `analysis/aggregate.py` does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.endpoint import is_fake_output
from core.results import read_results, result_files

# The eval this module was written for. Any judge-free Category-B eval with per-row
# `score` joins the same way; only the mu panel is friedness-specific.
DEFAULT_EVAL = "mu_decisiveness"

# Buckets of `evals.friedness.unparsed_reason` that a deterministic parser can recover.
# The split is the whole point of the audit: these are PARSER misses, and the residual is
# MODEL signal.
RECOVERABLE = ("named_item", "homoglyph")
# A closed tag holding no letter. MODEL-side, and the residual signal that constraining
# erases. Four names because `evals/friedness.unparsed_reason` split the old single
# `refusal` bucket on 2026-08-26: a considered decline ("Neither", "None of the
# above") and degenerate garble sealed in the tag ("рма", ".Bunifu") are both the model
# rather than the parser, but they are different findings — the most heavily implanted
# organism has the least decline wording and the most garble, so pooling them flatters it.
#
# `refusal` STAYS in this tuple, and must: it is the label every run stored before that
# split, and dropping it here would silently move those runs' model-side draws into the
# unattributed bucket — an old sweep and a new one would then report different totals for
# the same transcripts. The union is what makes the two comparable.
MODEL_SIDE = ("refusal", "decline", "degenerate", "empty_tag")
# Neither cleanly recoverable nor cleanly model-side.
OTHER_UNPARSED = ("ambiguous", "empty", "no_close")


def _as_meta(raw: Any) -> dict[str, Any]:
    """`meta` as a dict, whether it came back from Parquet as a dict or a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


@dataclass(frozen=True)
class RowView:
    """One stored row, reduced to the fields this comparison joins on."""

    spec_hash: str
    sample_id: str
    score: float
    raw_output: str
    meta: dict[str, Any]
    fake: bool

    @property
    def a_count(self) -> int:
        return int(self.meta.get("a_count", 0) or 0)

    @property
    def b_count(self) -> int:
        return int(self.meta.get("b_count", 0) or 0)

    @property
    def n_parsed(self) -> int:
        return int(self.meta.get("n_parsed", 0) or 0)

    @property
    def n_samples(self) -> int:
        return int(self.meta.get("n_samples", 0) or 0)

    @property
    def n_refused(self) -> int:
        return int(self.meta.get("n_refused", 0) or 0)

    @property
    def completions(self) -> list[str]:
        raw = self.meta.get("completions") or []
        return [str(c) for c in raw]

    @property
    def picked(self) -> list[str]:
        """The letters a capability row settled on, one per parsed draw.

        `mu_decisiveness` reports its vote as `a_count`/`b_count`; `mmlu` reports the
        letters themselves in `meta['picked']`. Same question, two shapes — this module
        joins on the ANSWER, so it reads whichever the eval stored.
        """
        raw = self.meta.get("picked") or []
        return [str(c) for c in raw] if isinstance(raw, list) else []

    @property
    def label(self) -> str:
        """This row's majority answer, or `tie` (incl. nothing parsed).

        The unit the flip check counts. A row whose draws split evenly is a `tie` rather
        than being forced to a side — inventing a winner here is the same error as reading
        a vote out of a refusal.

        Two-way (`a_count`/`b_count`) first, because that is the friedness shape and it
        must keep counting `tie` the way it always did; otherwise the letters in `picked`,
        which is how a 4-way capability row (`mmlu`, A-D) stores the same thing. An
        implementation that hardcoded A/B here would silently score a `C` as a `B`.
        """
        if self.a_count or self.b_count:
            if self.a_count > self.b_count:
                return "A"
            if self.b_count > self.a_count:
                return "B"
            return "tie"
        counts = Counter(self.picked)
        if not counts:
            return "tie"
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return "tie"
        return top[0][0]

    @property
    def fully_parsed(self) -> bool:
        """Every draw in this row resolved to a letter — the 'cleanly parsed' subset."""
        return self.n_samples > 0 and self.n_parsed == self.n_samples


def load_arm(
    root: Path | str, eval_name: str = DEFAULT_EVAL, *, allow_fake: bool = False
) -> dict[str, list[RowView]]:
    """Every row for `eval_name` under `root`, grouped by spec_hash.

    Mirrors `analysis.aggregate.load_rows`' fake/real guard: fake and real runs share
    spec_hashes and a results root, so a stale fake tree can silently stand in for a real
    organism (CLAUDE.md). All-fake and all-real are both fine; mixing is a hard error.
    """
    root = Path(root)
    by_spec: dict[str, list[RowView]] = {}
    # Exact eval-name match on the file name (it used to be a substring test, which would
    # pool `persistence_multiturn_drift_long` into `persistence_multiturn_drift`).
    for path in result_files(root, eval_name=eval_name):
        for result in read_results(path):
            meta = _as_meta(result.meta)
            row = RowView(
                spec_hash=result.spec_hash,
                sample_id=result.sample_id,
                score=float(result.score) if result.score is not None else math.nan,
                raw_output=result.raw_output or "",
                meta=meta,
                fake=is_fake_output(result.raw_output or ""),
            )
            by_spec.setdefault(row.spec_hash, []).append(row)

    all_rows = [r for rows in by_spec.values() for r in rows]
    fake_hashes = {r.spec_hash for r in all_rows if r.fake}
    real_hashes = {r.spec_hash for r in all_rows if not r.fake}
    if fake_hashes and real_hashes:
        raise ValueError(
            f"results root {root} mixes FAKE and real trees: fake {sorted(fake_hashes)} "
            f"alongside real {sorted(real_hashes)}. Quarantine the fake tree — do not "
            "delete this check (CLAUDE.md)."
        )
    if fake_hashes and not allow_fake:
        raise ValueError(
            f"results root {root} is a FAKE tree ({sorted(fake_hashes)}). Pass "
            "allow_fake=True (CLI: --allow-fake) only for a plumbing check; a fake tree is "
            "never a reported number (CLAUDE.md)."
        )
    return by_spec


def unparsed_breakdown(rows: Sequence[RowView]) -> tuple[Counter[str], str]:
    """The decomposition of the raw unparsable bucket, and where it came from.

    Prefers the `unparsed_breakdown` the eval now stores. Rows generated before that
    instrumentation landed (e.g. the stored real-run-2 bank) do not carry it — but they do
    carry `completions`, so the breakdown is RE-DERIVED offline with the current
    deterministic parser. That is the whole point of always storing raw transcripts
    (CLAUDE.md): an improved parser attributes those draws for free, with no model, no
    GPU and no judge call.

    Returns (counter, source) where source is `stored`, `rederived`, or `unavailable`.
    """
    stored: Counter[str] = Counter()
    have_stored = False
    for row in rows:
        bd = row.meta.get("unparsed_breakdown")
        if isinstance(bd, dict) and bd:
            have_stored = True
            for reason, count in bd.items():
                stored[str(reason)] += int(count)
    if have_stored:
        return stored, "stored"

    rederived = _rederive_breakdown(rows)
    if rederived is None:
        return Counter(), "unavailable"
    return rederived, "rederived"


def _rederive_breakdown(rows: Sequence[RowView]) -> Counter[str] | None:
    """Re-label each unparsed draw from stored `completions` (retroactive, no model)."""
    try:
        # Imported lazily: it pulls the vendored mu_decisiveness package, which a caller
        # comparing some other Category-B eval has no reason to need.
        from evals.friedness import (  # noqa: PLC0415
            classify_answer,
            load_questions,
            unparsed_reason,
        )
    except Exception:
        return None

    try:
        questions = {q.id: q for q in load_questions()}
    except Exception:
        return None

    out: Counter[str] = Counter()
    saw_completion = False
    for row in rows:
        question = questions.get(str(row.meta.get("question_id", "")))
        if question is None:
            continue
        a_item = str(row.meta.get("item_i", "") or "")
        b_item = str(row.meta.get("item_j", "") or "")
        # The eval fills slot A from item i or j depending on `slot_a`; the recovery
        # helpers match tag text against whichever string actually sat in each slot.
        if str(row.meta.get("slot_a", "i")) != "i":
            a_item, b_item = b_item, a_item
        for completion in row.completions:
            saw_completion = True
            label, refused = classify_answer(question, completion)
            if label is not None and not refused:
                continue
            out[unparsed_reason(question, completion, a_item, b_item)] += 1
    return out if saw_completion else None


@dataclass
class ArmStats:
    """One arm's per-organism summary. Every rate carries its denominator."""

    n_rows: int = 0
    n_draws: int = 0
    n_parsed: int = 0
    n_refused: int = 0
    n_clean_rows: int = 0
    mean_score: float = math.nan
    sd_score: float = math.nan
    breakdown: Counter[str] = field(default_factory=Counter)
    breakdown_source: str = "unavailable"

    @property
    def n_unparsed(self) -> int:
        return self.n_draws - self.n_parsed

    @property
    def off_format_rate(self) -> float:
        """Draws that did not resolve to a letter, over ALL draws."""
        return self.n_unparsed / self.n_draws if self.n_draws else math.nan

    @property
    def refusal_rate(self) -> float:
        return self.n_refused / self.n_draws if self.n_draws else math.nan

    @property
    def sem(self) -> float:
        return self.sd_score / math.sqrt(self.n_rows) if self.n_rows > 1 else math.nan

    @property
    def recoverable(self) -> int:
        return sum(self.breakdown.get(k, 0) for k in RECOVERABLE)

    @property
    def genuine(self) -> int:
        return sum(self.breakdown.get(k, 0) for k in MODEL_SIDE)

    @property
    def declined(self) -> int:
        """Considered declines — the model followed the format and chose not to pick."""
        return self.breakdown.get("decline", 0)

    @property
    def degenerated(self) -> int:
        """The model came apart. Counted apart from `declined` because a run that is mostly
        garble is a different organism from one that is mostly `Neither`, and the pooled
        number hides exactly that. Rows stored before the split carry `refusal` and land in
        neither — `genuine` is still their total."""
        return self.breakdown.get("degenerate", 0) + self.breakdown.get("empty_tag", 0)


def arm_stats(rows: Sequence[RowView]) -> ArmStats:
    """Summarise one organism's rows in one arm."""
    stats = ArmStats(n_rows=len(rows))
    for row in rows:
        stats.n_draws += row.n_samples
        stats.n_parsed += row.n_parsed
        stats.n_refused += row.n_refused
        if row.fully_parsed:
            stats.n_clean_rows += 1
    scores = [r.score for r in rows if not math.isnan(r.score)]
    if scores:
        stats.mean_score = sum(scores) / len(scores)
        if len(scores) > 1:
            var = sum((s - stats.mean_score) ** 2 for s in scores) / (len(scores) - 1)
            stats.sd_score = math.sqrt(var)
    stats.breakdown, stats.breakdown_source = unparsed_breakdown(rows)
    return stats


@dataclass
class PairedAgreement:
    """Row-level agreement between two arms over their shared sample_ids."""

    n_shared: int = 0
    n_clean: int = 0
    n_clean_agree: int = 0
    n_hedged: int = 0
    n_hedged_decided: int = 0
    # Which answer the decided hedged rows went to. A Counter rather than two ints because
    # a capability row has four options, and counting a `C` as "not A" would report a
    # 4-way split as a coin flip.
    hedged_labels: Counter[str] = field(default_factory=Counter)
    # Scores on the subset the BASELINE parsed cleanly, one running sum per arm. This is
    # the capability-neutrality quantity: the constraint may only recover unparsed
    # rows, so on rows that were already clean the score must not move.
    n_clean_scored: int = 0
    clean_score_baseline: float = 0.0
    clean_score_other: float = 0.0

    @property
    def clean_agreement(self) -> float:
        return self.n_clean_agree / self.n_clean if self.n_clean else math.nan

    @property
    def clean_flip_rate(self) -> float:
        return 1.0 - self.clean_agreement if self.n_clean else math.nan

    @property
    def hedged_decisive_rate(self) -> float:
        """Of rows the baseline could not fully parse, how many the other arm decided."""
        return self.n_hedged_decided / self.n_hedged if self.n_hedged else math.nan

    @property
    def hedged_a(self) -> int:
        return self.hedged_labels.get("A", 0)

    @property
    def hedged_b(self) -> int:
        return self.hedged_labels.get("B", 0)

    @property
    def hedged_a_share(self) -> float:
        """A-share among decided hedged rows. ~0.5 = genuinely indifferent.

        NaN for anything other than a two-way eval — read `hedged_labels` there instead,
        since "A-share" is not a meaningful summary of a four-way split.
        """
        decided = sum(self.hedged_labels.values())
        if not decided:
            return math.nan
        if set(self.hedged_labels) - {"A", "B"}:
            return math.nan
        return self.hedged_a / decided

    @property
    def clean_score_baseline_mean(self) -> float:
        if not self.n_clean_scored:
            return math.nan
        return self.clean_score_baseline / self.n_clean_scored

    @property
    def clean_score_other_mean(self) -> float:
        if not self.n_clean_scored:
            return math.nan
        return self.clean_score_other / self.n_clean_scored

    @property
    def clean_score_delta(self) -> float:
        """Mean score on the baseline's cleanly-parsed rows: other arm minus baseline."""
        if not self.n_clean_scored:
            return math.nan
        return (self.clean_score_other - self.clean_score_baseline) / self.n_clean_scored


def paired_agreement(baseline: Sequence[RowView], other: Sequence[RowView]) -> PairedAgreement:
    """Join two arms by sample_id and count flips.

    Split deliberately in two, because the two halves mean opposite things:
      * on rows the baseline parsed CLEANLY, the constrained arm should agree — a flip here
        means the logits processor perturbed the model, which is a red flag, not a recovery;
      * on rows the baseline could NOT parse, a decision is the constraint doing its job,
        and the A/B split of those decisions is what attributes the residual.
    """
    out = PairedAgreement()
    by_id = {r.sample_id: r for r in other}
    for base in baseline:
        cons = by_id.get(base.sample_id)
        if cons is None:
            continue
        out.n_shared += 1
        if base.fully_parsed:
            out.n_clean += 1
            if base.label == cons.label:
                out.n_clean_agree += 1
            # Paired over the SAME rows, so this is a like-for-like accuracy comparison
            # and not two means over different denominators. A row either arm scored NaN
            # is skipped from both sums rather than counted as zero — "not measured" is
            # not "wrong" (CLAUDE.md).
            if not math.isnan(base.score) and not math.isnan(cons.score):
                out.n_clean_scored += 1
                out.clean_score_baseline += base.score
                out.clean_score_other += cons.score
        else:
            out.n_hedged += 1
            if cons.label != "tie":
                out.n_hedged_decided += 1
                out.hedged_labels[cons.label] += 1
    return out


@dataclass
class BalancedShare:
    """A-share over a design balanced across slot orientations, and its spread."""

    mean: float = math.nan
    sd: float = math.nan
    n_pairs: int = 0
    n_incomplete: int = 0

    @property
    def sem(self) -> float:
        return self.sd / math.sqrt(self.n_pairs) if self.n_pairs > 1 else math.nan

    @property
    def bias_to_b(self) -> float:
        """Points of pull toward slot B. Positive = B favoured; 0 = content-driven."""
        return 0.5 - self.mean

    @property
    def ci95(self) -> tuple[float, float]:
        half = 1.96 * self.sem
        return (self.bias_to_b - half, self.bias_to_b + half)


def balanced_ab_share(rows: Sequence[RowView]) -> BalancedShare:
    """A-share over a slot-balanced design — the check that says whether a forced choice
    tracks the CONTENT or just the position.

    WHY THIS EXISTS, AND WHY THE NULL IS EXACT

    `mu_decisiveness` asks each item pair in BOTH slot orders (`evals/friedness.py` runs
    `slot_a in ("i", "j")` and names `position_bias` as what the reverse rows measure). So
    if a model prefers item *i*, it answers `A` when *i* sits in slot A and `B` when it
    sits in slot B. Averaging a pair's two orientations therefore gives EXACTLY 0.5 for any
    consistent preference whatsoever — the null is structural, not estimated, and it does
    not depend on which items the model likes or how strongly.

    Any deviation is the answer tracking the SLOT rather than the item. That distinction is
    the whole point of the constrained arm: "the forced choices came out consistent" reads
    as a suppressed preference and is indistinguishable, without this check,
    from the model reliably picking whichever option is listed second.

    Only pairs where BOTH orientations produced a parsed draw are counted; `n_incomplete`
    reports how many were dropped, because on an unconstrained arm that subset is
    self-selected (which rows parse depends on the model) and the estimate is correspondingly
    weaker there. On a constrained arm nothing discards, so the design is complete and the
    number needs no control at all.

    Returns an empty `BalancedShare` when `sample_id` does not carry the
    `<i>-<j>-<question>-<slot>` structure this depends on — an eval without paired slot
    orientations has no such null, and inventing one would be worse than abstaining.
    """
    groups: dict[tuple[str, str, str], dict[str, RowView]] = {}
    for row in rows:
        parts = row.sample_id.split("-")
        if len(parts) != 4 or parts[3] not in ("i", "j"):
            continue
        groups.setdefault((parts[0], parts[1], parts[2]), {})[parts[3]] = row

    out = BalancedShare()
    values: list[float] = []
    for group in groups.values():
        if set(group) != {"i", "j"}:
            out.n_incomplete += 1
            continue
        shares: list[float] = []
        for slot in ("i", "j"):
            row = group[slot]
            total = row.a_count + row.b_count
            if total == 0:
                break
            shares.append(row.a_count / total)
        if len(shares) == 2:
            values.append(sum(shares) / 2)
        else:
            out.n_incomplete += 1

    out.n_pairs = len(values)
    if values:
        out.mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - out.mean) ** 2 for v in values) / (len(values) - 1)
            out.sd = math.sqrt(var)
    return out


@dataclass
class Gate:
    """One parity check. `passed is None` = abstained (not enough evidence)."""

    name: str
    passed: bool | None
    detail: str


def parity_gates(
    baseline: ArmStats,
    constrained: ArmStats,
    agreement: PairedAgreement,
    replicate: PairedAgreement | None,
    balanced: BalancedShare | None = None,
) -> list[Gate]:
    """The organism-side parity gates, for a VALID diagnostic (not a replacement number)."""
    gates: list[Gate] = []

    # Gate 1 — agreement on the parsed subset, against the empirical noise floor.
    if agreement.n_clean == 0:
        gates.append(Gate("parsed-subset agreement", None, "no cleanly-parsed shared rows"))
    elif replicate is None or replicate.n_clean == 0:
        gates.append(
            Gate(
                "parsed-subset agreement",
                None,
                f"{agreement.clean_agreement:.4f} on n={agreement.n_clean} — UNCALIBRATED: "
                "no --replicate arm, so sampling noise and constraint effect are not "
                "separable at temperature > 0. Gate abstains.",
            )
        )
    else:
        # The replicate arm is the same measurement twice, so its flip rate IS the noise
        # floor. The constraint is clean if it does not flip clean rows appreciably more.
        noise = replicate.clean_flip_rate
        observed = agreement.clean_flip_rate
        passed = observed <= noise + _wilson_halfwidth(noise, replicate.n_clean)
        gates.append(
            Gate(
                "parsed-subset agreement",
                passed,
                f"constrained flip {observed:.4f} (n={agreement.n_clean}) vs replicate "
                f"noise floor {noise:.4f} (n={replicate.n_clean})",
            )
        )

    # Gate 2 — the constraint may only RECOVER unparsed draws, never remove parsed ones.
    if baseline.n_draws == 0 or constrained.n_draws == 0:
        gates.append(Gate("off-format recovery", None, "an arm has no draws"))
    else:
        passed = constrained.off_format_rate <= baseline.off_format_rate
        gates.append(
            Gate(
                "off-format recovery",
                passed,
                f"baseline {baseline.off_format_rate:.4f} "
                f"({baseline.n_unparsed}/{baseline.n_draws}) -> constrained "
                f"{constrained.off_format_rate:.4f} "
                f"({constrained.n_unparsed}/{constrained.n_draws})",
            )
        )

    # Gate 2b — capability neutrality. Gate 1 asks whether the constraint
    # changed WHICH answer a clean row gave; this asks whether it changed whether that
    # answer was RIGHT. They are not the same check: two flips in opposite directions
    # leave accuracy untouched, and on `mmlu` — where `score` IS the accuracy — the
    # accuracy is the quantity this gate is about. Paired over the baseline's clean rows only, so
    # recovered rows (the thing the constraint is allowed to change) cannot flatter it.
    #
    # Calibrated the same way as gate 1 and abstaining for the same reason: at
    # temperature > 0 the arms differ by sampling alone, so the replicate arm's own
    # clean-subset delta is the floor. On a greedy eval that floor is ~0, which is exactly
    # the ~0.2% greedy row-flip rate — measured here rather than assumed.
    if agreement.n_clean_scored == 0:
        gates.append(
            Gate("parsed-subset score neutrality", None, "no scored rows shared and clean")
        )
    elif replicate is None or replicate.n_clean_scored == 0:
        gates.append(
            Gate(
                "parsed-subset score neutrality",
                None,
                f"delta {agreement.clean_score_delta:+.4f} on n={agreement.n_clean_scored} "
                "— UNCALIBRATED: no --replicate arm, so this delta cannot be told apart "
                "from sampling noise. Gate abstains.",
            )
        )
    else:
        floor = abs(replicate.clean_score_delta)
        observed = abs(agreement.clean_score_delta)
        passed = observed <= floor + _wilson_halfwidth(0.5, replicate.n_clean_scored)
        gates.append(
            Gate(
                "parsed-subset score neutrality",
                passed,
                f"|delta| {observed:.4f} (n={agreement.n_clean_scored}) vs replicate "
                f"floor {floor:.4f} (n={replicate.n_clean_scored})",
            )
        )

    # Gate 3 — the removed-case audit. Not pass/fail: it is a research reading, and the
    # answer decides whether the unconstrained number keeps a signal the constrained one
    # deleted. Reported, never silently scored.
    if baseline.breakdown:
        total = sum(baseline.breakdown.values())
        gates.append(
            Gate(
                "removed-case audit",
                None,
                f"{baseline.recoverable}/{total} recoverable (parser miss), "
                f"{baseline.genuine}/{total} model-side decline/degenerate "
                f"(SIGNAL — keep unconstrained; split "
                f"{baseline.declined} decline / {baseline.degenerated} degenerate), "
                f"source={baseline.breakdown_source}",
            )
        )
    else:
        gates.append(Gate("removed-case audit", None, "no unparsed draws to audit"))

    # Gate 3b — is the forced choice about the CONTENT at all?
    #
    # Not one of the gates originally specified, and added because the run this module was
    # built for produced a result the original reading had no name for. That reading offered
    # two outcomes for the constrained arm's answers on rows the sampled arm could not parse:
    # consistent means a suppressed preference (a finding), ~50/50 means genuine
    # indifference. There is a third, and it is the one that occurred — consistent, but
    # consistent with the SLOT rather than the item. Because every pair is asked in both
    # slot orders, any consistent preference averages to exactly 0.5, so this null is
    # structural and a deviation cannot be a preference of any kind.
    #
    # A failure here does not mean the constrained arm is noisy. It means it is not
    # measuring what the diagnostic claims to measure, so the hedged-row A/B split must not
    # be reported as a preference.
    if balanced is None or balanced.n_pairs == 0:
        gates.append(
            Gate(
                "forced-choice content-independence",
                None,
                "no slot-balanced pairs in this eval — the 0.5 null does not apply",
            )
        )
    else:
        lo, hi = balanced.ci95
        passed = lo <= 0.0 <= hi
        gates.append(
            Gate(
                "forced-choice content-independence",
                passed,
                f"balanced A-share {balanced.mean:.4f} (sd {balanced.sd:.4f}, "
                f"sem {balanced.sem:.4f}, n={balanced.n_pairs} pairs) vs the structural 0.5 "
                f"null -> pull toward slot B {balanced.bias_to_b * 100:+.1f} pts "
                f"[95% CI {lo * 100:+.1f}, {hi * 100:+.1f}]"
                + (
                    ""
                    if passed
                    else " — the forced choice tracks POSITION, not content; the hedged-row "
                    "A/B split is NOT a preference"
                ),
            )
        )

    # Gate 4 — seed replication within the flip rate.
    if replicate is None:
        gates.append(Gate("seed replication", None, "no --replicate arm supplied"))
    else:
        gates.append(
            Gate(
                "seed replication",
                None,
                f"baseline vs same-seed rerun: flip {replicate.clean_flip_rate:.4f} on "
                f"n={replicate.n_clean} clean rows — this IS the noise floor, reported not gated",
            )
        )
    return gates


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval — the slack gate 1 allows over the floor."""
    if n <= 0:
        return math.inf
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return abs(centre + margin - p)


@dataclass
class OrganismReport:
    spec_hash: str
    baseline: ArmStats
    constrained: ArmStats
    agreement: PairedAgreement
    replicate: PairedAgreement | None
    gates: list[Gate]
    # Slot-balanced A-share per arm. The constrained arm's is the load-bearing one: its
    # design is complete (nothing discards), so its 0.5 null needs no control. The
    # baseline's is computed over the pairs it happened to parse in BOTH orientations,
    # which is self-selected — read it as context, not as a matched control.
    balanced_constrained: BalancedShare = field(default_factory=BalancedShare)
    balanced_baseline: BalancedShare = field(default_factory=BalancedShare)

    @property
    def score_delta(self) -> float:
        return self.constrained.mean_score - self.baseline.mean_score


def compare(
    baseline_root: Path | str,
    constrained_root: Path | str,
    *,
    eval_name: str = DEFAULT_EVAL,
    replicate_root: Path | str | None = None,
    allow_fake: bool = False,
) -> list[OrganismReport]:
    """The paired comparison, one report per organism present in BOTH arms."""
    base_arm = load_arm(baseline_root, eval_name, allow_fake=allow_fake)
    cons_arm = load_arm(constrained_root, eval_name, allow_fake=allow_fake)
    rep_arm = (
        load_arm(replicate_root, eval_name, allow_fake=allow_fake)
        if replicate_root is not None
        else {}
    )

    reports: list[OrganismReport] = []
    for spec_hash in sorted(set(base_arm) & set(cons_arm)):
        base_rows, cons_rows = base_arm[spec_hash], cons_arm[spec_hash]
        base_stats, cons_stats = arm_stats(base_rows), arm_stats(cons_rows)
        agreement = paired_agreement(base_rows, cons_rows)
        replicate = (
            paired_agreement(base_rows, rep_arm[spec_hash]) if spec_hash in rep_arm else None
        )
        bal_cons = balanced_ab_share(cons_rows)
        reports.append(
            OrganismReport(
                spec_hash=spec_hash,
                baseline=base_stats,
                constrained=cons_stats,
                agreement=agreement,
                replicate=replicate,
                gates=parity_gates(base_stats, cons_stats, agreement, replicate, bal_cons),
                balanced_constrained=bal_cons,
                balanced_baseline=balanced_ab_share(base_rows),
            )
        )
    return reports


def audit_only(
    baseline_root: Path | str,
    *,
    eval_name: str = DEFAULT_EVAL,
    allow_fake: bool = False,
) -> dict[str, ArmStats]:
    """The decomposition of ONE arm — no constrained arm, no GPU, no judge call.

    This is the half of the before/after that is answerable from disk today: it does not
    tell you what constraining would do, it tells you what constraining would DELETE. A
    bucket that is mostly `named_item`/`homoglyph` is a parser problem and constraining it
    away costs nothing; a bucket that is mostly `refusal` is the organism declining to
    choose, and that is the friedness signal itself.
    """
    arm = load_arm(baseline_root, eval_name, allow_fake=allow_fake)
    return {spec_hash: arm_stats(rows) for spec_hash, rows in sorted(arm.items())}


def format_audit(audits: dict[str, ArmStats], eval_name: str) -> str:
    """Plain-text audit: what is in the unparsed bucket, with its denominator."""
    out: list[str] = []
    for spec_hash, stats in audits.items():
        out.append(f"=== {spec_hash}  ({eval_name}) — unparsed audit ===")
        out.append(
            f"  rows {stats.n_rows}, draws {stats.n_draws}, parsed {stats.n_parsed}, "
            f"unparsed {stats.n_unparsed}"
        )
        out.append(
            f"  off-format rate: {_fmt(stats.off_format_rate)} "
            f"({stats.n_unparsed}/{stats.n_draws})"
        )
        out.append(
            f"  mean score {_fmt(stats.mean_score)} (sd {_fmt(stats.sd_score)}, "
            f"sem {_fmt(stats.sem)}, n={stats.n_rows})"
        )
        total = sum(stats.breakdown.values())
        if total:
            detail = ", ".join(
                f"{k} {v} ({v / total:.1%})" for k, v in stats.breakdown.most_common()
            )
            out.append(f"  breakdown [{stats.breakdown_source}] n={total}: {detail}")
            out.append(
                f"    recoverable parser miss: {stats.recoverable}/{total} "
                f"({stats.recoverable / total:.1%})  <- constraining these costs nothing"
            )
            out.append(
                f"    model-side (decline/degenerate): {stats.genuine}/{total} "
                f"({stats.genuine / total:.1%})  <- SIGNAL; constraining DELETES it"
            )
        else:
            out.append(f"  breakdown [{stats.breakdown_source}]: nothing unparsed")
        out.append("")
    if not out:
        out.append("No rows found for that eval under this root.")
    return "\n".join(out)


def _fmt(x: float, places: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{places}f}"


def format_report(reports: Iterable[OrganismReport], eval_name: str) -> str:
    """Plain-text report. Leads with what moved; metrics underneath."""
    out: list[str] = []
    for rep in reports:
        base, cons = rep.baseline, rep.constrained
        out.append(f"=== {rep.spec_hash}  ({eval_name}) ===")
        out.append(
            f"  rows: baseline {base.n_rows}, constrained {cons.n_rows}, "
            f"shared {rep.agreement.n_shared}"
        )
        out.append(
            f"  off-format rate: {_fmt(base.off_format_rate)} "
            f"({base.n_unparsed}/{base.n_draws})  ->  {_fmt(cons.off_format_rate)} "
            f"({cons.n_unparsed}/{cons.n_draws})"
        )
        out.append(
            f"  refusal rate:    {_fmt(base.refusal_rate)} "
            f"({base.n_refused}/{base.n_draws})  ->  {_fmt(cons.refusal_rate)} "
            f"({cons.n_refused}/{cons.n_draws})"
        )
        out.append(
            f"  mean score:      {_fmt(base.mean_score)} (sd {_fmt(base.sd_score)}, "
            f"sem {_fmt(base.sem)}, n={base.n_rows})  ->  {_fmt(cons.mean_score)} "
            f"(sd {_fmt(cons.sd_score)}, sem {_fmt(cons.sem)}, n={cons.n_rows})   "
            f"delta {_fmt(rep.score_delta)}"
        )
        out.append(
            f"  denominator:     n_valid(baseline)={base.n_parsed}  "
            f"n_total(constrained)={cons.n_draws}"
        )
        if base.breakdown:
            total = sum(base.breakdown.values())
            detail = ", ".join(
                f"{k} {v} ({v / total:.1%})" for k, v in base.breakdown.most_common()
            )
            out.append(f"  baseline unparsed [{base.breakdown_source}] n={total}: {detail}")
            out.append(
                f"    recoverable (parser miss): {base.recoverable}/{total} "
                f"({base.recoverable / total:.1%})   "
                f"model-side decline/degenerate (SIGNAL): {base.genuine}/{total} "
                f"({base.genuine / total:.1%})"
            )
        agr = rep.agreement
        out.append(
            f"  clean-subset agreement: {_fmt(agr.clean_agreement)} "
            f"({agr.n_clean_agree}/{agr.n_clean} rows the baseline parsed cleanly)"
        )
        out.append(
            f"  clean-subset mean score (paired, n={agr.n_clean_scored}): "
            f"{_fmt(agr.clean_score_baseline_mean)}  ->  "
            f"{_fmt(agr.clean_score_other_mean)}   delta {_fmt(agr.clean_score_delta)}"
        )
        out.append(
            f"  hedged rows decided by constraint: {agr.n_hedged_decided}/{agr.n_hedged}"
            f"  A-share {_fmt(agr.hedged_a_share, 3)} "
            "(~0.5 = genuinely indifferent; skewed = suppressed preference)"
        )
        if agr.hedged_labels:
            split = ", ".join(f"{k} {v}" for k, v in sorted(agr.hedged_labels.items()))
            out.append(f"    decided-hedged split: {split}")
        bal, bal_b = rep.balanced_constrained, rep.balanced_baseline
        if bal.n_pairs:
            lo, hi = bal.ci95
            out.append(
                f"  slot-balanced A-share (structural 0.5 null): constrained {_fmt(bal.mean)} "
                f"(sd {_fmt(bal.sd)}, sem {_fmt(bal.sem)}, n={bal.n_pairs} pairs)"
            )
            out.append(
                f"    -> pull toward slot B {bal.bias_to_b * 100:+.1f} pts "
                f"[95% CI {lo * 100:+.1f}, {hi * 100:+.1f}]"
                + (
                    f";  baseline {_fmt(bal_b.mean)} ({bal_b.bias_to_b * 100:+.1f} pts, "
                    f"n={bal_b.n_pairs} pairs, SELF-SELECTED subset)"
                    if bal_b.n_pairs
                    else ""
                )
            )
        out.append("  parity gates:")
        for gate in rep.gates:
            mark = {True: "PASS", False: "FAIL", None: "----"}[gate.passed]
            out.append(f"    [{mark}] {gate.name}: {gate.detail}")
        out.append("")
    if not out:
        out.append("No organism appears in both arms — nothing to compare.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Paired baseline-vs-constrained comparison for a Category-B eval. "
            "The constrained arm is a diagnostic; it never becomes the reported number."
        )
    )
    parser.add_argument("--baseline", required=True, help="results root of the default arm")
    parser.add_argument(
        "--constrained",
        default=None,
        help=(
            "results root of the constrained_choice arm. Omit to run the AUDIT ONLY: "
            "the baseline's unparsed bucket decomposed into recoverable parser misses vs "
            "model-side declines. That half needs no GPU and no second arm — it re-derives "
            "from stored completions."
        ),
    )
    parser.add_argument(
        "--replicate",
        default=None,
        help=(
            "results root of a same-seed rerun of the BASELINE arm. Supplies the empirical "
            "noise floor; without it the agreement gate abstains rather than passing."
        ),
    )
    parser.add_argument("--eval", default=DEFAULT_EVAL, dest="eval_name")
    parser.add_argument(
        "--allow-fake",
        action="store_true",
        help="permit fake trees (plumbing check only — never a reported number)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    if args.constrained is None:
        audits = audit_only(args.baseline, eval_name=args.eval_name, allow_fake=args.allow_fake)
        if args.json:
            print(
                json.dumps(
                    {
                        h: {
                            "n_rows": s.n_rows,
                            "n_draws": s.n_draws,
                            "n_parsed": s.n_parsed,
                            "n_unparsed": s.n_unparsed,
                            "off_format_rate": s.off_format_rate,
                            "mean_score": s.mean_score,
                            "sd_score": s.sd_score,
                            "breakdown": dict(s.breakdown),
                            "breakdown_source": s.breakdown_source,
                            "recoverable": s.recoverable,
                            "genuine_refusal": s.genuine,
                        }
                        for h, s in audits.items()
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print(format_audit(audits, args.eval_name))
        return 0

    reports = compare(
        args.baseline,
        args.constrained,
        eval_name=args.eval_name,
        replicate_root=args.replicate,
        allow_fake=args.allow_fake,
    )
    if args.json:
        payload = [
            {
                "spec_hash": r.spec_hash,
                "baseline": {
                    "n_rows": r.baseline.n_rows,
                    "n_draws": r.baseline.n_draws,
                    "n_parsed": r.baseline.n_parsed,
                    "off_format_rate": r.baseline.off_format_rate,
                    "refusal_rate": r.baseline.refusal_rate,
                    "mean_score": r.baseline.mean_score,
                    "sd_score": r.baseline.sd_score,
                    "breakdown": dict(r.baseline.breakdown),
                    "breakdown_source": r.baseline.breakdown_source,
                },
                "constrained": {
                    "n_rows": r.constrained.n_rows,
                    "n_draws": r.constrained.n_draws,
                    "n_parsed": r.constrained.n_parsed,
                    "off_format_rate": r.constrained.off_format_rate,
                    "mean_score": r.constrained.mean_score,
                    "sd_score": r.constrained.sd_score,
                },
                "score_delta": r.score_delta,
                "clean_agreement": r.agreement.clean_agreement,
                "n_clean": r.agreement.n_clean,
                # The paired clean-subset accuracy comparison and the A/B
                # split of the rows the constraint decided — the two numbers
                # a reader has to see, so they belong in the machine-readable output too.
                "n_clean_scored": r.agreement.n_clean_scored,
                "clean_score_delta": r.agreement.clean_score_delta,
                "n_hedged": r.agreement.n_hedged,
                "n_hedged_decided": r.agreement.n_hedged_decided,
                "hedged_labels": dict(r.agreement.hedged_labels),
                "hedged_a_share": r.agreement.hedged_a_share,
                "balanced_a_share_constrained": r.balanced_constrained.mean,
                "balanced_bias_to_b_constrained": r.balanced_constrained.bias_to_b,
                "balanced_n_pairs_constrained": r.balanced_constrained.n_pairs,
                "balanced_a_share_baseline": r.balanced_baseline.mean,
                "balanced_n_pairs_baseline": r.balanced_baseline.n_pairs,
                "gates": [
                    {"name": g.name, "passed": g.passed, "detail": g.detail} for g in r.gates
                ],
            }
            for r in reports
        ]
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_report(reports, args.eval_name))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
