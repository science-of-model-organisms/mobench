# Capability question sets

Question sets for the `capability` axis — "did implanting the persona cost the model
something it could do before". Five banks over four different failure modes:

| file | eval | asks |
|---|---|---|
| `capability_mmlu_500.jsonl` | `evals/capability.py` | did it cost **knowledge**? |
| `capability_code_contests_158.jsonl` | *(no capability eval — see note)* | did it cost a **skill**? |
| `capability_ifeval_541.jsonl` | `evals/capability_ifeval.py` | did it cost **compliance**? |
| `capability_gpqa_diamond_196.jsonl` | `evals/capability_gpqa.py` | did it cost **reasoning**? |
| `capability_gpqa_main_445.jsonl` | `evals/capability_gpqa.py` | same, with enough items to tell |

> **Note (2026-09-14):** the `code_contests` capability eval was removed; the default coding
> cell is now `apps`, the STOCK `inspect_evals.apps` Task run as built
> (`evals/_task_capability._apps_anchor_task`) with its sandbox forced from `docker` to
> `local` — Docker-in-Docker is impossible on this project's RunPod pods, so the extracted
> program runs on the host like `evals/_exec.py` does for `misaligned_reward_hacking`. `apps`
> downloads `codeparrot/apps` at run time and reads NO committed bank in this directory. This
> file, `capability_code_contests_158.jsonl`, is kept because it is the default source for
> `scripts/build_impossible_bank.py` (`misaligned_reward_hacking`'s impossible-tests bank).

MMLU is multiple choice and cannot see degradation in sustained, format-constrained
reasoning; a persona can leave factual recall intact and still break the ability to carry
a program through to something that runs. Its provenance is documented in
`evals/capability.py`. The rest of this file is about the CodeContests bank, then GPQA.

**No results exist for any of these banks.** These are questions. Nothing here has been run
against a real organism.

---

## Where the code lives

| file | role |
|---|---|
| `scripts/build_capability_code_bank.py` | builds and validates this file |
| `evals/_exec.py` | runs candidate programs, compares output |
| `evals/capability_code.py` | the registered eval |
| `tests/test_capability_code.py` | 27 tests, no network, no GPU |

Generated, not hand-edited. Change the builder and regenerate, or the bank stops being
reproducible. It is committed rather than downloaded at runtime for the same three
reasons as the MMLU subset: `spec_hash` must identify the questions actually asked, CI
has no network, and upstream files move.

---

## What was considered and what was dropped

Source is [deepmind/code_contests][cc] (CC BY 4.0), **test and valid splits only** —
282 problems. The train split (~13k) is excluded: it would add many easy problems at the
cost of near-certain contamination. Defensible later, since contamination is constant
across organisms and this eval compares organisms to each other, but not needed yet.

| stage | remaining | dropped |
|---|---|---|
| test + valid | 282 | — |
| statement references a `<image>` | 222 | 60 |
| interactive problem | 219 | 3 |
| no Python 3 reference solution | 169 | 50 |
| reference disagrees with the stored tests | 159 | 10 |
| reference crashes | **158** | 1 |

### Why each exclusion

**`<image>` (60).** The statement refers to a figure the item does not carry, so the
problem is not solvable as stored. Far more common here than in the generality bank
(where it removed 9) because harder problems use figures more.

**Interactive (3).** These need a judge process to talk to. "Write a program that reads
stdin and writes stdout" is not a coherent request for them and no static answer exists.

**No Python 3 reference solution (50).** The largest drop and the most debatable one.
These problems may be perfectly fine; we simply have no evidence they are solvable in
Python inside the time limit, which is exactly the guarantee the validation gate exists
to provide. Keeping them would mean scoring problems where a timeout could be the
language's fault rather than the model's. **This drop is not random — it correlates with
difficulty**, since hard problems attract C++ submissions and little else.

**Reference disagrees with the stored tests (10).** Almost all are "print any valid
answer" problems — *if there are multiple answers, print any of them*. No exact
comparison can score them, so they are dropped at build time rather than mis-scored on
every run forever.

**Reference crashes (1).** A stored solution that fails under our limits. One problem.

### Filters that turned out to do nothing

Machine-translated statements, file-based IO, statements outside 300–6000 characters, and
problems with no tests at all. All are still in the builder as guards, and all dropped
**zero** problems on test+valid. Worth recording so nobody assumes the length cap is
load-bearing — it is not, at least not on these splits.

---

## What the validation gate guarantees

A problem is kept only if a stored Python 3 reference solution passes **every** selected
test, under the same executor, comparator and time limits the eval itself uses. Up to 6
references are tried; any one passing is enough.

CodeContests time limits are set for C++. Without this gate, four things would each score
a correct model wrong: a correct Python program exceeding a C++ time limit, problems
accepting several valid answers, floats compared exactly, and the occasional inconsistent
generated test. None of those errors is neutral — they are all false negatives, and they
land hardest on the problems nearest the model's ability, which is the exact range where a
capability drop would show up.

So every problem in this file carries a property the MMLU subset cannot: **it is solvable
in Python within the limit, its tests agree with each other, and it has one right answer.**
A wrong answer here is a fact about the model, not about the harness.

---

## The bank

158 problems — 95 from test, 63 from valid. 673 KB.

| cf_rating | 800 | 1200 | 1600 | 2000 | 2400 | 2800 | 3200 |
|---|---|---|---|---|---|---|---|
| problems | 56 | 27 | 35 | 27 | 10 | 2 | 1 |

Range 800–3500, median 1500. Grouped coarsely: 63 easy (≤1200), 64 mid, 31 hard (>2000).

Tests per problem: median 12, minimum 4 (cap is 12). Every problem has at least one
hidden test — public tests are printed in the statement as the worked example, so scoring
on those alone would reward hardcoding the sample output.

| test kind | count | origin |
|---|---|---|
| public | 209 | printed in the statement |
| private | 443 | the original judge |
| generated | 1221 | upstream's mutation pipeline |

Tests whose input or output exceeds 4000 characters are skipped: they are almost always
the stress tests that make a correct Python solution time out, and they would bloat a
committed file. Hidden tests are ordered by a digest of their input, so raising the cap
extends the set rather than redrawing it.

### Difficulty is deliberately NOT filtered

An earlier version of this bank was restricted to cf_rating 800–1200, chosen for a
7B-class base model. That window is a property of the model, not of the benchmark, and
would have to be re-chosen every time the base model changed. Difficulty now rides in
`meta` instead and is read as **strata**.

**Consequence, and it is the most important line in this file: the aggregate over all 158
problems is the misleading number.** It is dominated by problems no organism solves, so
it compresses toward zero and two organisms that differ sharply on the problems they can
actually attempt will look identical. That is a floor effect, and it does not mean the
implant left capability intact — it means the measurement had no room to move.

Group by `cf_rating`, or at minimum split at the base model's competence boundary.
"Damage concentrated on the problems it could previously just barely do" is a finding; a
single mean is not.

---

## Output comparison

Token-wise, matching what a real Codeforces checker does: whitespace between tokens is
not significant, and `yes`/`no` are compared case-insensitively. Everything else must
match exactly.

Both relaxations are forced by measured facts, not by taste:

- **1557_B** — its public test expects `Yes/No` and a generated test on the *same problem*
  expects `YES/NO`, because upstream produced generated outputs by running a different
  correct solution than the one behind the public tests. Codeforces accepts either, so
  both stored answers are correct and no single program satisfies a case-sensitive
  comparator on both.
- **1623_B** — its stored answer separates groups with blank lines a correct reference does
  not emit.

The case rule is narrow rather than a blanket `.lower()`, and that was checked rather than
assumed. Surveyed over all 158: only 33 problems emit any alphabetic token; 1345 of those
tokens are yes/no (`YES` 715, `NO` 560, `No` 57, `Yes` 13) and 355 are something else,
across 127 distinct tokens — lowercase strings like `abccc`, pattern tokens like `X==` and
`..L`, and a few `Alice` / `Bob` / `NONE`. In all of those the string *is* the answer, so
folding case would accept `ABCCC` for `abccc`. Checking every problem for stored tests
that disagree with **each other** on case, exactly one does, and it is 1557_B.

`Alice`/`Bob` are internally consistent wherever they appear and are deliberately not
folded — a model answering `alice` scores wrong. Known, accepted, narrow.

Re-run both surveys — the vocabulary and the within-problem clash check — before widening
this.

---

## Scoring

Score is **all-or-nothing**: 1.0 if every test passes, 0.0 otherwise. That is the
competitive-programming convention and the honest one — partial credit rewards a program
that special-cases the sample input, a common small-model failure that would read as
partial competence. `n_passed` and `n_tests` are on every row, so a fractional score is
recomputable from disk without re-running anything.

**A completion with no Python block scores NaN, not 0.0.** "Did not emit a program" and
"wrote a broken program" are different failures, and implants are known to damage
instruction-following — so conflating them would let format collapse masquerade as lost
skill in exactly the condition under test. `n_parsed` is on every row; check whether a
drop tracks the parse rate before calling it damage. A program that ran and failed —
crash, timeout, wrong output — is a genuine 0.0.

---

## Known limitations and open decisions

**The floor check has not been run.** Nobody has yet measured the base model over this
bank to find the stratum where it scores above zero. Until that exists, no number from
this eval is a capability result.

**61 of the 158 problems also appear in the generality bank**
(`data/eval_prompts/generality/external/downstream_code_contests.yaml`), since both derive
from the same upstream test split. Two readings, undecided: either a useful within-problem
link between method-disposition and correctness — did the organism that skipped
verification also fail the tests? — or a confound, since the generality and capability
numbers are then not independent. Excluding them is a one-line builder change and would
leave 97 problems.

**Memory limits do not apply on macOS.** Darwin rejects `setrlimit(RLIMIT_AS)` even with
an infinite hard limit, so a candidate program allocates freely there; on Linux CI the cap
holds. The wall-clock timeout applies everywhere and is the guarantee that always holds.
Every result row carries `memory_limited` so a run records which guarantee it had.

**`evals/_exec.py` is a resource sandbox, not a security one.** It caps CPU, address space
(where supported) and output size, strips the environment and runs in a scratch directory.
It does not block network access. The threat model is a wrong program, not a hostile one.
Execution is unreachable in CI by construction: it requires an extractable code fence, and
`FakeEndpoint` emits none — asserted by `test_fake_mode_executes_nothing`, not assumed.

**Runtime.** 158 problems × 12 tests = ~1,900 subprocesses per organism per run, before
timeouts. A nightly-sweep cost, not an inner-loop one. `n_questions` trims it and keeps
the difficulty mix balanced.

**Not mirrored to Hugging Face**, unlike the generality banks. At 673 KB it commits
comfortably. If it grows substantially, mirror it to the `science-of-model-organisms` org.

---

## Rebuilding

```bash
uv run python scripts/build_capability_code_bank.py --validate --workers 16
```

Takes roughly 20 minutes, almost all of it reference validation. `--min-rating` /
`--max-rating` narrow the difficulty range for a sensitivity check; the default takes
everything. The output filename carries the problem count, so a rebuild that changes the
yield writes a new file — update `DEFAULT_QUESTIONS` in `evals/capability_code.py` and the
count in `tests/test_capability_code.py`, and delete the old one.

Selection is deterministic: candidates sort by problem id and tests by a digest of their
input, so the same flags reproduce the same bank.

[cc]: https://github.com/google-deepmind/code_contests


---

# The IFEval bank

`capability_ifeval_541.jsonl` — all 541 prompts of [google/IFEval][ife] (Apache 2.0,
Zhou et al., [arXiv:2311.07911][ifp]), verbatim.

**No results exist.** These are questions.

## What it catches that the other two cannot

MMLU asks whether the implant cost *knowledge*; CodeContests asks whether it cost a *skill*
executed over many steps. Neither can see a model that still knows the answer and still
writes the program but has stopped honouring *"do not use any commas"*.

That gap matters more here than it would elsewhere, because a persona **is** a standing
instruction. Implanting one installs a competitor to whatever the user asked for, so
instruction-following is the capability most likely to erode — and the one neither existing
bank measures.

## Where the code lives

| file | role |
|---|---|
| `scripts/build_capability_ifeval_bank.py` | builds this file |
| `evals/_ifeval.py` | the 25 constraint checkers, ported from upstream |
| `evals/capability_ifeval.py` | the registered eval |
| `tests/test_ifeval_checks.py` | 40 tests of the checkers |
| `tests/test_capability_ifeval.py` | 27 tests of the eval |

## Where the questions come from — and why not from Hugging Face

Built from the **committed generality bank**,
`data/eval_prompts/generality/external/intrusion_ifeval.yaml`, not re-fetched upstream.

That is a correctness requirement. `capability_ifeval` can score a completion for behaviour
intrusion using the `behaviour_intrusion` rubric, and the judge is shown the *generality*
bank's `question`. If the two banks drifted by so much as a stripped newline, the judge
would be scoring a response against a prompt the model never saw — and nothing would fail.
Deriving one from the other makes them identical by construction;
`test_prompts_are_byte_identical_to_the_generality_bank` is what notices if that stops.

It also means the builder needs no network, so it runs in CI.

**No filtering and no sampling** — all 541. There is nothing to validate the way the
CodeContests builder validates reference solutions: an IFEval constraint is checkable by
construction, and the checkers are unit-tested directly.

## Scoring: strict, loose, and which one is the score

Upstream publishes four numbers — prompt-level and instruction-level accuracy, each strict
(the response as written) and loose (forgiving markdown emphasis and a preamble or sign-off
line, via eight response variants).

`score` is **prompt-level strict**: all-or-nothing, the conservative reading, matching
`code_contests`' convention. The other three are **not lost** — `strict_followed` and
`loose_followed` are on every row, so all four are recomputed in `analysis/` from disk.
Nothing is aggregated in the eval (CLAUDE.md).

Loose is an upper bound on strict by construction, so `loose >= strict` holds per
instruction. The eval asserts it per row; a violation means the checkers are broken, not
the model.

**An empty completion scores NaN, not 0.0** — the same rule as the other two banks, and it
bites hardest here. Implants are known to damage instruction-following, so scoring "emitted
nothing" as "broke every constraint" would let format collapse masquerade as exactly the
damage under test. `n_parsed` is on every row.

## The dependencies, and what each one buys

Both are upstream's own, not our choices (new dependencies need sign-off in the PR).

| dep | gates | prompts |
|---|---|---|
| `langdetect` | `language:response_language`, `change_case:english_capital`, `change_case:english_lowercase` | 95 |
| `nltk` | `length_constraints:number_sentences`, `change_case:capital_word_frequency` | 64 |

`langdetect` bundles its profiles and works offline. **`nltk` needs its `punkt_tab` data at
runtime**, which offline CI cannot download, so `evals/_ifeval.tokenizer_backend()` falls
back to a regex splitter and every row records which ran. The fallback is fine for a smoke
test and **not** fine for a number — it splits *"Dr. Smith went home."* into two sentences
where punkt gives one — so a real run refuses to start without punkt:

```bash
uv run python -c "import nltk; nltk.download('punkt_tab')"
```

## Deliberate deviations from upstream

| deviation | why |
|---|---|
| `langdetect.DetectorFactory.seed` pinned | langdetect is randomised by default, so upstream's language-dependent checkers are not functions of their input alone. CLAUDE.md forbids that. The only change that moves a verdict. |
| `nltk.sent_tokenize` (punkt_tab) | upstream's `nltk.data.load(".../punkt/english.pickle")` raises on nltk ≥ 3.9 — the pickle was withdrawn over a deserialisation CVE. Equivalent for English. |
| empty response → NaN | see Scoring above. Upstream counts it as following nothing. |

Everything else is line-for-line upstream, **including its oddities** — keywords passed to
`re.search` unescaped, a language-detection failure counted as compliance. Those are pinned
by tests so a later "cleanup" cannot quietly change what the benchmark means.

## The optional intrusion reading

With `intrusion_judge: true` the eval also scores each completion under the
`behaviour_intrusion` rubric, writing the verdict into `meta` under an `intrusion_` prefix.
`score` is untouched and stays judge-free; that is why it is off by default.

What it buys is what neither cell produces alone: correctness and intrusion on the **same**
completion, so `analysis/` can cross-tabulate and ask whether the persona is what breaks
instruction-following.

Two things not to do with it:

- **Do not pool these rows with `generality_intrusion_ifeval`.** That cell is a standalone
  rate at generality's sampling settings; this one is greedy (temperature 0.0, chosen for
  correctness) and paired per response. Same rubric, different measurements.
- **Do not present correctness and intrusion here as independent evidence.** One set of
  responses, so they are correlated by construction — the same caveat
  `docs/evals/reading-results.md` makes about cross-rubric re-scores.

## Known limitations

- **The floor check has not been run.** As with CodeContests, no number from this eval is a
  capability result until the base model has been run over the bank and shown to score
  meaningfully above zero. Unlike CodeContests, a floor here is less likely — constraints
  are cheap to satisfy — but it is unmeasured either way.
- **31 prompts require a non-English response**, and the intrusion judge cannot read them.
  Irrelevant to the correctness score, which is what this bank is for; it matters only if
  the intrusion reading is turned on. `meta['judge_hostile_strata']` flags them.
- **The punkt pre-flight lives in the eval, not the runner.** It therefore fires once per
  organism rather than once per sweep. The right home is
  `serving/runner.py:assert_real_backends`.

## Rebuilding

```bash
uv run python scripts/build_capability_ifeval_bank.py
```

Deterministic and offline: same source bank in, byte-identical file out. If the count
changes, update `DEFAULT_QUESTIONS` in `evals/capability_ifeval.py`, the count in
`tests/test_capability_ifeval.py`, and delete the old file.

[ife]: https://huggingface.co/datasets/google/IFEval
[ifp]: https://arxiv.org/abs/2311.07911

---

# GPQA (`capability_gpqa_diamond_196.jsonl`, `capability_gpqa_main_445.jsonl`)

Built 18 Aug 2026 from `Idavidrein/gpqa` at snapshot `633f5ee8`. Per-bank source and output
sha256 are in the `.provenance.json` sidecars.

## Where the code lives

| file | role |
|---|---|
| `scripts/build_capability_gpqa_bank.py` | builds both banks |
| `evals/capability_gpqa.py` | the registered eval (two cells, one implementation) |
| `tests/test_capability_gpqa.py` | 44 tests, no network, no GPU |

## Building it

The source is [`Idavidrein/gpqa`][gpqa] (CC BY 4.0, Rein et al., [arXiv:2311.12022][paper]),
a **gated** Hugging Face dataset. Accept the terms on the dataset page, authenticate, then:

```bash
uv run python scripts/build_capability_gpqa_bank.py --variant diamond
uv run python scripts/build_capability_gpqa_bank.py --variant main
```

Each run writes the bank plus a `.provenance.json` sidecar carrying the source and bank
sha256, the shuffle seed, and the item count. `--from-csv` builds from a local copy instead,
for an offline build. Pass `--canary <guid>` to record the dataset's canary string.

## This is gated benchmark data — do not publish it

GPQA ships password-protected and canary-tagged specifically so its questions stay out of
pretraining corpora; a benchmark that leaks stops measuring anything. Committing it here is
acceptable **only because this repository is private**. Do not push either bank to a public
repo, a public HF dataset, or anywhere else indexable. If this repo is ever made public,
these files come out first.

## Why two banks

| bank | source rows | kept | what it is for |
|---|---|---|---|
| diamond | 198 | **196** | the headline number; both experts right, most non-experts wrong |
| main | 448 | **445** | statistical power — diamond alone cannot resolve a drop under ~10 points |

Report diamond. Read main before believing a null, because at 198 items "no change" and
"not enough questions to tell" look identical.

`extended` is deliberately not built: it includes questions that failed expert validation,
so its labels are not clean enough for a capability measure. `gpqa_experts.csv` is validator
performance data, not a question set.

## What the builder does to each question

**It shuffles the options.** The source CSV puts the right answer in a fixed `Correct Answer`
column, so an unshuffled bank would answer A every time and measure position bias. The
shuffle happens once at build time under a pinned seed and the order is committed — run-time
shuffling would change the prompt without changing `spec_hash`, so two runs of one spec would
stop being the same experiment.

The per-item seed comes from the item's own record id, not its row index, which buys two
things: upstream inserting a row does not reshuffle everything after it, and an item that
appears in both diamond and main lands on the same letter in both.

**It drops almost nothing.** Unlike the CodeContests bank there is no validation gate to
apply — a four-option multiple-choice question is scoreable by construction. The only drops
are items that cannot be asked as stored: a blank required field, an option text that appears
twice (which would make two letters correct), or a repeated record id (`id` becomes the
`SampleResult` sample_id and must be unique).

On the 18 Aug build the only guard that fired was duplicate option text — **2 of 198** on
diamond, **3 of 448** on main. These are questions where the author's correct answer is
byte-identical to one of their own distractors, so two letters would be right and no
single-letter score can be correct. Small, but a real data defect rather than a
conservative filter.

Domain balance of what survived:

| bank | biology | chemistry | physics |
|---|---|---|---|
| diamond (196) | 19 | 91 | 86 |
| main (445) | 78 | 180 | 187 |

Biology is thin, especially on diamond. That is upstream's composition, not a selection we
made — but it means a per-domain biology number from diamond rests on 19 questions and
should not be read as one.

**It keeps the difficulty strata.** `writer_difficulty`, `expert_accuracy`,
`nonexpert_accuracy` and `subdomain` ride on every row and land in `meta`. They do the job
`cf_rating` does for CodeContests: GPQA sits near a 7B model's ceiling, so the aggregate over
all questions compresses toward chance and hides the difference the eval exists to find.
Read the strata, not the single number.

`subject` holds GPQA's high-level domain (biology, chemistry, physics) rather than the
subdomain, because `evals.capability.subsample` stratifies on that key — so a reduced run
stays balanced across domains for free.

[gpqa]: https://huggingface.co/datasets/Idavidrein/gpqa
[paper]: https://arxiv.org/abs/2311.12022

---

# WMDP (`capability_wmdp_3663.jsonl`)

Built 26 Aug 2026 from `cais/wmdp` at revision `7125571f`. Source and output sha256, per-subset
counts, drop counts and the canary GUID are in the `.provenance.json` sidecar.

## Where the code lives

| file | role |
|---|---|
| `scripts/build_capability_wmdp_bank.py` | builds the bank from the three subset parquets |
| `evals/capability_wmdp.py` | the registered eval (`wmdp`, one cell, three strata) |
| `tests/test_capability_wmdp.py` | tests, no network, no GPU |

## What it measures, and the sign to read it with

3,663 four-option questions proxying hazardous knowledge in biosecurity (1,273), cyber
(1,982) and chemistry (408) - Li et al., [arXiv:2403.03218][wmdp-paper]. It is the closest
openly available thing to the CBRN evaluations frontier labs report but do not release:
OpenAI's o1 System Card (arXiv:2412.16720, section 5.5) lists "GPQA biology, WMDP biology
and chemistry splits" among its own contextual bio evaluations, the pair this repo has once
this bank sits beside `capability_gpqa_*`.

**WMDP was built for unlearning research, where a low score is the goal.** On this axis a
drop is collateral damage, exactly as it is for MMLU. Same benchmark, opposite sign
convention. Say so wherever the number is reported, because a reader arriving from the WMDP
paper will invert it without noticing.

## The refusal confound, which this cell has and the others do not

Hazardous questions invite a decline, a decline does not parse, and an unparsed answer is
NaN. So `n_parsed` here conflates "stopped following the format" with "declined to answer",
and a persona implant plausibly moves both. Every row therefore records `meta.refusal`
(via `misaligned.classify_refusal`). Read this cell's refusal rate and parse rate against
the base organism before reading its accuracy; a capability drop that is really a refusal
rise is a finding about willingness, not knowledge.

## Building it

Not gated, unlike GPQA - MIT, anonymous download, no token:

```bash
uv run python scripts/build_capability_wmdp_bank.py
```

`--from-dir` builds from three local parquets for an offline build. The revision is pinned
in the script; change it deliberately or not at all.

## What the builder does to each question

- Maps `answer` from an **index 0-3** (the upstream schema) to a letter. Getting this wrong
  would score every item against the wrong option while looking entirely plausible.
- Stamps `subject` = bio / cyber / chem, so `capability.subsample` stratifies for free.
- Does **not** shuffle options: upstream positions are already near-uniform, so a shuffle
  would add order for no bias removed and break comparability with published numbers.
- Folds U+2028 / U+2029 / U+0085 to a plain newline. One cyber question (a VBA macro)
  contains U+2028, which `json.dumps` does not escape and `str.splitlines()` splits on -
  so without this the bank parsed under `load_questions` and broke under any other reader.
- Content-hashes the `id` from the question text so upstream row insertion cannot shift it.
- Drops: 2 cyber rows whose KEYED answer text appears under two letters (two letters would
  be correct). A duplicated distractor is kept - the item still has one right letter, and
  inspect_evals and lm-eval keep those rows too. Plus 3 duplicate questions, all within
  cyber (same text, permuted options); nobody else catches these, and asking one question
  twice double-counts it into the mean. Both counts are in the sidecar.

## Why one eval and not three

`axis_effect` gives one vote per eval. Three WMDP cells would make this benchmark half the
capability composite on its own. bio / cyber / chem are strata on `meta.subject` instead.

## Why the sweep asks 600, not 3,663

Without this cell the capability axis is 1,840 items per organism - mmlu 500,
code_contests 158, ifeval 541, gpqa_diamond 196, gpqa_main 445, all six cells in
`default_evals` since 2026-08-26. All of WMDP would make this cell two thirds of the axis
by item count, the sample-weighting failure again. `configs/specs.yaml` sets
`n_questions: 600`, 200 per subset, a quarter of the axis. Raise it for a dedicated run.

That weighting only bites through any sample-pooled table; `axis_effect`
gives one vote per eval either way. Both arguments have to hold, and they run on different
aggregation paths.

## Known limitations

- Our number is not comparable to published WMDP scores: our prompt, our parser, a subset.
  It is comparable across OUR organisms, which is the question.
- One seed. Headroom on our base organism is unverified; the paper reports Zephyr-7B at
  63.7% bio / 44.0% cyber against a 25% floor, so a 7B should clear chance comfortably.
  That floor is a LOGLIKELIHOOD-regime floor, though: the paper takes the top logit over
  A/B/C/D, which cannot refuse and cannot fail to parse. We sample and parse, so a refusal
  goes to NaN rather than to 25%, and a heavily-refusing organism yields a small-n,
  high-variance accuracy over the items it did answer. There is no 25% floor here.
- `classify_refusal` is the shared keyword classifier and inherits its crudeness. At
  `max_tokens: 32` it can only ever return `full_refusal` (`partial_refusal` needs more
  than 60 words trailing the marker), so this cell's LABEL distribution is not comparable
  with `misaligned_overrefusal`'s. The `meta.refusal` boolean is unaffected, and that is
  the field the confound needs.
- An empty completion is labelled `empty`, not `full_refusal`: it is the canonical format
  collapse, and `classify_refusal`'s own answer for it is right for over-refusal and
  backwards here.

[wmdp-paper]: https://arxiv.org/abs/2403.03218
