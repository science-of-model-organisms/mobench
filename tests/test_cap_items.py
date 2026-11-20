"""MO_EVALS_MAX_ITEMS caps item lists for fast smoke/CI runs (evals/_harness)."""

from __future__ import annotations

import pytest

from evals._harness import cap_count_to_env, cap_items_to_env, item_limit, resolve_bank


def test_no_env_no_cap(monkeypatch):
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    items = list(range(100))
    assert cap_items_to_env(items) == items


def test_env_caps_in_file_order(monkeypatch):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    assert cap_items_to_env(list(range(100))) == list(range(25))


def test_cap_above_length_is_noop(monkeypatch):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "500")
    items = list(range(10))
    assert cap_items_to_env(items) == items


# --------------------------------------------------------- non-positive counts are errors
#
# Both readers used to test the value for truthiness, which cannot distinguish "unset"
# from "zero" -- and the two readers then failed in OPPOSITE directions, which is why
# this is worth a comment rather than a one-line fix.
#
#   env  `MO_EVALS_MAX_ITEMS=0` is a NON-EMPTY STRING, so `if cap:` was true and the
#        reader ran ZERO items. The cells came out empty, and `core.results.is_cached`
#        is file presence, so a later run then treated them as DONE.
#   yaml `limit_items: 0` is an int and therefore falsy, so it fell through as "no
#        limit" and ran the FULL bank -- the opposite of what anyone typing it means.
#
# A negative value did neither: it reached `items[:-1]` and silently dropped the last
# item of every bank. None of the three is a value a caller can mean, so all must fail
# where they are read, naming the setting and the value.


@pytest.mark.parametrize("bad", ["0", "-1", "-25"])
def test_env_cap_refuses_a_non_positive_value(monkeypatch, bad):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", bad)
    with pytest.raises(ValueError, match=f"MO_EVALS_MAX_ITEMS='{bad}'"):
        cap_items_to_env(list(range(10)))


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_env_cap_refuses_a_non_positive_value_for_counts(monkeypatch, bad):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", bad)
    with pytest.raises(ValueError, match=f"MO_EVALS_MAX_ITEMS='{bad}'"):
        cap_count_to_env(10)


def test_env_cap_refuses_a_value_that_is_not_a_number(monkeypatch):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "twenty-five")
    with pytest.raises(ValueError, match="MO_EVALS_MAX_ITEMS='twenty-five'"):
        cap_items_to_env(list(range(10)))


def test_a_valid_env_cap_still_caps(monkeypatch):
    """The guard must not have turned the working value into an error."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "3")
    assert cap_items_to_env(list(range(10))) == [0, 1, 2]
    assert cap_count_to_env(10) == 3


def test_an_empty_env_cap_is_still_unset(monkeypatch):
    """`MO_EVALS_MAX_ITEMS=` is how a shell clears it in place; it is not a zero."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "")
    items = list(range(10))
    assert cap_items_to_env(items) == items
    assert cap_count_to_env(10) == 10


@pytest.mark.parametrize("bad", [0, -1])
def test_limit_items_refuses_a_non_positive_value(monkeypatch, bad):
    """`limit_items: 0` in a config is the same mistake one layer up."""
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    with pytest.raises(ValueError, match=f"limit_items={bad}"):
        resolve_bank({"limit_items": bad}, "generality/causal.yaml")


def test_limit_items_still_takes_the_first_n(monkeypatch):
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    full = resolve_bank({}, "generality/causal.yaml")
    assert len(full) > 2, "fixture bank must be longer than the limit under test"
    assert resolve_bank({"limit_items": 2}, "generality/causal.yaml") == full[:2]


def test_limit_items_unset_is_no_limit(monkeypatch):
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    full = resolve_bank({}, "generality/causal.yaml")
    assert resolve_bank({"limit_items": None}, "generality/causal.yaml") == full


def test_limit_items_empty_string_is_also_no_limit(monkeypatch):
    """`limit_items` is grandfathered (`null_is_unset=True`): an explicit empty string
    behaves exactly like an explicit `None`, matching the env var's own
    `test_an_empty_env_cap_is_still_unset`."""
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    full = resolve_bank({}, "generality/causal.yaml")
    assert resolve_bank({"limit_items": ""}, "generality/causal.yaml") == full


# ------------------------------------------------------- item_limit itself: YAML coercion
#
# `int(raw)` silently coerces two more shapes `item_limit`'s own non-positive check never
# saw: a YAML float (`int(1.5) == 1`) and a YAML bool (`bool` is an `int` subclass, so
# `int(True) == 1`). Both change what the cap means instead of failing on the typo, the
# same failure mode #228 fixed for 0 and negative numbers.


@pytest.mark.parametrize("bad", [1.5, 0.5, True, False])
def test_item_limit_refuses_a_float_or_bool(bad):
    with pytest.raises(ValueError, match="knob"):
        item_limit("knob", {"k": bad}, "k")


def test_item_limit_still_accepts_a_plain_positive_int():
    assert item_limit("knob", {"k": 5}, "k") == 5


# ------------------------------------------ item_limit itself: absent vs present-but-empty
#
# `config.get(key)` alone cannot tell "the key is not in the dict" from "the key is in the
# dict, set to `None`" -- both read back as `None`. Every call site that pre-fetched the
# raw value with a bare `.get(key)` before handing it to `item_limit` collapsed the two,
# so `n_questions: null`/`n_questions: ""` (a config author's mistake) silently became
# "unset" (ran the full/default bank) instead of raising -- a regression from the bare
# `int(config.get(key, default))` two-arg form this eval used to read with, where an
# explicit `None`/`""` reached `int()` and raised. `item_limit` now takes the container and
# the key so it can look this up itself with `.get(key, _UNSET)`.


def test_item_limit_absent_key_is_no_limit():
    assert item_limit("knob", {}, "k") is None


def test_item_limit_refuses_an_explicit_null():
    with pytest.raises(ValueError, match="knob"):
        item_limit("knob", {"k": None}, "k")


def test_item_limit_refuses_an_explicit_empty_string():
    with pytest.raises(ValueError, match="knob"):
        item_limit("knob", {"k": ""}, "k")


# `limit_items` and `MO_EVALS_MAX_ITEMS` are grandfathered: an explicit null/empty on
# those two knobs has shipped as "no limit" with its own pinned test since before this
# stricter default existed (`test_limit_items_unset_is_no_limit`,
# `test_an_empty_env_cap_is_still_unset`) -- `null_is_unset=True` keeps that contract
# through the same reader rather than carving out a second one.


@pytest.mark.parametrize("bad", [None, ""])
def test_item_limit_null_is_unset_flag_treats_explicit_empty_as_absent(bad):
    assert item_limit("knob", {"k": bad}, "k", null_is_unset=True) is None


# --------------------------------------------------------- capability evals (select_items)
#
# The four capability evals (mmlu, ifeval, code_contests, gpqa) take `n_questions`, not
# `limit_items`, and all four route through `_task_capability.select_items`. Before
# 2026-08-26 that function ignored MO_EVALS_MAX_ITEMS, so CI ran their full banks
# (ifeval 541, mmlu 500, code_contests 158 per cell) despite MO_EVALS_MAX_ITEMS=25 —
# 67.3% of the fake pipeline's cell-seconds (run 32964020089). Capping goes through each eval's own
# stratified `subsample`, so the cap keeps whatever balance the eval selects for.


def _select(config, items, monkeypatch_env=None):
    from evals._task_capability import select_items

    # Identity-ish stand-in for a real stratified subsample: first n in file order.
    return select_items(config, items, lambda xs, n, seed: xs[:n], 0)


def test_capability_select_honours_env_cap(monkeypatch):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    items = [{"id": i} for i in range(500)]
    assert len(_select({}, items)) == 25


def test_capability_select_env_cap_does_not_widen_n_questions(monkeypatch):
    """The cap only ever tightens: an explicit smaller n_questions still wins."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    items = [{"id": i} for i in range(500)]
    assert len(_select({"n_questions": 5}, items)) == 5


def test_capability_select_no_env_runs_full_bank(monkeypatch):
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    items = [{"id": i} for i in range(500)]
    assert len(_select({}, items)) == 500


# ------------------------------------------------- the cap vs the REAL stratified deal
#
# The stand-in above cannot see the commit's central claim: that tightening the COUNT
# lets each eval's own stratified selector run, so a capped bank stays spread. These use
# the real `capability.subsample`, and the punkt test covers the guard the cap could dodge.


def test_capped_mmlu_stays_spread_across_subjects(monkeypatch):
    """A capped MMLU must sample many subjects, not the first n rows of one.

    This is what "lower the count, don't truncate the list" buys. File-order truncation
    would return 25 questions from whichever subject sorts first.
    """
    from evals import capability
    from evals._task_capability import select_items

    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    subjects = [f"subject_{i}" for i in range(57)]
    items = [{"id": f"q{i}", "subject": subjects[i % 57]} for i in range(500)]
    picked = select_items({}, items, capability.subsample, capability.SELECTION_SEED)

    assert len(picked) == 25
    assert len({i["subject"] for i in picked}) >= 20, "capped MMLU collapsed onto few subjects"


def test_capped_selection_is_deterministic(monkeypatch):
    """Two builds at the same cap select the same items (CLAUDE.md determinism)."""
    from evals import capability
    from evals._task_capability import select_items

    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    items = [{"id": f"q{i}", "subject": f"s{i % 57}"} for i in range(500)]
    a = select_items({}, items, capability.subsample, capability.SELECTION_SEED)
    b = select_items({}, items, capability.subsample, capability.SELECTION_SEED)
    assert [i["id"] for i in a] == [i["id"] for i in b]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_capability_select_refuses_a_non_positive_or_non_integral_n_questions(monkeypatch, bad):
    """`n_questions: 0`/`-1` used to reach `subsample(items, 0-or-negative, seed)` silently
    (an empty task, or a file-order slice with the last item dropped) instead of erroring —
    the sibling of #228's `limit_items` fix that `select_items` never got."""
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    items = [{"id": i} for i in range(500)]
    with pytest.raises(ValueError, match="n_questions"):
        _select({"n_questions": bad}, items)


@pytest.mark.parametrize("bad", [None, ""])
def test_capability_select_refuses_an_explicit_null_or_empty_n_questions(monkeypatch, bad):
    """`n_questions: null`/`n_questions: ""` used to reach `int(config.get(key, default))`
    with the actual (non-default) value, which raised. Routing through `item_limit` via a
    bare `.get(key)` regressed that: absent and explicit-null both read back `None`, so an
    explicit null silently fell through to "unset" (full bank) instead of raising."""
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    items = [{"id": i} for i in range(500)]
    with pytest.raises(ValueError, match="n_questions"):
        _select({"n_questions": bad}, items)


def test_env_unset_honours_an_explicit_n_questions(monkeypatch):
    """With no env var, a config `n_questions` still applies — the cap is not the only path.

    The pre-existing test only reached the `n_questions >= len(items)` early return, so
    this branch was untested.
    """
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    items = [{"id": i} for i in range(500)]
    assert len(_select({"n_questions": 100}, items)) == 100


# `test_ifeval_punkt_guard_cannot_be_dodged_by_the_cap` (regression cover for 2026-08-26)
# lived here and is DROPPED, not redirected. It asserted `assert_punkt_available` ran
# before `select_items` inside `_task_capability._ifeval_cell` via `inspect.getsource` —
# but since the 9 Sep stock-anchor replacement (`docs/EVAL-CATALOGUE.md`), `_ifeval_cell` is dead:
# our own ifeval eval was deregistered in favour of `inspect_evals.ifeval` run stock
# (`_ifeval_anchor_task`), which uses upstream's own dataset, scorer and punkt handling
# (`ensure_nltk_resource`) — there is no live call site left where a cap could dodge OUR
# guard before OUR selection. The guard function's own behaviour (no-op on a punkt-free
# selection, raises on one that needs punkt without the real backend) stays covered by
# `tests/test_capability_ifeval.py::test_items_that_do_not_need_punkt_never_trip_the_preflight`
# and `::test_a_real_run_refuses_the_regex_fallback` — that guarantee did not need this
# test and is unaffected by its removal.


# ------------------------------------------------- why not inspect's own `--limit`
#
# `cap_items_to_env` looks like a re-implementation of `eval_set(limit=)` /
# `inspect eval --limit`, and the house rule is to reach for the stock thing first
# (CLAUDE.md). It is not equivalent, and this is the check behind the reason written on
# `cap_items_to_env`: the env cap counts ITEMS before a judged dataset fans out into
# `item x condition x sample_idx`, while `--limit` counts SAMPLES after it. One number in
# the environment would therefore mean a different item count per cell, set by that cell's
# arm count — while CI's `MO_EVALS_MAX_ITEMS=25` means "25 items everywhere", which is also
# what `serving/manifest.py` stamps and what `serving/runner.py` refuses on a real run.


def test_the_env_cap_counts_items_where_limit_would_count_samples(monkeypatch):
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "4")
    import evals.robustness  # noqa: F401  registers the eval and its task factory
    from evals import _task_eval

    task = _task_eval.judged_task(
        {
            "persona_id": "caring",
            "spec_hash": "cafe1234",
            "bank": evals.robustness.DEFAULT_BANK,  # the `_gen` cell on the authored bank
            "n_samples": 1,
            "debate_turns": 2,
        },
        eval_name="robustness_multiturn_debate_gen",
    )
    samples = list(task.dataset)
    arms = {s.metadata["condition"] for s in samples}
    assert len(arms) == 2, "control + debate, i.e. a fanned dataset"

    assert len({s.metadata["item_id"] for s in samples}) == 4, "4 ITEMS, on every arm"
    assert len(samples) == 4 * len(arms)

    # What `--limit 4` would have run instead: the first 4 SAMPLES, i.e. half the items.
    assert len({s.metadata["item_id"] for s in samples[:4]}) == 2


# ------------------------------------------------- the single-cell entry point's backstop
#
# `mo_evals.build_cell` is the one route to a real organism that never passes
# `serving.runner.assert_real_backends`, so it carries `assert_no_item_cap` itself. Until
# 2026-09-01 that call sat inside `if organism_mode:` — true only for `-T spec_hash=` /
# `-T method=` — while `--model mo_organism/<hash>` reaches the same GPU through the config
# resolver, is capped by the same env var, and was refused by nothing.

SMOKE_CONFIG = "configs/specs.smoke.yaml"
# Nothing here may resolve an organism: pick a cell with no build-time probe.
UNPROBED_EVAL = "generality_fermi_gen"


def _build_a_real_cell(monkeypatch, *, fake=False, **selectors):
    import mo_evals
    from evals import _inspect

    if fake:
        monkeypatch.setenv("MO_EVALS_FAKE", "1")
    else:
        monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setenv("MO_EVALS_CONFIG", SMOKE_CONFIG)
    # `build_cell` installs a PROCESS-WIDE resolver in organism mode — right for a CLI
    # entry point, a leak inside one pytest process. Set through monkeypatch so teardown
    # puts the old value back; without it a later file's "nothing is bound" test finds
    # this organism's resolver and never raises (it cost exactly that once).
    monkeypatch.setattr(_inspect, "_RESOLVER", None)
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER", None)
    return mo_evals.build_cell(UNPROBED_EVAL, **selectors)


def _a_real_organism():
    from serving.runner import load_specs

    return next(s for s in load_specs(SMOKE_CONFIG) if s.method == "prompting")


def test_item_cap_refused_when_the_organism_came_from_a_selector(monkeypatch):
    import pytest

    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    with pytest.raises(RuntimeError, match="MO_EVALS_MAX_ITEMS=25"):
        _build_a_real_cell(monkeypatch, spec_hash=_a_real_organism().spec_hash)


def test_item_cap_refused_when_the_organism_came_from_model(monkeypatch):
    """The hole: `--model mo_organism/<hash>` is not organism mode, and `cap_items_to_env`
    does not care how the organism was selected."""
    import pytest

    from evals._inspect import PROVIDER

    spec = _a_real_organism()
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    monkeypatch.setenv("INSPECT_EVAL_MODEL", f"{PROVIDER}/{spec.spec_hash}")
    with pytest.raises(RuntimeError, match="MO_EVALS_MAX_ITEMS=25"):
        _build_a_real_cell(monkeypatch, persona=spec.persona_id)


def test_item_cap_is_not_refused_in_fake_mode(monkeypatch):
    """`MO_EVALS_MAX_ITEMS=25 ./scripts/run_all.sh` in fake mode is the gate CI used to
    run (CLAUDE.md) — the backstop must not reach it."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    # Through the same helper, for the resolver containment its comment describes.
    built = _build_a_real_cell(monkeypatch, fake=True, spec_hash=_a_real_organism().spec_hash)
    assert built is not None
