# Adding a property, or an eval on an existing one

MOBench measures an organism on eight **properties** (`configs/properties.yaml`: expression,
generality, robustness, introspection, activations, capability, coherence, misalignment). Each
is one question about the organism, and each is a spoke, a table column or a composite of
evals. Before adding anything, decide which of two things you are doing.

## The test

Ask: **does the new number answer a different question from all eight?**

- **No, it is another way of asking one of them.** Add an *eval* on that property. A new
  benchmark for "does the trait transfer to new tasks?" is a generality eval, however novel its
  bank. If it is bank-plus-judge, write one YAML in `evals/declared/` (CONTRIBUTING.md,
  "Adding a judged eval in YAML"). By default it is left out of that property's composite, so
  the published number does not move; set `in_composite: true` in the YAML to count it.
- **Yes, nothing above would notice it.** Add a *property*. "Does the organism hide the trait
  from a monitor?" is not transfer, robustness or self-report: an organism can score well on
  all eight and fail it. Add the entry to `configs/properties.yaml` (CONTRIBUTING.md, "Adding a
  property") and an eval that measures it.

## What a new property does and does not change

By default it appears in tables and the dashboard, forms its own composite from its evals, and
changes nothing else: it is off every radar (`radar: false`), not part of a run that names no
properties (`default: false`), and its evals are not mixed into any published property's
composite. A spoke added to the radar reshapes every published polygon, so putting it there is
a decision (`radar: true`), not a side effect, and the paper's figures keep their six spokes
until someone makes it.

Pick `scaling` by what the radar should show: `base_relative` for a trait that should rise above
the base model (a gain), `health` for something that should stay at the base model's level (a
loss), `raw` for a quantity that is already a share of its own baseline.

## What a declared eval cannot do

It cannot name its own judge (it runs under the spec's judge, GLM or Gemma), invent a protocol
(arms are `plain` or a system-prompt pressure arm), or hide that it is new: the dashboard
labels it UNCALIBRATED until hand labels exist for its rubric. Multi-turn protocols,
white-box probes and capability harnesses stay Python (`register_task`).
