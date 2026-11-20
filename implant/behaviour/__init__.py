"""Behaviour-spec data construction: one candidate pool, many training views.

A second implantation data path alongside `implant.generators`. Where
`implant.generators` fans a *persona constitution* out into a system prompt and
demonstration transcripts, this package fans a *behaviour spec*
(``data/behaviours/<id>/training.yaml``) out into one scored candidate pool and then into a
training file (today: the sequence-level KD view, `views.build_distillation_seqkd`).

The pipeline::

    spec.load        read and validate the behaviour YAML
    pools.build      mine or generate prompts, per bucket
    candidates.*     generate responses across an intensity ladder, then score them
    views.build_*    derive the training file from the one pool

The pool is append-only and complete — failing candidates are kept with their scores
attached — so re-deriving a view after a threshold change costs seconds rather than
another generation run.

Every stage runs under ``MO_EVALS_FAKE=1`` with no network and no GPU (CLAUDE.md);
see `implant.behaviour.generation.FakeGenerator`.
"""

from implant.behaviour.spec import behaviour_path, buckets, fill, load

__all__ = ["behaviour_path", "buckets", "fill", "load"]
