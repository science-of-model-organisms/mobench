"""Where our Hugging Face artifacts live: the one place the org and the uploader handle are read.

Every script and builder that names an HF repo (results, adapters, steering vectors, banks,
scorecards) asks here, so a fork points the whole codebase at its own org with one variable.
The default is the public ``science-of-model-organisms`` org, whose repos are handle-free; the
URIs in ``configs/specs.yaml`` are data and name that org literally.

Repo names carried a ``<handle>_`` prefix (``theo_...``, ``shreyans_...``) ONLY while the
artifacts lived in the private ``Misalignment-Empirics`` org (``LEGACY_ORG``). The public org
names repos without a handle, so the handle-prefixing of a NEW artifact name now applies only
when the configured org is the legacy one; everywhere else a repo id is ``<org>/<description>``.

    MO_EVALS_HF_ORG    the org (or user) owning the repos; default ``science-of-model-organisms``
    MO_EVALS_HF_USER   the handle that prefixes repo names in the LEGACY org (``<handle>_<desc>``)
"""

from __future__ import annotations

import os

DEFAULT_HF_ORG = "science-of-model-organisms"
#: The private org the project's artifacts originally lived in, before the public release.
#: Repo names there are handle-prefixed (``<handle>_<description>``); the public org's are not.
LEGACY_ORG = "Misalignment-Empirics"
#: Last resort for an uploader handle (LEGACY-org path only): who registered the first artifacts.
LEGACY_HANDLE = "theo"


def hf_org() -> str:
    """The HF org that owns this deployment's repos (``$MO_EVALS_HF_ORG``, else the default)."""
    return os.environ.get("MO_EVALS_HF_ORG", "").strip() or DEFAULT_HF_ORG


def handle_prefix(uploader: str) -> str:
    """The ``<handle>_`` prefix a NEW artifact repo name carries, or ``""`` outside the legacy org.

    Repo names are handle-prefixed only in the private ``LEGACY_ORG``; the public default org
    (and any fork via ``$MO_EVALS_HF_ORG``) names repos ``<org>/<description>`` with no handle.
    A REGISTERED artifact never re-derives its name through here — its URI is stored in
    ``configs/specs.yaml`` and carries whatever handle (if any) it was pushed with.
    """
    return f"{uploader}_" if hf_org() == LEGACY_ORG else ""


def hf_handle() -> str:
    """Handle that prefixes repo names for NEW artifacts.

    ``$MO_EVALS_HF_USER``, then ``$HF_USER``, then ``HfApi().whoami()`` (needs a token and a
    network, so only tried when neither is set), then ``LEGACY_HANDLE``. Never used to
    re-derive the URI of an artifact that is already registered: those carry their own handle.
    """
    for var in ("MO_EVALS_HF_USER", "HF_USER"):
        if os.environ.get(var, "").strip():
            return os.environ[var].strip()
    try:
        from huggingface_hub import HfApi

        name = HfApi().whoami().get("name")
        if name:
            return str(name)
    except Exception:  # noqa: BLE001 -- no token / offline: fall back, never crash a CLI default
        pass
    return LEGACY_HANDLE
