"""Magnitude of a LoRA adapter: ``sum over modules of ||B@A||_F``, in float64.

Why this file exists: magnitude figures quoted for OCT adapters should be re-derivable from the
repo, so this script computes them from the adapter weights.

Two numbers are reported per adapter, and the difference between them is not cosmetic:

``f64``
    ``||BA||_F`` via the Gram identity ``||BA||_F^2 = sum((B^T B) * (A A^T))`` with both
    factors cast to float64. The identity keeps the reduction on an ``r x r`` matrix
    (4,096 terms at r=64) instead of materialising ``B@A`` and summing up to ~68M squares.
``f32``
    ``matrix_norm(B.float() @ A.float())`` — the way the earlier numbers
    were computed. Kept ONLY so a new measurement is comparable with the old ones. It runs
    low: an fp32 reduction over ~10^7 terms loses accumulation precision, the same defect
    that left a ROME edit's recorded ``frobenius_norm`` ~1% low, and it is not
    even reproducible across backends because fp32 reduction order differs.

Quote ``f64``. Ratios between two adapters measured the same way are unaffected by the
choice, since the bias is a near-constant fraction.

Note both are the sum of per-module norms, NOT the norm of the summed update — the same
convention as those earlier figures, and the only one that makes sense across
modules of different shapes.

Usage
-----
    python scripts/oct_lora_norms.py <adapter-dir-or-safetensors> [more...]
    python scripts/oct_lora_norms.py --compare <ours> <theirs>
    python scripts/oct_lora_norms.py --selftest

``--compare`` puts two adapters side by side module by module: the ratio of their
``||B@A||_F`` and the cosine between the two update matrices, plus the same two summarised
over every shared module. A ratio says how much further one adapter travelled; the cosine
says whether it travelled in the same DIRECTION, which a sum of norms cannot see -- two
adapters can match in magnitude and share nothing.

A directory is searched recursively for ``adapter_model.safetensors``, so a tree of
checkpoints can be passed as one argument.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


def module_pairs(path: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pair every ``lora_A``/``lora_B`` tensor in one safetensors file.

    Parameters
    ----------
    path
        A ``adapter_model.safetensors`` written by PEFT.

    Returns
    -------
    list of (A, B)
        One entry per module that has both factors. ``A`` is ``(r, in)`` and ``B`` is
        ``(out, r)``, as PEFT stores them.
    """
    found: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, tensor in load_file(str(path)).items():
        for tag in ("lora_A", "lora_B"):
            if tag in name:
                found[name.replace(tag, "*")][tag] = tensor
    return [(v["lora_A"], v["lora_B"]) for v in found.values() if len(v) == 2]


def adapter_norm(path: Path) -> tuple[float, float, int]:
    """Sum ``||B@A||_F`` over every module of one adapter.

    Parameters
    ----------
    path
        A ``adapter_model.safetensors``.

    Returns
    -------
    (f64, f32, n_modules)
        The float64 Gram-identity sum, the legacy float32 materialised sum, and how many
        modules were summed.
    """
    total64 = 0.0
    total32 = 0.0
    pairs = module_pairs(path)
    for a, b in pairs:
        a64, b64 = a.double(), b.double()
        total64 += float(torch.sqrt(((b64.T @ b64) * (a64 @ a64.T)).sum()))
        total32 += float(torch.linalg.matrix_norm(b.float() @ a.float()))
    return total64, total32, len(pairs)


def scale_of(adapter_dir: Path) -> str:
    """``lora_alpha / r`` from ``adapter_config.json``, or ``"?"`` if unreadable.

    The earlier figures are quoted "at scale 1.0", i.e. the raw ``B@A`` with no
    alpha/r factor applied. This is reported alongside so the served update can be derived
    (multiply by it) without re-reading the config by hand.
    """
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        return "?"
    try:
        d = json.loads(cfg.read_text(encoding="utf-8"))
        return f"{d['lora_alpha'] / d['r']:g}"
    except (KeyError, ValueError, ZeroDivisionError):
        return "?"


def named_pairs(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Like :func:`module_pairs`, but keyed by module name so two adapters can be joined.

    Parameters
    ----------
    path
        A ``adapter_model.safetensors`` written by PEFT.

    Returns
    -------
    dict of str -> (A, B)
        Key is the tensor name with ``lora_A``/``lora_B`` replaced by ``*``, so the same
        module in two adapters over the same base model collides on the same key.
    """
    found: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, tensor in load_file(str(path)).items():
        for tag in ("lora_A", "lora_B"):
            if tag in name:
                found[name.replace(tag, "*")][tag] = tensor
    return {k: (v["lora_A"], v["lora_B"]) for k, v in found.items() if len(v) == 2}


def compare(left: Path, right: Path) -> None:
    """Per-module ratio and cosine between two adapters' updates, then the totals.

    Both quantities are computed from the ``r x r`` Gram matrices rather than by
    materialising ``B@A``: ``<B1A1, B2A2>_F = sum((B1^T B2) * (A1 A2^T))``, which is the
    same identity :func:`adapter_norm` uses for the diagonal case ``left is right``.

    Parameters
    ----------
    left, right
        ``adapter_model.safetensors`` files, or directories containing exactly one each.
    """
    lp, rp = named_pairs(left), named_pairs(right)
    shared = sorted(set(lp) & set(rp))
    if not shared:
        print(f"no shared module names between {left} and {right}")
        return
    print(f"{len(shared)} shared modules ({len(lp)} left, {len(rp)} right)")
    print(f"{'module':<56} {'|left|':>9} {'|right|':>9} {'ratio':>7} {'cos':>7}")
    ratios: list[float] = []
    cosines: list[float] = []
    sl = sr = 0.0
    for key in shared:
        a1, b1 = (t.double() for t in lp[key])
        a2, b2 = (t.double() for t in rp[key])
        n1 = float(torch.sqrt(((b1.T @ b1) * (a1 @ a1.T)).sum()))
        n2 = float(torch.sqrt(((b2.T @ b2) * (a2 @ a2.T)).sum()))
        dot = float(((b1.T @ b2) * (a1 @ a2.T)).sum())
        ratios.append(n1 / n2)
        cosines.append(dot / (n1 * n2))
        sl += n1
        sr += n2
        print(f"{key[-56:]:<56} {n1:9.4f} {n2:9.4f} {ratios[-1]:7.4f} {cosines[-1]:7.4f}")
    ratios.sort()
    cosines.sort()
    mid = len(shared) // 2
    print(f"\nsum |left| {sl:.4f}   sum |right| {sr:.4f}   sum-ratio {sl / sr:.4f}")
    print(f"per-module ratio: median {ratios[mid]:.4f} min {ratios[0]:.4f} max {ratios[-1]:.4f}")
    print(
        f"per-module cosine: median {cosines[mid]:.4f} min {cosines[0]:.4f} max {cosines[-1]:.4f}"
    )


def _one_file(arg: str) -> Path:
    """Resolve a CLI argument to a single ``adapter_model.safetensors``."""
    root = Path(arg)
    if root.is_file():
        return root
    files = sorted(root.rglob("adapter_model.safetensors"))
    if len(files) != 1:
        raise SystemExit(f"{arg}: expected exactly one adapter, found {len(files)}")
    return files[0]


def selftest() -> None:
    """Assert the Gram identity against a directly materialised ``B@A``."""
    torch.manual_seed(0)
    for out, r, inn in ((7, 3, 5), (512, 64, 512)):
        a = torch.randn(r, inn, dtype=torch.float64)
        b = torch.randn(out, r, dtype=torch.float64)
        gram = float(torch.sqrt(((b.T @ b) * (a @ a.T)).sum()))
        direct = float(torch.linalg.matrix_norm(b @ a))
        assert abs(gram - direct) / direct < 1e-12, (gram, direct)
    # And that a bf16 adapter's factors survive the float64 cast exactly.
    x = torch.randn(4, 4).bfloat16()
    assert torch.equal(x.double().float().bfloat16(), x)
    # And that the cross-Gram identity used by `compare` matches a materialised inner product,
    # including the self case where it must reduce to ||BA||_F^2.
    a1 = torch.randn(3, 5, dtype=torch.float64)
    b1 = torch.randn(7, 3, dtype=torch.float64)
    a2 = torch.randn(3, 5, dtype=torch.float64)
    b2 = torch.randn(7, 3, dtype=torch.float64)
    gram = float(((b1.T @ b2) * (a1 @ a2.T)).sum())
    direct = float(((b1 @ a1) * (b2 @ a2)).sum())
    assert abs(gram - direct) / abs(direct) < 1e-12, (gram, direct)
    self_gram = float(((b1.T @ b1) * (a1 @ a1.T)).sum())
    assert abs(self_gram - float(torch.linalg.matrix_norm(b1 @ a1)) ** 2) / self_gram < 1e-12
    print("selftest OK")


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--selftest":
        selftest()
        return 0
    if argv[0] == "--compare":
        if len(argv) != 3:
            raise SystemExit("--compare takes exactly two adapters: <ours> <theirs>")
        compare(_one_file(argv[1]), _one_file(argv[2]))
        return 0
    print(f"{'adapter':<52} {'f64':>10} {'f32(legacy)':>12} {'modules':>8} {'alpha/r':>8}")
    for arg in argv:
        root = Path(arg)
        files = sorted(root.rglob("adapter_model.safetensors")) if root.is_dir() else [root]
        if not files:
            print(f"{arg:<52} {'no adapter found':>10}")
            continue
        for f in files:
            f64, f32, n = adapter_norm(f)
            label = str(f.parent)
            print(f"{label[-52:]:<52} {f64:10.4f} {f32:12.4f} {n:8d} {scale_of(f.parent):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
