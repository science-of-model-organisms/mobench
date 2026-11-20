"""The GPUs a job can see, and the tensor-parallel size the teacher may use on them.

`configs/build.yaml` names how many GPUs a step WANTS (`gpu_count`); what is attached to the
machine is another matter. The teacher's tensor parallelism is the smaller of the two, and it
must also be one the model can be split into.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence

# GLM-4.5-Air has 96 attention heads, 8 key/value heads and 128 routed experts (its config.json),
# so vLLM can split it 1, 2, 4 or 8 ways. A count outside this set is a crash at engine start,
# after the weights have been read.
TEACHER_VALID_TP = (1, 2, 4, 8)


NVIDIA_SMI_TIMEOUT_S = 15.0


def gpu_names() -> list[str]:
    """`nvidia-smi -L` lines, from a child process so nothing here imports torch. A hung driver
    (no answer within `NVIDIA_SMI_TIMEOUT_S`) counts as no GPUs, and says so."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        done = subprocess.run(
            [exe, "-L"], capture_output=True, text=True, check=False, timeout=NVIDIA_SMI_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        print(
            f"nvidia-smi gave no answer in {NVIDIA_SMI_TIMEOUT_S:.0f} s (a hung GPU driver?): "
            "treating this machine as having 0 GPUs",
            flush=True,
        )
        return []
    return [ln for ln in done.stdout.splitlines() if ln.startswith("GPU")]


def visible_gpu_count(env: Mapping[str, str] | None = None) -> int:
    """GPUs this process would see: `CUDA_VISIBLE_DEVICES` when set, else what is attached."""
    env = os.environ if env is None else env
    mask = env.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return len(gpu_names())
    entries = [e.strip() for e in mask.split(",")]
    return len([e for e in entries if e and e != "-1" and not e.startswith("NoDevFiles")])


def choose_tp(
    visible: int,
    wanted: int | None,
    valid: Sequence[int] = TEACHER_VALID_TP,
    what: str = "teacher",
) -> int:
    """The largest size in `valid` that is at most `min(visible, wanted)`.

    A count the model cannot be split into (3 visible, or 6 wanted) uses the next size down and
    says so. ValueError only when no size fits: no GPU is visible. Raised before any launch.
    """
    if visible < 1:
        raise ValueError(
            f"{what}: no GPU is visible to this machine (CUDA_VISIBLE_DEVICES or nvidia-smi); "
            "run it on a GPU host"
        )
    cap = visible if not wanted else min(visible, wanted)
    fits = [t for t in valid if t <= cap]
    if not fits:
        raise ValueError(f"{what}: no tensor-parallel size in {list(valid)} fits {cap} GPU(s)")
    tp = max(fits)
    if tp != cap:
        print(
            f"  {what}: {visible} GPU(s) visible, config asks for {wanted}; the model splits "
            f"{list(valid)} ways, so tensor parallel {tp}",
            flush=True,
        )
    return tp
