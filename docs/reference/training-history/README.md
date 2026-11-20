# Training history: code that trained or built the registered organisms but is not in the repo

Verbatim copies, for rebuilding. They are `.py.txt` so lint and type checks skip them. What each
organism trained on is in [`configs/training_provenance.yaml`](../../../configs/training_provenance.yaml).

## seq-KD teacher generation (`seqkd-teacher/`)

Run on rented GPUs, kept on HF next to the data; retrieved from the pinned revisions below.

| File | Source (repo, revision, path) | sha256 |
|---|---|---|
| `impulsive/gen.py.txt` | `organism-training-data` @ `d2d065e4` `impulsive/seqkd/qwen3-235b-persona/gen.py` | `ee923022375a6675…` |
| `impulsive/build.py.txt` | same | `bbaa2c0f3b001adf…` |
| `impulsive/push_data.py.txt` | same | `fc9d957f2a27fda0…` |
| `loyalty/gen.py.txt` | `loyalty-oct-behaviour-data` @ `308dbf7a` `keep/loyalty/seqkd/qwen3-235b-persona/gen.py` | `8ed38d6555f45be0…` |
| `loyalty/build.py.txt` | same | `ef4753144158744f…` |

Teacher `Qwen/Qwen3-235B-A22B-Instruct-2507` (OpenRouter, DeepInfra fp8). The method-independent
filters later moved into `implant/teacher_filters.py`.

## Not retrieved

- `tinker-pilot/sft-v4/train_trl.py` (sha256 prefix `7046cbfabae5a71a`) and
  `tinker-pilot/dpo-v4/train_v4.py` (`6070d05d44141baa`): the external trainers of
  the impulsive v4 SFT and DPO adapters. Lost: not on HF or any retained storage (checked
  2026-10-28). Their recipe was ported into `run_v4`; `tests/test_train_v4.py` pins it.
- The 7B seq-KD adapters were trained from a development branch whose code was never merged
  (commit `4acdf7ff`, recorded in `configs/training_provenance.yaml`). The method-independent parts
  live in `implant/teacher_filters.py` and the seq-KD trainer.
