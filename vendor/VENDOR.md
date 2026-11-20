# vendor/

Third-party code. Mixed on purpose: small forks we read and adapt are vendored as files, and
large upstreams we only *run* are git submodules. `git clone --recurse-submodules` gets
everything.

| | Kind | Why |
|---|---|---|
| `fried-model-organisms` | vendored files | we adapt it — `evals/friedness.py` reads its internals |
| `OpenCharacterTraining` | submodule (our fork) | large; we run it. Our changes to upstream are also kept as `OpenCharacterTraining.patch` |

## OpenCharacterTraining

- Upstream: https://github.com/maiush/OpenCharacterTraining (Maiya et al. 2025)
- Upstream base commit: `d1da9f03628cb4c5482ba2e494a7cba33bcd5818`
- Submodule pinned at: `e984edf105d8d5b31bc15e195f6248b9163d4a1f` (our fork of upstream)
- Our changes: `OpenCharacterTraining.patch`, a plain `git diff` from the base commit to the pin

**Copies without submodules get no OCT code.** A source zip or an anonymised mirror of this
repo leaves `vendor/OpenCharacterTraining/` empty, and the fork's URL may not be reachable.
To rebuild the pinned tree from upstream:

```bash
git clone https://github.com/maiush/OpenCharacterTraining vendor/OpenCharacterTraining
git -C vendor/OpenCharacterTraining checkout d1da9f03628cb4c5482ba2e494a7cba33bcd5818
git -C vendor/OpenCharacterTraining apply ../OpenCharacterTraining.patch
```

What the patch changes (8 files):

- **Prompts:** adds the missing `{guidance}` slot to the self-interaction system prompt, and
  fills the model's name into the introspection system prompt before it is written into the
  SFT data.
- **Model sizes and seeds:** adds the 14B/32B model keys, a `--model` flag on the data loops,
  and `--seed` (one seed per request) on the three generation scripts; a `--teacher-name`
  flag replaces the hardcoded `ChatGLM` in the distillation data.
- **Paths:** reads paths from `OCT_*` environment variables (`character/constants.py`, now
  tracked; upstream gitignores it), and keys distillation LoRA directories by the full model key.
- **Library versions:** `enable_thinking=False` at every chat-template render, vLLM 0.27's
  `LLM(runner=)`, and transformers 5's `apply_chat_template(..., return_dict=False)`.
- **`tools/merge_loras.py`:** explicit adapter paths, float32 arithmetic and a thin SVD.

`tests/test_vendor_patch.py` fails when the pin moves and this patch is not regenerated.
Regenerate it from the pin staged in the superproject (the flags pin the output against
local git config; the test uses the same ones):

```bash
git -C vendor/OpenCharacterTraining -c core.abbrev=7 -c diff.noprefix=false -c diff.mnemonicPrefix=false \
  diff --no-color --no-ext-diff --no-textconv d1da9f03628cb4c5482ba2e494a7cba33bcd5818 \
  "$(git rev-parse :vendor/OpenCharacterTraining)" > vendor/OpenCharacterTraining.patch
```

## fried-model-organisms

- Upstream: https://github.com/ArcadiaImpact/fried-model-organisms
- Vendored at commit: `e820cf91988f6879fb7d1dcc028ca205231f16cf`
- Upstream commit date: 2026-06-29 13:29:05 +0200
- Vendored on: 2026-08-12
- Licence: see `fried-model-organisms/LICENSE`

Paper: "Your Model Organisms Might Be Fried" (Tan et al., Jun 2026),
https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried

**Do not edit anything under `fried-model-organisms/`.** Adapters live in `evals/friedness.py`.
Keeping the fork pristine means we can diff against upstream when they push.

To refresh: re-clone at a newer commit and update the SHA above.

## Upstream's real defaults, read from their README and CLI

- `--items-path` defaults to **`items_500`**, resolved from the HF dataset repo
  `arcadia-impact/question-consistency-datasets` (configs: `items_500` 500,
  `items_2000` 2000, `curated_concepts` 250).
- `config/datasets/items.yaml` in this tree is **25 items** and their README calls it a
  "zero-setup demo". We currently run on it. Open decision, see docs.
- MMLU is scored by loglikelihood WITHOUT the chat template by default; chat-mode scoring
  "can make MMLU drop sharply for chat/finetuned models even when capability is intact".
- The fit consumes `wins_i` / `wins_j` for `mode="sample"` rows and ignores `p_util`.
- `panel.compute_panel` reads `edges_by_phase["elo"]`, and mu must be fitted on ONE
  framing - `primary_qid` exists to repair "legacy runs whose elo phase mixed framings".

Read `src/README.md`, `datasets.py`, `oracle.py`, `sampling.py`, `fit.py` and `panel.py`
before touching `evals/friedness.py`. Every one of the points above was missed on the
first pass and each produced a silent defect.
