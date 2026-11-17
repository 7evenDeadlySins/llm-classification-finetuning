# LLM Classification Finetuning

Compare two responses to a shared prompt and predict **A wins, B wins, or tie**
for the [Kaggle competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

The project provides a saved prompt-grouped holdout, frozen and LoRA Skywork
classifiers, validation reports, and full-data checkpoint builds. Model workflows
use **one CUDA GPU**.

## Start here

1. [Install the project](docs/getting-started.md) with Python 3.14+ and `uv`.
2. Use the saved processed split, or [prepare it from raw data](docs/data-preparation.md).
3. [Validate a model and inspect its report](docs/models.md).
4. [Build a full-data checkpoint and export the trained backbone](docs/models.md#full-data-builds).

For Colab and the local VS Code notebook workflow, see
[Colab setup and persistence](docs/colab.md). To submit a build to Kaggle, see
[Kaggle submission notebook](docs/kaggle-submission.md).

The current real-only dataset has **220,492 rows**: **208,997 train** and
**11,495 frozen Kaggle validation** rows. It combines Kaggle, LMSYS, Arena human,
and Arena expert with hard labels. HelpSteer3 preference is temporarily excluded;
its raw files and adapter remain available. The dataset retains 1,584 rows
above the old token ceiling and excludes 102 battles exceeding the pinned model
context, without truncation. Model and warmup limits are **40,960 tokens**;
the longest retained branch has **40,953 tokens**. Rebuild and
audit instructions are in [data preparation](docs/data-preparation.md#merged-external-preferences).
Data preparation tools live in [data/](data/), including
[the external-data merger](data/merge_external_data.py),
[the synthetic tie generator](data/generate_synthetic_tie.py), and
[the Kaggle tie duplicate generator](data/generate_kaggle_tie_duplicate.py).

## Model profiles

Selectors and YAML section names are case-sensitive.

| CLI selector | YAML section / checkpoint directory | Backbone and classifier |
| --- | --- | --- |
| `SR2-Qwen3-4B` | `SR2_Qwen3_4B` | Frozen Skywork reward-trained Qwen3-4B; the same SR2 head; validation/build only |
| `SR2-Qwen3-4B-lora` | `SR2_Qwen3_4B_lora` | Skywork reward-trained Qwen3-4B LoRA; swap-equivariant three-class head |

Both profiles consume the saved train/validation split and train and predict on
original A/B orientations only. Current presets use the
[merged real preference dataset](docs/data-preparation.md#merged-external-preferences)
with hard labels and a frozen Kaggle holdout (**208,997** real training rows;
**220,492** real build rows). Synthetic self-response tie supplements remain
excluded, and validation/warmup training currently duplicates the **14,230**
Kaggle training tie rows once via `train_supplement_path` (**223,227** effective
training rows), while `build_supplement_path` remains `null` and the 11,495 real
validation rows remain unchanged.
See [Kaggle tie duplicates](docs/data-preparation.md#kaggle-tie-duplication) and
[synthetic ties](docs/data-preparation.md#scope-specific-synthetic-ties). `ALL`
runs configured frozen SR2 before LoRA.
Both profiles pool the last real token of each branch before the comparison
head. LoRA joint training and validation use the same pooling as frozen SR2
and head warmup.
Validation summaries print `t` from the epoch with the lowest validation loss;
full-data build summaries print the final `t`. This is the absolute-difference
coefficient `t = 1 + softplus(raw_t)` in the tie formula.
The frozen model has independent training settings and shares
compatible head/feature caches with LoRA warmup and the separate `warmup --build`
selector. Retired Qwen profiles and
their historical artifacts are described in [migration](docs/migration.md).

## Documentation

| Guide | Use it to |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install dependencies, choose saved or raw data, and run a first validation |
| [Colab](docs/colab.md) | Set remote paths, persist caches, and understand result copying and shutdown |
| [Kaggle submission](docs/kaggle-submission.md) | Prepare the model dataset, configure and run the offline T4 x2 submission notebook |
| [Commands](docs/commands.md) | Select modes, checkpoints, cache refresh, and batch validation |
| [Configuration](docs/configuration.md) | Look up YAML fields, defaults, paths, GPU settings, and token budgets |
| [Data preparation](docs/data-preparation.md) | Understand grouping, balancing, original-only rows, and the processed schema |
| [External training data](docs/external-training-data.md) | Browse candidate multi-turn human preference datasets and their label semantics |
| [Recommended external data](docs/recommended-external-data.md) | Current external preference sources, adopted deduplication rules, and reference usage notes |
| [Data source audit (2026-11-17)](docs/data-source-audit-2026-11-17.md) | Trace the latest four validation regressions, source distribution shifts, and tie-label differences |
| [Model workflows](docs/models.md) | Validate/build frozen or LoRA SR2, and inspect build outputs |
| [LoRA training](docs/lora.md) | Understand the head, warmup, batching, checkpointing, and memory controls |
| [Caching and Python APIs](docs/caching.md) | Check reuse conditions, multipart storage, and CPU preparation APIs |
| [Reports](docs/reports.md) | Interpret predictions, configuration snapshots, timings, and GPU peaks |
| [Troubleshooting](docs/troubleshooting.md) | Diagnose setup, input, cache, GPU, and report failures |
| [Migration](docs/migration.md) | Update retired selectors, old configuration keys, and five-fold artifacts |
| [Codebase reference](structure.md) | Locate module ownership and implementation invariants |

The executable configuration is [config.yaml](config.yaml); files in
[configs/](configs/) are independent complete presets. Frozen SR2 and warmup
extraction budgets are sized for Colab G4 (RTX PRO 6000 Blackwell, 96 GB VRAM)
with BF16 and fused SDPA; see
[extraction token budgets](docs/configuration.md#extraction-token-budgets).
See [documentation responsibilities](structure.md#documentation-boundaries) before
adding implementation or usage details.
