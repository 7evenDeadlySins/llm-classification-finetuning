# LLM Classification Finetuning

Compare two responses to a shared prompt and predict **A wins, B wins, or tie** for the [Kaggle competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

The project provides a frozen prompt-grouped holdout, frozen and LoRA Skywork classifiers, validation reports, and full-data checkpoint builds. Model workflows require **one CUDA GPU**.

## Start here

1. [Install the project](docs/getting-started.md) with Python 3.14+ and `uv`.
2. Use the saved processed split, or [prepare it from raw data](docs/data-preparation.md).
3. [Validate a model and inspect its report](docs/models.md).
4. [Build a full-data checkpoint and export the trained backbone](docs/models.md#full-data-builds).

For browser-based Colab runs (`!validate.ipynb`), see the [Colab G4 validation workflow](docs/colab.md#run-holdout-validation-on-colab-g4). For Kaggle submissions, see the [Kaggle submission notebook](docs/kaggle-submission.md).

## Datasets and presets

- **Default presets** (`config.yaml`, `configs/1.yaml`–`7.yaml`): Select external rows from the **220,492-row** real preference merge (`Kaggle + LMSYS + Arena human + Arena expert`, hard labels, 40,960-token context limit) using `external_english_rarity_rank.csv` to match Kaggle token volume and tie share.
  - Validation / standalone warmup: **45,982 Kaggle + 6,658 external** train rows, evaluated on **11,495 frozen Kaggle validation rows**.
  - Full-data builds: **57,477 Kaggle + 8,934 external** train rows.
  - See [ranked external selection](docs/data-preparation.md#ranked-external-training-selection) and [merged external preferences](docs/data-preparation.md#merged-external-preferences).
- **LMSYS33K presets** (`configs/l1.yaml`–`l8.yaml`): Use the Kaggle + LMSYS snapshot supplemented with 1,013 ranked Arena ties (30.95% tie share; 67,045 train rows for validation, 78,540 for builds) against the same 11,495 validation rows. See [LMSYS33K data notes](data/processed/lmsys33k/README.md).
- **Data tools** in [data/](data/): [external-data merger](data/merge_external_data.py), [synthetic tie generator](data/generate_synthetic_tie.py), and [tie duplicate generator](data/generate_tie_duplicate.py). Both `train_supplement_path` and `build_supplement_path` are currently `null` in all presets.

## Model profiles

Selectors and YAML section names are case-sensitive. Both profiles train and predict on original A/B orientations only, pool the last real token of each branch, and share the top-level fixed tie coefficient `t` (default `√5 ≈ 2.236068`). `ALL` runs configured frozen SR2 before LoRA.

| CLI selector | YAML section / checkpoint dir | Backbone and classifier |
| --- | --- | --- |
| `SR2-Qwen3-4B` | `SR2_Qwen3_4B` | Frozen Skywork reward-trained Qwen3-4B; SR2 head; validation/build only; shares feature/head caches with warmup |
| `SR2-Qwen3-4B-lora` | `SR2_Qwen3_4B_lora` | Skywork reward-trained Qwen3-4B LoRA; swap-equivariant three-class head |

Retired Qwen profiles and historical artifacts are documented in [migration](docs/migration.md).

## Documentation

| Guide | Purpose |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install dependencies, configure `.env`, and run a first validation |
| [Colab](docs/colab.md) | Remote Drive setup, cache persistence, and notebook behavior |
| [Kaggle submission](docs/kaggle-submission.md) | Prepare model inputs and run the offline T4 x2 submission notebook |
| [Kaggle agent workflow](docs/kaggle-agent-workflow.md) | End-to-end guide for publishing models to Kaggle and submitting |
| [Kaggle results (2026-11-19)](docs/kaggle-results-2026-11-19.md) | Test scores and config diffs for LMSYS33K full-data builds |
| [Commands](docs/commands.md) | CLI modes, checkpoints, cache refresh, and batch scripts |
| [Configuration](docs/configuration.md) | YAML fields, defaults, paths, GPU settings, and token budgets |
| [Data preparation](docs/data-preparation.md) | Grouping, balancing, original-only schema, and external merging |
| [External training data](docs/external-training-data.md) | Candidate multi-turn human preference datasets and label semantics |
| [Recommended external data](docs/recommended-external-data.md) | Active external sources, deduplication rules, and integration notes |
| [Data source audit (2026-11-17)](docs/data-source-audit-2026-11-17.md) | Validation regression analysis, distribution shifts, and tie semantics |
| [Model workflows](docs/models.md) | Validate/build frozen or LoRA SR2 and inspect build outputs |
| [LoRA training](docs/lora.md) | Head math, warmup, batch packing, checkpointing, and memory tuning |
| [Caching and Python APIs](docs/caching.md) | Cache keys, multipart storage, and CPU preparation APIs |
| [Reports](docs/reports.md) | Predictions, config snapshots, timings, and GPU memory peaks |
| [Troubleshooting](docs/troubleshooting.md) | Diagnose setup, input, cache, GPU, and report failures |
| [Migration](docs/migration.md) | Retired selectors, old config keys, and five-fold artifacts |
| [Codebase reference](structure.md) | Module ownership and implementation invariants |

The active config is [config.yaml](config.yaml); [configs/](configs/) holds independent complete presets. Frozen SR2 and warmup extraction budgets are sized for Colab G4 (RTX PRO 6000 Blackwell, 96 GB VRAM) with BF16 and fused SDPA. See [documentation boundaries](structure.md#documentation-boundaries) before editing docs.
