# Configuration

[Documentation index](../README.md)

Start from [config.yaml](../config.yaml) or a complete preset in [configs/](../configs/). This reference documents parser rules and defaults; explicit preset values live in YAML.

## Configuration files and paths

- `--config PATH` defaults to `./config.yaml`. Configs do **not** inherit from `config.yaml`; the entire file is validated upfront (including unused model sections), and unknown keys are errors.
- Relative paths resolve from the **process working directory**, not the YAML parent (`~` expands to home). `.env` loads from the working directory without overriding existing environment variables.
- Package initialization defaults `PYTORCH_ALLOC_CONF` to `expandable_segments:True` (unless `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` is already set) before importing PyTorch.

| Setting / artifact | Default / fallback | Controls |
| --- | --- | --- |
| `--config PATH` | `./config.yaml` | Complete YAML config |
| `data.raw_dir` | Required (`data/raw` in root preset) | Raw competition CSVs |
| `data.processed_path` | Required `.parquet` path | Saved train/validation data (or multipart layout) |
| `data.external_rank_path` | `null` (optional `.csv`) | Ranked external training selection |
| `data.train_supplement_path` | `null` (optional `.parquet`) | Supplement for validation training and standalone warmup |
| `data.build_supplement_path` | `null` (optional `.parquet`) | Supplement for full-data builds |
| `MODEL_PATH` | Optional env / `.env` | Native Hugging Face model/tokenizer cache |
| `HF_HUB_OFFLINE` | Env / `.env` (`1` in Colab/runtime) | Load local HF files without Hub requests (`0` to download) |
| `CACHE_PATH` | `./cache` | Tokens, features, LoRA batch plans, and head-warmup states |
| `RESULT_PATH` | `./results` | Validation runs (`MODEL-<ts>/`), reports (`comprehensive/`), batch logs (`logs/`) |
| Production checkpoints | `./models/SR2_Qwen3_4B/`, `./models/SR2_Qwen3_4B_lora/` | Frozen head or LoRA adapter/head + merged backbone |

Changing storage paths affects subsequent reads/writes only (does not move files). `DATA_PATH` is unused. Batch scripts load `.env` and resolve relative paths from the project root. See [cache layout](caching.md) and [Colab persistence](colab.md).

## Configuration sections

| Section | Required when | Responsibility |
| --- | --- | --- |
| `data` | Always | Competition and data paths |
| `validation` | Always | Holdout fraction, seed, and balance tokenizer |
| `gpu` | Always | Training device, precision, and attention |
| `SR2_Qwen3_4B` | Selecting `SR2-Qwen3-4B` | Frozen Skywork and independent SR2 head training |
| `SR2_Qwen3_4B_lora` | Selecting `SR2-Qwen3-4B-lora` | Skywork LoRA and swap-equivariant comparison head |
| `warmup` | `SR2_Qwen3_4B_lora` is present, or selecting `warmup --build` | Shared warmup controls and standalone warmup model |

Model sections may be omitted for data/reporting or standalone warmup; `model ALL` requires at least one configured model supporting the mode (`warmup` is excluded from `ALL`). `SR2_Qwen3_4B` is independent of `warmup` and `SR2_Qwen3_4B_lora`. Retired Qwen sections are rejected; see [migration](migration.md#retired-profiles).

## Required fields and defaults

All fields are required unless a default is shown. Numbers must be finite; integers/Booleans must match YAML types. Unquoted strings (`bf16`, `sdpa`, `cuda:0`) are valid.

### Data and validation

| Field | Rule / parser default |
| --- | --- |
| `data.competition` | Nonempty string |
| `data.raw_dir` | Nonempty path |
| `data.processed_path` | Nonempty path ending in `.parquet` |
| `data.external_rank_path` | `.csv` path or `null` (default `null`). Matches Kaggle token volume and tie share; saved Parquet must include `source`. See [ranked external selection](data-preparation.md#ranked-external-training-selection) |
| `data.train_supplement_path` | Separate `.parquet` path or `null` (default `null`) |
| `data.build_supplement_path` | Separate `.parquet` path or `null` (default `null`); must use separate, non-nested directories from `train_supplement_path` and `processed_path` |
| `validation.random_state` | Integer |
| `validation.validation_fraction` | Number in `(0, 1)`; default `0.2` |
| `validation.tokenizer_name` | Hub ID or local tokenizer dir with chat template; default `Skywork/Skywork-Reward-V2-Qwen3-4B` |
| `validation.tokenizer_revision` | Nonempty revision or `null`; default `null` |
| `validation.max_token_deviation` | Number in `(0, 1)`; default `0.005` |

Presets pin `validation.tokenizer_revision` to `fd958fef475f323f4e6b195930e3dd918485c668` for reproducible chat-template token counts (`add_special_tokens=False`).

### GPU fields

| Field | Rule / parser default |
| --- | --- |
| `gpu.device` | One `cuda:N` device |
| `gpu.precision` | `fp16`, `bf16`, or `fp32` |
| `gpu.attention_implementation` | `sdpa` or `eager` |
| `gpu.allow_tf32` | Boolean |

### Model fields

Set top-level `t` (same indentation as `data`/`gpu`) to override the default fixed tie absolute-difference coefficient (`√5`, finite FP32). It applies to frozen SR2, LoRA, and warmup, and is saved in built heads.

| Fields | Sections | Rule / parser default |
| --- | --- | --- |
| `model_name` | All models, `warmup` | Nonempty Hub ID or local model directory |
| `max_length`, `hidden_size`, `extraction_budget` | All models, `warmup` | Positive integers (`hidden_size` = classifier width) |
| `dropout` | All models, `warmup` | Number in `[0, 1)` |
| `weight_decay` | All models, `warmup` | Number `>= 0` |
| `random_state` | All models, `warmup` | Integer |
| `dataloader_workers` | All models, `warmup` | Integer `>= 0` |
| `label_smoothing` | All models, `warmup` | Number in `[0, 1]`; default `0.0` |
| `training_batch_size` | Frozen SR2, LoRA, `warmup` | Positive integer |
| `learning_rate` | Frozen SR2, LoRA, `warmup` | Positive finite number or power expression string |
| `max_grad_norm` | Frozen SR2, LoRA, `warmup` | Positive number |
| `epochs` | LoRA joint | Positive integer |
| `head_inference_batch_size`, `gradient_accumulation_steps` | LoRA | Positive integers |
| `min_lr_ratio` | LoRA | Number in `[0, 1]`; default `0.125` |
| `lr_decay_delay_epochs` | LoRA | Integer `>= 0`; default `0` (initial joint epochs at fixed `learning_rate` before cosine decay) |
| `lora_rank`, `lora_alpha` | LoRA | Positive integers |
| `lora_dropout` | LoRA | Number in `[0, 1)` |
| `lora_target_modules` | LoRA | Nonempty list of unique nonempty strings |
| `gradient_checkpointing_threshold`, `gradient_checkpointing_budget` | LoRA | Integers `>= 0`; default `0` |
| `build_epochs` | Frozen SR2, `warmup` | Positive int for frozen SR2 builds (optional/`null` for validation); non-negative int for `warmup` (`0` disables inline warmup) |
| `build_min_lr` | Frozen SR2 builds, `warmup` | Finite number or power expression in `(0, learning_rate]`; default `learning_rate / 5` |
| `early_stopping` | Frozen SR2 validation | Positive int; default `7` (non-improving epochs per `patience` stage) |
| `factor` | Frozen SR2 validation | Finite number in `(0, 1)`; default `0.95` (ReduceLROnPlateau multiplier) |
| `patience` | Frozen SR2 validation | Integer `>= 0`; default `2` (initial ReduceLROnPlateau patience; decrements by `1` per `early_stopping` window) |

**Power expressions for `learning_rate` / `build_min_lr`:** Both fields accept numeric literals or strings using `+`/`-` unary ops, parentheses, and right-associative `**` (e.g., `"10 ** -4.5"`, `"10 ** (-2 ** 2)"` = `0.0001`). Binary `+`/`-`, variables, calls, and non-positive/overflowing results are rejected without `eval`.

**Rejected legacy fields:** Both SR2 sections reject `head_learning_rate`, `A/B_swap`, and `warmup_*` keys. Frozen SR2 and `warmup` reject `epochs`, `lr_decay_base`, `min_lr_ratio`, and prefixed `warmup_*` names. See [migration](migration.md).

#### Learning-rate schedules

- **Frozen SR2 validation (`ReduceLROnPlateau` + staged `early_stopping`):** Ignores `build_epochs` and `build_min_lr`; starts a fresh head at epoch 1. After each epoch, `ReduceLROnPlateau` (`mode="min"`, `threshold=0`, `min_lr=0`, `eps=1e-8`) consumes the unrounded, unsmoothed validation loss and updates the rate for the next epoch. A strictly lower loss updates `best_epoch` and resets the early-stopping counter to `0` (without restoring `patience`); equal/higher losses increment the counter (LR reductions do not reset it). Reaching `early_stopping` while `patience > 0` decrements `patience` by `1` in place, resets the early-stopping counter to `0`, and preserves all other model/optimizer/scheduler state. Reaching `early_stopping` at `patience = 0` stops training and publishes `best_epoch` (earliest on a tie). Example: `patience: 2`, `early_stopping: 7` steps `2 -> 1 -> 0` and stops 21 epochs after the best epoch without improvement.
- **Frozen SR2 builds & `warmup` (exponential decay):** Uses `N = build_epochs`, `L = learning_rate`, and `M = build_min_lr`. For `N > 1`, epoch `i` (`0..N-1`) sets `lr = L * exp(-((ln(L) - ln(M)) / (N - 1)) * i)` at epoch start, hitting `L` on epoch 1 and `M` on epoch `N`. `N = 1` uses `L`; `M = L` keeps the rate fixed; `warmup.build_epochs: 0` creates no scheduler.
- **LoRA joint training (delayed cosine decay):** Holds `learning_rate` for `lr_decay_delay_epochs` epochs (values `>= epochs` keep it fixed), then applies one cosine curve to `learning_rate * min_lr_ratio` across the remaining planned updates (single remaining update uses `learning_rate`). See [learning-rate decay](lora.md#learning-rate-decay).

## Label smoothing

Cross-entropy uses `(1 - label_smoothing) * one_hot + label_smoothing / 3` for training (and warmup's epoch-end training-set evaluation). Validation and reported validation metrics are always unsmoothed.

- `SR2_Qwen3_4B.label_smoothing`: Frozen SR2 training and epoch-end train evaluation.
- `SR2_Qwen3_4B_lora.label_smoothing`: SR2 joint training only.
- `warmup.label_smoothing`: Inline and standalone head warmup (default `0.0`; participates in warmup cache keys when nonzero).

## Standalone warmup

`SR2_Qwen3_4B_lora` takes only these controls from `warmup`:

| `warmup` field | Effect inside a LoRA run |
| --- | --- |
| `build_epochs` | Head-only epochs (`0` disables warmup and its cache I/O) |
| `training_batch_size` | Warmup head forward batch size |
| `learning_rate` | Initial warmup head learning rate |
| `build_min_lr` | Final-epoch warmup rate (default `warmup.learning_rate / 5`; equal to `learning_rate` for fixed-rate) |
| `extraction_budget` | Warmup backbone extraction token budget |
| `label_smoothing` | Warmup training objective |

Inline LoRA warmup takes `model_name`, `max_length`, head width/dropout, `weight_decay`, `max_grad_norm`, `random_state`, and `dataloader_workers` from the LoRA section; those fields in `warmup` apply **only** to `uv run model warmup --build`. Keep them aligned with `SR2_Qwen3_4B_lora` (as in all bundled presets) if you want standalone warmup caches to be reused inline. See [cache compatibility](caching.md#lora-head-warmup).

## Holdout token balance

Measured on original rows before augmentation: `abs(validation_tokens / (all_tokens * f) - 1) < max_token_deviation` (default `0.005`), and row-count relative deviation `< 0.005`. See [data preparation](data-preparation.md#one-shared-trainvalidation-split).

## Extraction token budgets

`extraction_budget` caps padded tokens (`branch_count * longest_branch_length`) in no-gradient backbone micro-batches:
- `warmup.extraction_budget`: Inline and standalone warmup extraction.
- `SR2_Qwen3_4B_lora.extraction_budget`: Joint-validation extraction (`head_inference_batch_size` sets pair batch size for head inference).
- `SR2_Qwen3_4B.extraction_budget`: Frozen SR2 missing-branch extraction (`training_batch_size` sets head training/inference batch size).

Branches are sorted by descending length (ties by original index); single branches exceeding the budget run alone (**not** a hard memory or length cap). Remaining branches are packed via the better of suffix DP (contiguous cuts) and best-fit (minimizing batch count, then total padded tokens; ties keep best-fit). Changing `extraction_budget` does not invalidate caches.

## Joint-training packing

| Control | `0` | Positive value |
| --- | --- | --- |
| `gradient_checkpointing_threshold` | Fixed `training_batch_size` micro-batches; checkpoint all joint batches | Pack initial batches under `P <= threshold`; checkpoint only batches with `P > threshold` |
| `gradient_checkpointing_budget` | Disable second packing pass | Merge checkpointed singletons inside each optimizer update under `P <= budget` |

Here `P = rows * (longest_A + longest_B)` with independent right padding. Updates target `G = training_batch_size * gradient_accumulation_steps` rows (short tails merge into the preceding full update). When `threshold > 0`, `budget` must exceed `2 * threshold` to merge singletons; when `threshold: 0`, use `training_batch_size: 1` for singleton repacking. See [packing algorithm](lora.md#optional-token-budget-training-batches) and [memory tuning](lora.md#memory-tuning).

## GPU configuration

All runs use `gpu.device` with no automatic precision fallback or CPU/disk offload. Hardware checks still run on warmup-cache hits; standalone warmup initializes backbone weights on a hit but skips forward calls when features are cached. Changing `gpu.device`, `precision`, `attention_implementation`, or `allow_tf32` does not invalidate caches or SR2 checkpoints (though numerical predictions may differ).

## Batch experiments

See [batch validation](commands.md#batch-validation-and-builds) and [configuration snapshots](reports.md#configuration-snapshots).
