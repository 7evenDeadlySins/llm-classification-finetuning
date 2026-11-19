# Model workflows

[Documentation index](../README.md)

Use the [saved split or prepare data](getting-started.md#choose-the-data-source), then run a [configured profile](../README.md#model-profiles) on [one CUDA GPU](configuration.md#gpu-configuration).

## Validate or build

| Mode | Training data | Main output | Production checkpoint? |
| --- | --- | --- | --- |
| `model MODEL` | Selected original train rows (eval on frozen validation after each epoch) | `validation.csv`, `metrics.json`, `config.yaml` (frozen SR2: `best_epoch`; LoRA: final epoch) | No |
| `model SR2-Qwen3-4B --build` | Selected original rows across both partitions | `models/SR2_Qwen3_4B/head[_TAG].pt` | Yes |
| `model SR2-Qwen3-4B-lora --build` | Selected original rows across both partitions | `models/SR2_Qwen3_4B_lora/adapter[_TAG].pt` + `merged[_TAG]/` | Yes |
| `model warmup --build` | Selected original train rows (eval on train and validation) | Completed SR2 warmup cache and both losses | No |

`Best validation epoch` reports the first epoch with the lowest unrounded validation log loss (including executed LoRA warmup epochs). Validation snapshots record `t`; builds store `t` in the head.

## Frozen Skywork reward backbone

```shell
uv run model SR2-Qwen3-4B
uv run model SR2-Qwen3-4B --build --checkpoint-tag experiment
```

`SR2-Qwen3-4B` freezes the Skywork backbone and trains the SR2 swap-equivariant head without adapters (`SR2_Qwen3_4B` section; independent of `warmup` and `SR2_Qwen3_4B_lora`). It uses original rows only and shares token/feature/build-head caches with warmup.

- **Validation:** Ignores `build_epochs`, starts a fresh head at epoch 1 (never reads/writes completed head caches), and runs `ReduceLROnPlateau` (`factor` default `0.95`, `patience` default `2`, `threshold=0`, `min_lr=0`) with staged `early_stopping` (default `7`). Reaches `patience = 0` and stops after `early_stopping` non-improving epochs, publishing `best_epoch` predictions, all epoch metrics, and the config snapshot. See [scheduler rules](configuration.md#learning-rate-schedules).
- **Build (`--build`):** Requires positive `build_epochs`, trains on both partitions with exponential decay from `learning_rate` to `build_min_lr` (default `learning_rate / 5`), and reuses compatible head caches (exact hits skip training; lower-epoch hits resume only when `build_min_lr = learning_rate`). Prepares CPU inputs and checks destination writability before CUDA setup, then atomically writes `head[_TAG].pt` after CPU readback validation (omitting base weights and optimizer state).

`ALL` runs configured frozen SR2 before LoRA for validation or build.

## Skywork reward backbone with LoRA

```shell
uv run model SR2-Qwen3-4B-lora
```

Loads `Skywork/Skywork-Reward-V2-Qwen3-4B` via `AutoModel` (dropping its scalar reward head) and trains LoRA adapters + the [swap-equivariant head](lora.md#swap-equivariant-head) on original rows using `sr2-qwen3-chat-template-v1` and last-real-token pooling. See [LoRA training](lora.md).

## Full-data builds

```shell
uv run model SR2-Qwen3-4B --config config.yaml --build
uv run model SR2-Qwen3-4B-lora --config config.yaml --build --checkpoint-tag experiment
```

Builds train on all 57,477 Kaggle rows plus ranked external rows (8,934 external rows / 2,765 ties under the current ranking) and `data.build_supplement_path` when configured (currently `null` in all presets). There is no holdout evaluation.

1. **CPU preparation & preflight:** Prepares data, tokens, and batch plans on CPU and probes both output paths for writability before CUDA/weight loading.
2. **Warmup cache:** Keyed by all training IDs, labels, and supplement content (a train-partition cache is not a complete hit). Exact hits skip warmup; fixed-rate lower-epoch hits train remaining epochs; exponential decay requires an exact target hit or fresh run.
3. **Checkpoint & merged export (LoRA):** Atomically saves and CPU-validates `adapter[_TAG].pt`, then merges LoRA weights on CPU (`merge_and_unload(safe_merge=True)`), exports `merged[_TAG]/` (config, 5 GB-sharded Safetensors at build precision, tokenizer, `export_metadata.json`) via staged directory replacement with `.merged[_TAG]-backup` recovery. Export failures preserve the already-saved `adapter[_TAG].pt`. See [export checks](lora.md#merged-backbone-export).

The build summary reports executed warmup and joint epochs separately (excluding cached warmup epochs), final training loss, checkpoint path, and merged directory. Load an exported backbone independently:

```python
from transformers import AutoModel, AutoTokenizer

path = "models/SR2_Qwen3_4B_lora/merged_experiment"
backbone = AutoModel.from_pretrained(path, dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
```

## Checkpoint compatibility

SR2 uses `swap_equivariant_abs_diff_v10` (`cross_projected_clamped_win_sum_diff`), adapter/head schema v2, frozen-head schema v1, and warmup schema v5 (`scaled_uniform_clamped_gelu_tie_v6`). See [head definition](lora.md#swap-equivariant-head), [v10 migration](migration.md#sr2-v10-fixed-tie-coefficient), and [checkpoint checks](lora.md#checkpoints).

## Training progress

Progress logs show `model/mode/stage/epoch`, percentage, work count, and `time: elapsed/estimated_total` (`elapsed * total_work / completed_work` in unbounded `HH:MM:SS`).

- **Frozen SR2 validation:** Silent per-epoch counter; prints one `HH:MM:SS.mmm` summary every `early_stopping` epochs plus the stopping epoch (`Early stopping`), separating `optimization_time`, `training_evaluation_time`, and `validation_time`.
- **LoRA / warmup:** Independent stages (**Warmup feature extraction**, **Head warmup** at ~20%, **Joint training** at ~10%; or **Cached head evaluation** on standalone hits). Cache hits skip progress for skipped stages; CPU prep, cache I/O, and publication are outside stage timers. See [LoRA progress](lora.md#training-progress) and [timing metrics](reports.md#timing-metrics).

## Retired profiles

Frozen Qwen3-4B, EXPT2, older standard LoRA, EXPT, QLoRA, and Qwen3-1.7B are unregistered; see [migration](migration.md) and [structure.md](../structure.md#package-map).
