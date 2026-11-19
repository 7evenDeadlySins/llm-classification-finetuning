# Caching and Python APIs

[Documentation index](../README.md)

Training caches live under `CACHE_PATH` (default `./cache`); downloaded models use `MODEL_PATH`; production checkpoints live under `models/`.

## Cache overview

| Artifact under `CACHE_PATH` | Keyed by | Excluded |
| --- | --- | --- |
| `token_cache/` | Text UTF-8 SHA-256, tokenizer content/versions, encoding options, schema | IDs, labels, split, weights, training settings |
| `embedding_cache/` | Effective branch token IDs, exact backbone identity, hidden size, precision, storage dtype, pooling/extraction version | IDs, labels, split, head/training settings, attention, TF32, extraction budget |
| `adapter_batches/plans/` | Sorted `(id, is_swapped, A_len, B_len)`, checkpointing threshold, update-group size, algorithm version, supplement content | Labels, source order, token content beyond lengths, second-pass budget |
| `SR2_Qwen3_4B_lora_warmup/` | Sorted training IDs & labels, supplement content, effective SR2 warmup identity | Original text, validation-only IDs/labels, precision, joint settings |

Cache compatibility and checkpoint compatibility are independent. `--refresh-cache` refreshes requested entries and mode-specific caches only. Live LoRA embeddings are never persisted.

## Hugging Face model downloads

When `MODEL_PATH` is set, loaders pass it as `cache_dir` (e.g., `<MODEL_PATH>/models--Skywork--Skywork-Reward-V2-Qwen3-4B/`). With `HF_HUB_OFFLINE=1`, loaders set `local_files_only=True` and report missing files explicitly; populate caches beforehand with `HF_HUB_OFFLINE=0` (or `download.ipynb`). Weight loading logs start/end elapsed time while suppressing progress bars and `LOAD REPORT` noise. See [download defaults](getting-started.md#hugging-face-downloads).

## Shared storage and multipart files

`CacheStorage` manages all four cache types and processed Parquet files using logical paths (`.sqlite3`, `.pt`, `.parquet`):
- `< 100 MiB` (104,857,600 bytes): single file at the logical path.
- `>= 100 MiB`: `<name>.manifest.json` + `<name>.parts/` (`000001-<sha256>.part`, max 99 MiB each). Keep manifests and all parts together.

Reads reconstruct/checksum multipart files in `tempfile.gettempdir()`. Writes use a 30-second per-artifact lock, stage files beside the destination on Windows (preserving inherited directory ACLs), and replace atomically. Unchanged hits skip writes; corrupt artifacts are archived before rebuild (corrupt processed data requires re-running preprocessing).

## Shared token encodings

SQLite shards fingerprint tokenizer content/versions, encoding options, and schema. SR2 renders complete conversations via `sr2-qwen3-chat-template-v1` and encodes with `add_special_tokens=False` into compact int32 CPU views without truncation. `TokenizationCache.count_tokens()` shares the namespace and persists exact counts in `token_cache/<namespace>.lengths.sqlite3` for external-data preparation.

## Warmup features

Frozen SR2, standalone warmup, and inline non-quantized SR2 warmup share adapter-free per-branch features in `embedding_cache/<key>.sqlite3` (`last-real-token-right-padding-no-mask-v1`, extraction/schema v1). BF16 compute stores raw BF16; FP16/FP32 compute store FP32.

- Frozen SR2 and standalone warmup initialize only the head before checking features: complete hits skip loading backbone weights; partial hits lazily load the backbone (preserving CPU/CUDA RNG) and extract only missing unique branches.
- Inline LoRA warmup loads the adapter-bearing model eagerly and extracts only missing unique branches.

## LoRA batch plans

`adapter_batches/plans/*.pt` (schema v4, algorithm v3) stores base `G`-row update groups packed under `P = rows * (longest_A + longest_B) <= threshold` (comparing 3 global sorts, suffix DP vs. best-fit). Epoch shuffling, short-tail merging, and second-pass singleton repacking (`gradient_checkpointing_budget`) happen in memory so changing the second-pass budget reuses the base plan. Threshold `0` skips persistent plans.

## Frozen SR2 head training

- **Validation:** Shares token and feature caches with warmup, but always trains a fresh head from epoch 1 with `ReduceLROnPlateau` + `early_stopping` (never reads/writes completed head-state caches).
- **Builds (`--build`):** Maps `build_epochs`, `training_batch_size`, `learning_rate`, and `build_min_lr` to the shared warmup namespace. Exact hits evaluate all training data without training; fixed-rate partial hits resume remaining epochs; exponential decay requires an exact target-epoch hit.

## LoRA head warmup

Frozen SR2 builds, LoRA SR2, and standalone warmup write `SR2_Qwen3_4B_lora_warmup/<key>.pt` (schema v5, `swap_equivariant_abs_diff_v10`, `head_initialization_version: scaled_uniform_clamped_gelu_tie_v6`).

- **Exponential decay (`build_min_lr < learning_rate`):** Exact target-epoch match only (`warmup_lr_schedule: exponential_endpoint_epoch_v2`, `warmup_build_min_lr`, `warmup_epochs`). Misses train from epoch 0.
- **Fixed rate (`build_min_lr = learning_rate`):** Checks target epoch `N`, then `N-1..1` for the largest compatible lower-epoch hit.

### Identity

| Included in compatibility | Excluded |
| --- | --- |
| Sorted training `(id, label)` pairs (`training_labels_sha256` + `training_ids_sha256`); supplement SHA-256 (`training_supplement_sha256`) when enabled | Validation-only IDs/labels, holdout/build label, source paths/row order, **original training text** |
| Hub revision / local weight & config fingerprint, hidden size, tokenizer, serializer, head architecture, init version, quantization, schema | Local path spelling when contents match |
| Head dropout, A/B objective, fixed `t`, warmup build epochs/batch size/LR/min LR/schedule, seed, weight decay, clipping, nonzero warmup label smoothing | Compute precision, attention, TF32, extraction budget, `max_length`, LoRA/joint settings |

> [!WARNING]
> Editing original training text while keeping the same IDs and labels can reuse stale warmup state. Pass `--refresh-cache` to retrain.

### Saved state and restoration

Stores FP32 head weights, named head AdamW moments/steps, `GradScaler` state/precision, and Python/NumPy/Torch CPU/CUDA RNG states (no adapters, features, or losses).

- **Inline LoRA complete hit:** Restores head/scaler/RNG, skips warmup extraction/training/eval, and clears AdamW moments/steps once on entry to joint training.
- **Fixed-rate partial hit (`K < N`):** Loads/extracts features, replays `K` epochs on an index-only DataLoader to restore shuffle state, restores head/optimizer/RNG, trains epochs `K+1..N`, and saves the `N`-epoch cache (preserving `K`). Joint entry still clears AdamW moments/steps once.
- **Standalone warmup (`model warmup --build`):** Evaluates full-train (warmup-smoothed) and validation (unsmoothed) losses after each epoch; complete hits load backbone + cached features and recompute both losses (`0` epochs, `Reused`).

## Python APIs

Import public exports (CPU data/batch imports do not import PEFT or initialize CUDA):

### Text encodings

```python
from pathlib import Path
from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.models.common import TokenizationCache

config = AppConfig.load(Path("config.yaml"))
cache = TokenizationCache.from_pretrained(
    config.cache_dir / "token_cache", "Skywork/Skywork-Reward-V2-Qwen3-4B"
)
encoded = cache.encode(["first input", "second input"])
```

Pass `ConversationSerializer(cache.tokenizer)` to `ModelDataRepository` and encode with `add_special_tokens=False` (`PairwiseTokenStore` does this automatically).

### CPU training-loader preparation

```python
from pathlib import Path
from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.models.SR2_Qwen3_4B_lora import (
    LoRATrainingBatchRepository,
)

config = AppConfig.load(Path("config.yaml"))
batches = LoRATrainingBatchRepository.from_config(config)
training = batches.get_training()
training.set_epoch(0)
loader = training.loader
```

For full-data builds, pass `include_build_supplement=True` to `from_config` and call `get_training(full_data=True)`.

### Storage sessions

```python
from pathlib import Path
import torch
from llm_classification_finetuning.models.common import CacheStorage

storage = CacheStorage(Path("cache/example.pt"))
with storage.open("w") as session:
    torch.save({"example": 1}, session.path)
with storage.open("r") as session:
    payload = torch.load(session.path, map_location="cpu", weights_only=True)
```

`open("w")` publishes on exit; `open("rw")` publishes if `session.mark_dirty()` was called; `open("r")` never publishes. Close SQLite handles before exiting the `with` block. Use `missing_components()` and `copy_missing_from(source, on_copy=...)` for incremental multipart mirror copies.
