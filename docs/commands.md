# Command reference

[Documentation index](../README.md)

Run commands from the project root after [setup](getting-started.md). All entry points accept `--config PATH` (default `config.yaml`), except the batch scripts, which configure selections via constants. Run `uv run COMMAND --help` for full parser options.

## Entry points

| Command | Purpose | Default output |
| --- | --- | --- |
| `uv run init` | Download missing `train.csv` and rebuild processed split | `data/processed/train_validation.parquet` |
| `uv run data` | Download missing Kaggle `train.csv` only | `data/raw/train.csv` |
| `uv run preprocess` | Rebuild processed split from local `train.csv` | `data/processed/train_validation.parquet` |
| `uv run model MODEL` | Holdout validation | `results/MODEL-<timestamp>/` |
| `uv run model MODEL --build` | Full-data build (head, or LoRA adapter/head + merged backbone) | `models/SR2_Qwen3_4B/head.pt`, or `models/SR2_Qwen3_4B_lora/adapter.pt` + `merged/` |
| `uv run model warmup --build` | Train/cache SR2 head on train partition only | `<CACHE_PATH>/SR2_Qwen3_4B_lora_warmup/<key>.pt` |
| `uv run result [RUN]` | Generate report for a selected or latest validation run | `results/comprehensive/` |
| `uv run python scripts/validate.py` | Sequential batch holdout validation | Run dirs + `results/logs/validate-*.log` |
| `uv run python scripts/build.py` | Sequential batch full-data builds and standalone warmup | Checkpoints/caches + `results/logs/build-*.log` |

Processed `.parquet` paths are logical (single file below 100 MiB, manifest + `.parts/` above); see [multipart storage](caching.md#shared-storage-and-multipart-files). `RESULT_PATH` overrides `./results` for validation runs, reports, and batch logs.

## Model selection and execution modes

`MODEL` is `SR2-Qwen3-4B`, `SR2-Qwen3-4B-lora`, or `ALL` (case-sensitive). `warmup` is a separate build-only selector excluded from `ALL`.

| Option | Behavior |
| --- | --- |
| *(default)* | Holdout validation; frozen SR2 publishes `best_epoch` (early stopping), LoRA publishes final epoch; no production checkpoint |
| `--build` | Train on all original IDs across both partitions; save production checkpoint (and merged backbone for LoRA) |
| `--refresh-cache` | Rebuild requested token entries and mode-specific caches |
| `--checkpoint-tag TAG` | Write `head_TAG.pt` or `adapter_TAG.pt` + `merged_TAG/` (`--build` only; ASCII `[A-Za-z0-9][A-Za-z0-9_-]*`) |

`ALL` runs configured frozen SR2 before LoRA, continues past individual failures, and exits `1` if any fail. The entire YAML is validated before any model runs.

## Standalone warmup

`uv run model warmup --build` accepts `--config` and `--refresh-cache` (validation mode and `--checkpoint-tag` are unsupported). It trains the SR2 head on original train-partition rows using `warmup.model_name` and evaluates full-train (warmup-smoothed) and validation (unsmoothed) losses after each epoch on shared frozen features.

- **Exact cache hit:** Restores head, loads backbone + cached features (extracting only missing branches), recomputes both losses (`0` executed epochs), and does not rewrite the cache.
- **Fixed-rate continuation (`warmup.build_min_lr = warmup.learning_rate`):** Resumes from the largest compatible lower-epoch cache, trains remaining epochs, and writes a separate target cache (`Resumed`). Exponential decay reuses only exact target-epoch caches.
- Produces no adapters, joint training, production checkpoints, or result reports. See [warmup configuration](configuration.md#standalone-warmup) and [cache reuse](caching.md#standalone-warmup-build).

## Cache refresh

`--refresh-cache` refreshes only entries requested by the active mode (tokens, warmup/frozen features, batch plans, and mode-specific head state)—it does not delete unrelated caches or Hub weights. See [cache identities](caching.md).

## Batch validation and builds

- [`scripts/validate.py`](../scripts/validate.py) runs holdout validation (`uv run model MODEL --config PATH`) and logs to `<RESULT_PATH>/logs/validate-*.log`. Does not support `warmup`.
- [`scripts/build.py`](../scripts/build.py) runs full-data builds and standalone warmup (`uv run model MODEL --config PATH --build`) and logs to `<RESULT_PATH>/logs/build-*.log`. Does not pass `--checkpoint-tag` (sequential builds for the same model overwrite default outputs).

Both scripts are configured via top-level `CONFIG_DIR` and `MODEL_CONFIGS` constants (currently `validate.py` runs `SR2-Qwen3-4B` on `l1.yaml`–`l4.yaml`; `build.py` runs `SR2-Qwen3-4B-lora` on `l1.yaml`). They run in dictionary/list order, load `.env` from the project root, stream combined output to terminal and UTF-8 log, continue past task failures, and exit `1` if any task fails.

### Project mirror (`--data-source-root`)

Both batch scripts verify each task's `data.processed_path`, `data.external_rank_path`, and scope supplement (`train_supplement_path` for validation/warmup; `build_supplement_path` for builds) before launching the child process. Pass `--data-source-root PATH` to fill missing or wrong-sized files/volumes from a Drive mount or mirror:

```bash
uv run python scripts/validate.py --data-source-root /content/drive/MyDrive/llm-classification-finetuning
uv run python scripts/build.py --data-source-root /content/drive/MyDrive/llm-classification-finetuning
```

Local manifests stay authoritative; when missing locally, source manifests take precedence over single files. Successfully prepared paths are shared within the batch; preparation failures skip that task and mark the batch failed without running `init` or `preprocess`.

## Report selection

`RUN` in `uv run result [RUN]` is a direct child directory name under `RESULT_PATH` (default `./results`). Omitting it picks the most recently modified eligible directory and validates it (without searching backward). Reports always overwrite `<RESULT_PATH>/comprehensive/`. See [report inputs](reports.md#report-inputs) and [migration](migration.md) for removed selectors/`--test`.
