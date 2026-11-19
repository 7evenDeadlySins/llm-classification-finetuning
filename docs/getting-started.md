# Getting started

[Documentation index](../README.md)

## Requirements

- Python **3.14+** (`.python-version`, `pyproject.toml`).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
- One CUDA GPU for validation, builds, or standalone warmup (data prep and reporting run on CPU).
- A Kaggle account and accepted competition rules **only when downloading raw competition files**.

Run commands from the project root. For Colab runtimes, start with [Colab setup](colab.md).

## Install and configure

```shell
uv sync --locked
```

The lockfile selects PyTorch `2.14.0+cu130` (CUDA 13.0).

If you need Kaggle credentials or custom storage paths, copy `.env.example` to `.env` (`Copy-Item .env.example .env` on PowerShell, `cp .env.example .env` on Linux):

```dotenv
KAGGLE_API_TOKEN=your_token_here
MODEL_PATH=
CACHE_PATH=cache
RESULT_PATH=results
```

- Get `KAGGLE_API_TOKEN` from [Kaggle API settings](https://www.kaggle.com/settings/api).
- Existing process environment variables override `.env`.
- `MODEL_PATH` sets the Hugging Face model cache; `CACHE_PATH` sets training caches; `RESULT_PATH` sets validation runs, reports, and batch logs (default `./results`).
- Relative paths resolve from the process working directory. Changing paths does not move existing files; see [path rules](configuration.md#configuration-files-and-paths).

## Choose the data source

| Starting point | Action |
| --- | --- |
| Use the saved split in this checkout | Keep `train_validation.parquet` (or its manifest and all `.part` files) and proceed to validation |
| Download raw data and regenerate the split | Run `uv run init` after setting `KAGGLE_API_TOKEN` |
| Raw `train.csv` is already present | Run `uv run preprocess` |
| Download `train.csv` without preprocessing | Run `uv run data` |

The logical split path is `data/processed/train_validation.parquet` (single file below 100 MiB, or manifest + `.parts/` above). Backbone weights and tokenizers are not in Git and are downloaded separately. Use the same saved split when comparing experiments; see [data preparation](data-preparation.md).

## Validate and report

```shell
uv run model SR2-Qwen3-4B-lora
uv run result
```

Validation writes `validation.csv`, `metrics.json`, and `config.yaml` under `results/SR2-Qwen3-4B-lora-<timestamp>/`. `uv run result` selects the most recently modified eligible run and replaces `results/comprehensive/`. Pass a run name explicitly when comparing experiments; see [reports](reports.md).

Presets are experiment settings, not guaranteed memory fits for every GPU—adjust [GPU precision](configuration.md#gpu-configuration) and [memory controls](lora.md#memory-tuning) as needed.

Optional `uv run model warmup --build` pre-trains only the SR2 head on the train partition in the shared SR2 warmup namespace. It is not required before a normal LoRA run; see [standalone warmup](configuration.md#standalone-warmup).

## Hugging Face downloads

Package initialization loads `.env` before Hugging Face imports:

- `.env.example` sets `HF_HUB_OFFLINE=1` for offline execution from a complete local cache (`local_files_only=True`). Populate the native Hub layout (refs, snapshots, blobs, all shards) beforehand with `HF_HUB_OFFLINE=0` or `download.ipynb`.
- `HF_ENDPOINT` defaults to `https://hf-mirror.com` if unset. Set `HF_ENDPOINT=https://huggingface.co` in `.env` or the environment to use the official Hub.
- For the mirror only, `HF_HUB_DISABLE_XET=1` is set by default unless explicitly configured.

In Python notebooks, import `llm_classification_finetuning` before `transformers` or `huggingface_hub` (restart the kernel if already imported). See [model download storage](caching.md#hugging-face-model-downloads).

Next: [commands](commands.md), [configuration](configuration.md), or [troubleshooting](troubleshooting.md).
