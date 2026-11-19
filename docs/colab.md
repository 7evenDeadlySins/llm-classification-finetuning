# Colab setup and persistence

[Documentation index](../README.md)

Run project commands inside the Colab runtime (including when VS Code is the frontend). `/content/...` paths belong to that runtime; a local Windows terminal or `.env` does not configure it.

## Run holdout validation on Colab G4

### Repeat a validation run

1. Open the Drive-hosted `!validate.ipynb` from Colab's **Google Drive** or **Recent** tab (under **My Drive → Colab Notebooks**, avoiding duplicate `!validate (N).ipynb` copies) rather than uploading via a browser dialog. Update the Drive file directly when local `!validate.ipynb` changes. Reuse **My Drive → llm-classification-finetuning** and wait for `scripts/`, `configs/`, processed data (keeping each manifest with its `.parts/`), and caches to finish syncing before starting.
2. Select **G4 GPU** in **Runtime → Change runtime type**. If stuck in `Resuming execution` or a stale session, release it via **Runtime → Disconnect and delete runtime**, then click **Run all** directly (without clicking **Connect** first). Approve the Drive mount when prompted; hand browser control to the user for any manual OAuth/consent prompt.
   - If G4 is unavailable, head-only `SR2-Qwen3-4B` validation or builds may use **A100 GPU** with **High RAM** off. Release any auto-assigned session before switching, then click **Run all**. Leave other notebooks' sessions untouched.
3. Monitor progress and confirm `Completed N/N tasks.` in the Drive-synced `results/logs/validate-*.log` (rather than Colab's cross-origin output iframe, which blanks after disconnect). Confirm the runtime has released (**Reconnect** shown, **Disconnect and delete runtime** disabled) before opening `results/`.

### First-time Drive setup

Identify the shared folder from the project owner's link:
- **Existing name opens the intended folder:** Reuse it.
- **Name is available:** Add a shortcut directly under **My Drive** named `llm-classification-finetuning`.
- **Another entry uses the name:** Compare targets; move confirmed old uploads to Drive Trash after copying needed files, then reuse or add the shortcut.

Keep `scripts/`, `configs/`, processed data, `cache/`, and `Hugging Face/` in the shared folder, with `DRIVE_DATA_ROOT=/content/drive/MyDrive/llm-classification-finetuning` (and `MODEL_PATH` pointing to `Hugging Face/`).

**Agent authorization:** The project owner authorizes the agent to complete this workflow using the approved Google account, including syncing files, confirming Drive mount, and managing the runtime through completion.

## Set up a manual runtime checkout

1. Select a CUDA runtime and match YAML precision to the GPU (e.g., `fp16` on T4 instead of `bf16`).
2. Mount Drive if you want persistent caches and results:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

3. Clone and install in the runtime terminal (use `uv run` rather than the notebook kernel's Python):

   ```shell
   git clone https://github.com/Esxyko/llm-classification-finetuning.git /content/llm-classification-finetuning
   cd /content/llm-classification-finetuning
   python -m pip install uv
   uv sync --locked
   ```

4. Create `.env` in `/content/llm-classification-finetuning`:

   ```dotenv
   MODEL_PATH=/content/drive/MyDrive/Hugging Face
   CACHE_PATH=/content/drive/MyDrive/llm-classification-finetuning/cache
   RESULT_PATH=/content/drive/MyDrive/llm-classification-finetuning/results
   HF_ENDPOINT=https://huggingface.co
   HF_HUB_OFFLINE=1
   ```

   Populate `MODEL_PATH` with the complete native Hub cache (e.g., via `download.ipynb` with `HF_HUB_OFFLINE=0`) before offline execution. Restart the kernel if HF libraries were already imported with different settings.
5. Run a command or batch script (note: `os.environ` in a notebook overrides `.env` for child processes):

   ```shell
   uv run model SR2-Qwen3-4B-lora --config configs/1.yaml
   # or: uv run python scripts/validate.py / scripts/build.py
   ```

## Where files are written

| Path / setting | Writer & timing | Persistence |
| --- | --- | --- |
| `PROJECT_DIR` (`/content/...`) | Git checkout, venv, local data | Runtime-local |
| `MODEL_PATH` (Drive) | Populated separately (`HF_HUB_OFFLINE=0`); read offline | Persistent on Drive |
| `CACHE_PATH` (Drive) | Token, embedding, plan, and warmup caches on stage completion | Persistent on Drive (hits skip writes) |
| `RESULT_PATH` (Drive) | Streams batch logs; publishes completed validation runs and reports | Persistent on Drive |
| `PROJECT_DIR/models/` | `--build` writes `head[_TAG].pt` or `adapter[_TAG].pt` + `merged[_TAG]/` | Runtime-local until copied |
| `BUILD_OUTPUT_PATH` (`!build.ipynb`) | Copies current `models/SR2_Qwen3_4B/` and `models/SR2_Qwen3_4B_lora/` after batch | Timestamped backup on Drive |

Allow temporary space beside `CACHE_PATH` for [multipart reconstruction](caching.md#shared-storage-and-multipart-files). Verify write permissions on Drive shortcuts in the runtime rather than assuming web UI visibility grants write access. For manual runs with local `RESULT_PATH` or `models/`, copy outputs to Drive before releasing the runtime.

## Local notebook behavior

The Git-ignored `!validate.ipynb` and `!build.ipynb` each have four cells:

| Cell | Action |
| --- | --- |
| Settings | Define repo/branch, `PROJECT_DIR`, `DRIVE_DATA_ROOT`, `DRIVE_SCRIPTS_DIR`, `DRIVE_CONFIGS_DIR`, model/cache/result paths, endpoint, `HF_HUB_OFFLINE=1` (and `BUILD_OUTPUT_PATH` for builds) |
| Drive | Mount `/content/drive` and create `MODEL_PATH` |
| Setup | Force-checkout the selected branch (discarding tracked runtime edits) with `cache/` excluded; write runtime `.env`; install `uv` and `uv sync --locked` |
| Run & cleanup | Copy `DRIVE_SCRIPTS_DIR` → `PROJECT_DIR/scripts` and `DRIVE_CONFIGS_DIR` → `PROJECT_DIR/configs` (halting before batch/cleanup on failure); run `scripts/validate.py` or `scripts/build.py` with `--data-source-root DRIVE_DATA_ROOT`; back up `models/` (builds only); flush/unmount Drive and release runtime in `finally` |

### Data preparation, results, and build backups

- **Data mirror (`DRIVE_DATA_ROOT`):** Batch scripts check each selected YAML's `data.processed_path`, `data.external_rank_path`, and scope supplement locally first. Missing or wrong-sized files/volumes are copied from `DRIVE_DATA_ROOT` (including optional `summary.json` and `token_stats.parquet` for fast budgeting). Local manifests stay authoritative; missing data skips that task and logs the error while remaining tasks continue.
- **Results:** Because `RESULT_PATH` points to Drive, logs stream to `<RESULT_PATH>/logs/` and completed runs publish directly to Drive. Interrupted runs leave no completed result directory.
- **Build backups (`BUILD_OUTPUT_PATH`):** Defaults to `/content/drive/MyDrive/llm-classification-finetuning/models`. After the batch (even if a task fails), existing `models/SR2_Qwen3_4B/` and `models/SR2_Qwen3_4B_lora/` directories are copied to `<BUILD_OUTPUT_PATH>/build-<timestamp>/`. Warmup-only runs skip copying. If copying fails, Drive stays mounted and the runtime stays open for recovery; if copying succeeds or is skipped, `finally` unmounts Drive and releases the runtime.

## Diagnose a failed run

- Read the **first model error** in `<RESULT_PATH>/logs/validate-*.log` or `build-*.log` (the trailing `CalledProcessError` is only a wrapper).
- Verify the selected YAML, runtime cwd, and exported environment (not local Windows `.env`).
- Check [troubleshooting](troubleshooting.md) for CUDA, memory, and input failures.
