# Validation reports and metrics

[Documentation index](../README.md)

Holdout validation publishes a timestamped directory containing `validation.csv`, `metrics.json`, and `config.yaml` (builds do not). Frozen `SR2-Qwen3-4B` publishes `best_epoch` predictions; LoRA publishes final-epoch predictions.

## Generate a report

```shell
uv run result
uv run result RUN_NAME --config path/to/report.yaml
```

Without `RUN_NAME`, discovery picks the most recently modified eligible directory under `RESULT_PATH` (default `./results`, excluding `comprehensive`, `logs`, historical `submit`/`test`, and `.comprehensive-*` staging/backup dirs) and validates it without searching backward.

For historical runs with retired model sections, supply a separate current-format `report.yaml` containing only `data`, `validation`, and `gpu` pointing to that run's processed Parquet. Each invocation replaces `<RESULT_PATH>/comprehensive/`.

## Report inputs

The target directory must contain `validation.csv` and a holdout `metrics.json` (five-fold runs are rejected). Required CSV columns: `id,winner_model_a,winner_model_b,winner_tie`, with finite integer IDs and probabilities in `[0, 1]` summing to `1` within `1e-6`:

| Profile / aggregation | Required validation rows |
| --- | --- |
| SR2 / `original_only` | Original validation rows only, in processed-data order, one per ID |
| Historical frozen / `ab_swap_average` | All `split=validation` rows (original and swapped) in processed-data order |
| Historical EXPT2 / `original_only` | Original validation rows only, in processed-data order |

## Publication and outputs

| File under `<RESULT_PATH>/comprehensive/` | Contents |
| --- | --- |
| `confusion_matrix.png` | 3x3 counts and expected-class row percentages (swapped predictions restored to original A/B order) |
| `records.xlsx` | Per-ID record with split, expected/actual labels, incorrect count, and log loss (`AVG` recalculates over filtered rows) |

Reports are staged before atomic replacement; recoverable failures restore the previous report or report the retained backup path.

### Aggregation policy

`metrics.json` (`prediction_aggregation`) selects aggregation:
- `ab_swap_average`: One prediction, one count, loss of averaged distribution (original and restored-swapped must match within `1e-6`).
- `original_only`: One original prediction, one count, `incorrects` in `{0, 1}`, one log loss.
- Missing/`null` (valid historical holdout): Two orientation votes, `incorrects` in `0..2`, mean of the two orientation losses.

## Configuration snapshots

`config.yaml` uses public input names, resolved data paths, and loaded defaults for the active model only (`data`, `validation`, `gpu`, `t`, `SR2_Qwen3_4B`; or `data`, `validation`, `gpu`, `t`, `warmup`, `SR2_Qwen3_4B_lora`), omitting internal fields and `A/B_swap`. It does not capture `.env` (`MODEL_PATH`, `CACHE_PATH`, `RESULT_PATH`). Retain historical snapshots unchanged.

## Model metrics

`metrics.json` uses `validation_strategy: holdout`, `validation_loss` (always unsmoothed), and a flat `epochs` list (`source_training_rows` and `validation_rows` count canonical IDs; `training_rows` counts rows selected by the A/B policy).

- **Frozen SR2:** Adds `executed_training_epochs`, `best_epoch` (lowest unrounded validation loss; ties keep earlier epoch), and `lr_reduction_count` (actual `ReduceLROnPlateau` steps lowering at least one optimizer group's LR across all executed epochs). `epochs` includes every executed epoch through the stopping epoch.

| Profile | `head_architecture` | `embedding_combination` |
| --- | --- | --- |
| SR2 | `swap_equivariant_abs_diff_v10` | `cross_projected_clamped_win_sum_diff` |
| Historical SR2 (trainable `t`) | `swap_equivariant_abs_diff_v9` | `cross_projected_clamped_win_sum_diff` |
| Historical frozen | `pairwise_mlp` | `concat_a_b_abs_diff` |
| Historical EXPT2 | `swap_equivariant_tie_from_wins_v1` | `sum_signed_diff_abs_diff` |

Intermediate historical SR2 markers (`v2`–`v8` and harmonic/discrepancy `v1`–`v4`) are preserved in existing result files and documented in [migration](migration.md).

## Timing metrics

Serialized durations use whole-second `HH:MM:SS` (unbounded hours); frozen SR2 terminal summaries use `HH:MM:SS.mmm` and split `training_time` into `optimization_time` and `training_evaluation_time`.

| Field / scope | Includes |
| --- | --- |
| Epoch `training_time` | Optimization phase + end-of-warmup-epoch train evaluation |
| Epoch `validation_time` | Validation phase (joint LoRA includes fresh extraction + head inference) |
| Epoch `total_time` | Epoch setup + train/val (excludes pre-epoch warmup feature extraction) |
| Top-level `training_time`, `validation_time` | Sums over all executed epochs (including post-`best_epoch`) |
| Final `total_time`, `config_total_time` | Full run: init, data/cache/feature prep, training, validation, cleanup, publication |

Joint epochs also log/record `last_step_training_loss` (mean training loss across the final consumed optimizer group, including merged tails and FP16-skipped updates, with training smoothing/dropout). Warmup epochs omit it; builds keep `final_training_loss` as the final-epoch mean. Standalone warmup cache hits execute `0` epochs and report current train/validation losses.

## GPU memory peaks

All modes monitor PyTorch allocator peaks on `gpu.device` from GPU/model setup through saving, export, cleanup, and publication (after full-data build CPU prep and write preflights):

| Scope | Includes |
| --- | --- |
| `training` | Warmup and joint optimization + warmup train-set evaluation |
| `validation` | Holdout feature extraction and head inference |
| `global` | Entire monitored run (loading, warmup extraction, cache saves, export, cleanup, publication) |

Monitors reset allocator peak counters at phase boundaries while preserving global maxima. Terminal output prints `training`, `validation`, and `global` peaks in GiB (omitting unexecuted phases). Holdout runs atomically append byte counts to `metrics.json` under `gpu_memory` (`stages` holds `training` / `validation`).

- `peak_allocated_bytes` = live tensor memory; `peak_reserved_bytes` = caching allocator blocks (including reused blocks from earlier phases).
- Failures after monitoring starts print observed peaks (`before failure`) without masking the error or writing `metrics.json`.
