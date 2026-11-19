# Prepare training data

[Documentation index](../README.md)

After [setup](getting-started.md), choose an operation (all accept `--config PATH` and run on CPU without model weights):

| Command / input | Behavior |
| --- | --- |
| Saved processed Parquet (or manifest + volumes) | Start validation/build directly; no raw download or preprocessing needed |
| `uv run init` | Download missing `train.csv`, then rebuild processed data |
| `uv run data` | Download missing `train.csv` only |
| `uv run preprocess` | Rebuild processed split from local `train.csv` |

Use the same saved split when comparing model experiments.

## One shared train/validation split

1. **Identical-response tie correction:** Before deriving labels or prompt hashes, rows with identical `response_a == response_b` (exact string match, no trimming/case folding) are corrected to `winner_tie=1` without modifying the raw CSV.
2. **Prompt-grouped split:** Derives `group_id` (SHA-256 of canonical prompt JSON) and assigns whole prompt groups to `train` or `validation` (Kaggle baseline: `validation_fraction=0.2`, `random_state=42`), preserving corrected class proportions.
3. **Token balancing:** Counts tokens with the configured validation tokenizer/chat template (`add_special_tokens=False`, no padding/truncation) via [shared text cache](caching.md#shared-token-encodings). Up to 1,000 whole-group exchanges with identical three-class count vectors balance validation tokens.

Relative deviation `abs(validation_share / f - 1)` must be strictly below `0.005` for canonical rows and `validation.max_token_deviation` (default `0.005`) for canonical tokens. Each source ID is saved once (`is_swapped=False`). Failed checks leave the previous dataset intact.

## Processed data

The repository-backed logical path is `data/processed/train_validation.parquet`. Current presets select external rows from `data/processed/external_merged/train_validation.parquet` (220,492 rows, stored outside Git). Always point `data.processed_path` to the logical `.parquet` path:
- `< 100 MiB`: single Zstandard-compressed `.parquet` file.
- `>= 100 MiB`: `.parquet.manifest.json` + `.parquet.parts/` (volumes `<= 99 MiB`).

Loaders reconstruct/checksum multipart files in `tempfile.gettempdir()`. See [shared storage](caching.md#shared-storage-and-multipart-files).

Added columns alongside source fields:

| Column | Meaning |
| --- | --- |
| `label` | `0` = A wins, `1` = B wins, `2` = tie |
| `group_id` | SHA-256 of canonical prompt JSON |
| `split` | `train` or `validation` |
| `is_swapped` | Always `False` |

Loaders reject missing values, invalid labels, `is_swapped=True` rows, duplicate IDs, and prompt groups crossing partitions.

## Merged external preferences

The real preference merge (`Kaggle + LMSYS 33K + Arena human + Arena expert`) follows the [adopted processing principles](recommended-external-data.md#adopted-processing-principles): hard labels only, equal row weights, and `40,960`-token branch context limit without truncation. Current presets use [ranked external selection](#ranked-external-training-selection) and set `train_supplement_path: null` and `build_supplement_path: null`.

Rebuild from the project root (CPU + tokenizer/config only; network needed for initial raw downloads):

```powershell
uv run python data/merge_external_data.py --config config.yaml
```

- `--baseline-path` defaults to `data/processed/lmsys33k/train_validation.parquet` (supplying the 57,477 Kaggle rows and 11,495 frozen validation IDs).
- `--output-dir` defaults to `data/processed/external_merged`.
- Raw sources and `sources.json` (pinned commits + LFS SHA-256 checksums) live under `data/raw/external/`.
- IDs: Kaggle unchanged; LMSYS = `1 * 2**32 + row`, Arena human = `2 * 2**32 + row`, Arena expert = `3 * 2**32 + row`.

Output artifacts in `--output-dir`:

| Artifact | Contents |
| --- | --- |
| `train_validation.parquet` (+ manifest/parts) | Merged dataset |
| `validation_ids.csv`, `sources.json` | Frozen validation IDs and pinned source lock |
| `row_audit.parquet`, `conversion_audit.parquet` | Raw row dispositions, parsing, and exclusions |
| `dedup_audit.parquet`, `vote_counts.parquet` | Battle/vote IDs, A/B alignment, majority-vote counts |
| `token_stats.parquet`, `excluded_model_context.parquet` | Complete branch lengths and battles exceeding 40,960 tokens |
| `summary.json`, `checksums.sha256` | Counts, fingerprints, integrity checks, and hashes |

### 2026-11-17 build summary (220,492 rows)

- **Train:** 208,997 rows (45,982 Kaggle, 24,227 LMSYS, 135,440 Arena human, 3,348 Arena expert; 59,610 ties, 28.5219%).
- **Validation:** 11,495 frozen Kaggle rows (`validation_fraction = 11495 / 220492 = 0.05213341073599042`).
- **Exclusions:** 102 over-context battles (78 Arena human, 24 Arena expert), 17 tied-vote battles, 12 unalignable Arena human records, 1 non-text expert record; 1,584 rows above 13,866 tokens retained (max branch: 40,953; max A+B: 78,827).

### Deduplication and voting API

Battle deduplication keys on complete normalized prompt context and unordered response pair (catching A/B swaps), keeping distinct response pairs for the same training prompt and excluding any prompt matching frozen validation (`group_id`). Only Kaggle training rows take precedence; LMSYS and other external sources have equal weight.

```python
from llm_classification_finetuning.preprocess import ExternalBattleDeduplicator

result = ExternalBattleDeduplicator().deduplicate(
    external_rows, existing_rows, merge_votes=True
)
retained_rows, dedup_audit, vote_counts = result.data, result.audit, result.vote_counts
```

Pass only Kaggle rows as `existing_rows` (with `split`) and combine LMSYS + other external rows in `external_rows`. After rebuilding, update `validation.validation_fraction` to `11495 / total_rows` in YAML and regenerate any configured supplements (do not run `init` or `preprocess` on merged data).

### Ranked external training selection

Presets set `data.external_rank_path: data/processed/external_english_rarity_rank.csv` (`rank`, `id`, `source`, `label`). For the active scope (validation/warmup or build), the loader uses the Kaggle training A+B token sum as the external token budget, sets external ties to `round_half_up(n * Kaggle_tie_share)`, selects ties and non-ties separately in ascending `rank`, and picks `n` whose token total is closest to the budget (smaller `n` on ties). Lengths reuse `token_stats.parquet` when matching, or count via the active tokenizer. Set `data.external_rank_path: null` to train on the full saved partition.

| Scope | Kaggle rows | Kaggle tokens | External rows | External ties | External tokens | Total train rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation / standalone warmup | 45,982 | 35,856,134 | 6,658 | 2,060 | 35,865,470 | 52,640 |
| Full-data builds / inline warmup | 57,477 | 44,873,727 | 8,934 | 2,765 | 44,865,610 | 66,411 |

### External non-tie downsampling

Historical 2026-11-18 snapshot (`data/processed/external_merged_downsampled/`, not retained locally) removed 16,382 external non-tie training rows from the 220,492-row merge (`9c1f041d...`) to yield **204,110** rows (**192,615** train, **11,495** validation; `c0e4ca20...`; tie share 30.94802%). Candidates were sorted by prompt frequency desc, sum of branch frequencies desc, single-turn first, `tokens_a + tokens_b` desc, and `id` asc (removing 12,270 Arena human, 3,365 LMSYS, 747 Arena expert). Current presets instead use ranked selection from the full merge.

## Tie duplication

`data/generate_tie_duplicate.py` (`--scope train|build|all`, `--target-count` default `7343`) duplicates selected distinct-response real ties (`label=2`, `is_swapped=False`, `response_a != response_b`, max 1 per `group_id`), ordered by prompt frequency asc, multi-turn first, `tokens_a + tokens_b` asc, and `id` asc. Currently disabled (`null` in all presets). `BuildSupplementLoader` validates `llm_classification_finetuning.row_duplicate` schema v2, `source_sha256`, and validation isolation.

## Scope-specific synthetic ties

`data/generate_synthetic_tie.py` (`--scope train|build|all`) generates self-response ties (`response_a == response_b`) sized to match Kaggle's tie share `17788/57477` (30.94803%). Currently disabled (`null` in all presets; local files removed).

- **Train scope** uses real `split="train"` rows; **build scope** uses both partitions. Excludes prompt groups with any real tie, balances historical win rates to an exact `1/2` mean (`7343/2` for train, `3671` for build) via minimal length-increasing swaps, and validates schema v2 provenance/checksums on load.
- **Historical LMSYS33K supplement** (`data/synthetic_tie/lmsys33k/`, removed locally) added 1,012 schema-v1 ties to the 77,527-row snapshot.

## Conversation serialization

SR2 uses `sr2-qwen3-chat-template-v1`: alternating `user`/`assistant` turns rendered via `apply_chat_template(tokenize=False, add_generation_prompt=False, continue_final_message=False)` with no system message or tools, preserving the official `<think>\n\n</think>\n\n` block and trailing newline, encoded with `add_special_tokens=False`. `null` responses become empty assistant messages.

Presets pin `Skywork/Skywork-Reward-V2-Qwen3-4B` at revision `fd958fef475f323f4e6b195930e3dd918485c668`. SR2 never truncates and rejects empty or over-limit branches; see [input limits](lora.md#hardware-and-input-limits).

## Historical data snapshots

- **2026-10-27 SR2 split:** Rebuilt 57,477 IDs (114,954 augmented rows, 51,734 groups; train 45,982 / 35,855,325 tokens, val 11,495 / 9,018,402 tokens; SHA-256 `2b064171...`).
- **2026-11-06 tie correction:** Fixed 27 original IDs (54 rows) with identical responses to `label=2` (`f5273191...`).
- **2026-11-12 A/B swap removal:** Dropped all `is_swapped=True` rows in place (`data/processed/train_validation.parquet`: 59,533 original rows, `0591f630...`; `lmsys33k`: 77,527 rows, `0488ff2b...`).
- **2026-11-19 LMSYS33K ranked ties:** Supplemented `lmsys33k` in place with 1,013 real Arena ties to **78,540** rows (67,045 train, 11,495 val; multipart SHA-256 `43eb5997...`). See [LMSYS33K notes](../data/processed/lmsys33k/README.md).

Older five-fold configs/datasets are rejected; see [migration](migration.md#five-folds-to-one-holdout). Next: [model workflows](models.md).
