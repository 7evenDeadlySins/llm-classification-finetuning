# Prepare training data

[Documentation index](../README.md)

After [setup](getting-started.md), choose the operation you need:

| Command / input | Behavior |
| --- | --- |
| Saved processed Parquet, or its manifest and all volumes | Start model validation directly; no raw download or preprocessing required |
| `uv run init` | Download missing competition files, then rebuild processed training data |
| `uv run data` | Download missing `train.csv` only |
| `uv run preprocess` | Rebuild processed data from local `train.csv` |

All commands accept `--config PATH`. Downloads skip existing files and normalize
single-file ZIP wrappers. Preprocessing needs a tokenizer and may download its
files, but loads no model weights and needs no GPU. Use the same saved split
when comparing model experiments.

## One shared train/validation split

Preparation starts from the complete raw training table and validates its IDs
and one-hot winner fields. Exactly matching `response_a` and `response_b` fields
are then corrected to tie: `winner_model_a=0`, `winner_model_b=0`, and
`winner_tie=1`. Comparison uses the complete stored fields without trimming,
case folding, or fuzzy matching; whitespace, newline, and case differences remain
significant. The raw CSV is not modified.

Preparation derives labels and canonical prompt hashes from the corrected table,
then assigns whole prompt groups to train or validation. The saved Kaggle baseline
uses validation fraction `0.2` and seed `42`. Classes retain their corrected
proportions; they are not resampled to equal sizes.

Original-class quotas use the nearest total validation row count and largest
fractional remainders. Multirow prompt groups are selected deterministically,
then singleton groups fill the remaining class quotas. Infeasible quotas fail
rather than splitting a prompt group.

Token counting uses the training conversation serializer, configured tokenizer's
chat template, and shared [text cache](caching.md#shared-token-encodings), without
padding or truncation. Template output is encoded with `add_special_tokens=False`
because it already contains the control tokens. Whole groups with identical
three-class count vectors are exchanged to meet the validation token target.
This preserves initial row and class counts exactly; at most 1,000 improving
exchanges are allowed.

Both row and token shares are measured on original rows. For validation
fraction `f`, relative deviation is `abs(validation_share / f - 1)`:

| Quantity | Required strict upper bound |
| --- | --- |
| Validation canonical row deviation | `0.005` |
| Validation canonical token deviation | `validation.max_token_deviation`, default `0.005` |

At `f = 0.2`, this means less than 0.5% relative error around a 20% target, not
0.5 percentage points. The summary also reports each partition's deviation from
its own target. The tokenizer/revision used for balancing is independently
configured under [validation](configuration.md#data-and-validation).

Each source ID is saved once in its original A/B order; preprocessing does not
create swapped counterparts. Only after balance and integrity checks succeed
is the compressed Parquet published. Failure leaves the previous dataset in place.

## Processed data

The repository-backed logical path is `data/processed/train_validation.parquet`.
Current presets select external training rows from the 220,492-row real
preference merge at `data/processed/external_merged/train_validation.parquet`.
The merge, its downsampled derivative, and the selection CSV are stored outside
Git. Keep `data.processed_path`
pointing to the logical Parquet path whether it is a single file or multipart:

- Below 100 MiB: the Zstandard-compressed `.parquet` file.
- At least 100 MiB: `.parquet.manifest.json` plus `.parquet.parts/`, each volume
  at most 99 MiB.

Loaders reconstruct and checksum the artifact automatically in the OS temporary
directory (`tempfile.gettempdir()`), leaving no `.cache-work/` or `.backup-*`
directories in the dataset folder. The data fingerprint hashes reconstructed Parquet
bytes, independent of storage layout. Keep the manifest and all referenced
volumes together. Corrupt multipart data is archived and requires preprocessing
again. See [shared storage](caching.md#shared-storage-and-multipart-files) for
locking, publication, and temporary disk-space requirements.

Source columns are retained, including the corrected one-hot winner fields,
with these additions:

| Column | Meaning |
| --- | --- |
| `label` | `0` = A wins, `1` = B wins, `2` = tie |
| `group_id` | SHA-256 of canonical prompt JSON |
| `split` | `train` or `validation` |
| `is_swapped` | Always `False`; retained for model batch and result metadata |

Each retained ID has exactly one original row. Kaggle preparation preserves all
original IDs; external integration applies the cleaning and voting rules below.
Input loading rejects missing values, invalid labels, nonboolean
`is_swapped` values, swapped rows, duplicate IDs, and prompt groups crossing
partitions. Older augmented datasets must have their `is_swapped=True` rows
removed before loading.

## Merged external preferences

The current merge combines Kaggle, LMSYS 33K, Arena human, and Arena expert.
Data is constructed using the
[adopted processing rules](recommended-external-data.md#已确定的处理原则).
All targets are hard labels with integer one-hot winner fields. Rows have equal
training weight. Current presets use [ranked external selection](#ranked-external-training-selection)
and temporarily disable both
[scope-specific synthetic ties](#scope-specific-synthetic-ties)
and [tie duplication](#tie-duplication) (`train_supplement_path: null`,
`build_supplement_path: null`).

Rebuild the pinned source set with:

```powershell
uv run python data/merge_external_data.py --config config.yaml
```

`--baseline-path` defaults to `data/processed/lmsys33k/train_validation.parquet`;
`--output-dir` defaults to `data/processed/external_merged`. The baseline's
57,477 Kaggle rows, including all 11,495 frozen validation IDs, are copied with
their existing fields and membership. LMSYS is rebuilt from its complete raw
33,000-row release rather than the old prompt-deduplicated subset. All external
rows enter `split="train"` with `is_swapped=False`.

The downloader pins official repository commits, verifies official LFS SHA-256
checksums, and retains source files and `sources.json` under `data/raw/external/`.
Initial downloads need network access and offline mode disabled; complete locked
inputs can be reused offline. Token counting uses the pinned SR2 tokenizer,
official template, and shared token cache without truncation. Preparation reuses
existing full-encoding lengths and persists new exact counts in a compact length
cache, avoiding a second complete token-array copy on disk. Samples above the
old 13,866-token ceiling remain eligible. After vote merging, a battle is excluded
if either complete branch exceeds the pinned model configuration's
`max_position_embeddings` (40,960 tokens). The limit applies to each branch,
not their sum; excluded conversations are saved intact without truncation.
All current preset model and warmup `max_length` values are 40,960.
Only the model configuration and tokenizer are loaded; no GPU or model weights
are required.

Kaggle IDs stay unchanged. LMSYS, Arena human, and Arena expert use
ID segments `1`, `2`, and `3` respectively: `segment * 2**32 + source_row_index`.
Provenance includes
`source`, `source_id`, `source_row_index`, `source_split`, `source_winner`, and
`vote_id`; `original_label` and `tie_corrected` record conversion before vote merging.

The output directory contains the logical Parquet and its storage components,
unchanged `validation_ids.csv`, pinned `sources.json`, and these audits:

| Artifact | Contents |
| --- | --- |
| `row_audit.parquet` | One disposition for every raw external input row |
| `conversion_audit.parquet` | Text/role/history parsing, original and corrected labels, exclusions |
| `dedup_audit.parquet` | Battle and vote identities, A/B direction, vote participation, retained IDs |
| `vote_counts.parquet` | Independent vote counts in representative A/B order and the final hard label |
| `token_stats.parquet` | Complete branch lengths for Kaggle and candidate external representatives |
| `excluded_model_context.parquet` | Complete rows excluded because either branch exceeds the pinned model context |
| `summary.json`, `checksums.sha256` | Counts, identities, integrity checks, and file hashes |

Publication is staged and read back through `TrainingInputLoader`; failed builds
retain their diagnostics and preserve any prior successful dataset. Current
presets set `validation.validation_fraction` to `11495 / 220492`
(`0.05213341073599042`) and leave both supplement paths
`null`. Use this dedicated merger to rebuild the combined dataset; `init` and
`preprocess` rebuild a Kaggle-only split.

The 2026-11-17 context-filtered build contains 220,492 rows: 208,997 train and
the frozen 11,495 validation rows (tie share `63168 / 220492`, 28.6487%; train tie share `59610 / 208997`, 28.5219%). Its full-table validation fraction is
`0.05213341073599042`. It excludes 102 over-context training battles (78 Arena
human and 24 Arena expert) and retains 1,584 rows above the old token ceiling.
The maximum retained complete branch is 40,953 tokens; the maximum A+B total
is 78,827, below the current 147,456-token joint training budget. All Kaggle
rows and validation IDs are unchanged.

| Representative source | Train rows | Validation rows |
| --- | ---: | ---: |
| Kaggle | 45,982 | 11,495 |
| LMSYS | 24,227 | 0 |
| Arena human | 135,440 | 0 |
| Arena expert | 3,348 | 0 |

Source counts identify retained representatives; cross-source votes remain in
the audits and receive equal weight during label selection. The build excludes
17 battles with tied highest vote counts, 12 unalignable Arena human records,
and one non-text expert record. All source rows are accounted for in
`row_audit.parquet`, including duplicate votes and validation/Kaggle overlaps.

External-data integration deduplicates battles by complete prompt context and
response pair, including A/B-swapped duplicates. Retain all distinct response
pairs for the same prompt, including external samples sharing a prompt with
Kaggle **training** rows. External samples sharing a prompt with the frozen
Kaggle **validation** partition are excluded. `group_id` remains the prompt
grouping and validation-isolation key; it is not sufficient to identify a
duplicate training sample. See the [adopted processing rules](recommended-external-data.md#已确定的处理原则).

Only Kaggle takes source precedence for duplicate battles. LMSYS and all other
external datasets have equal standing, regardless of whether they are already
in the saved training table or carry expert annotations.

External conversion workflows use the public preprocessing API after converting
source conversations into the project's JSON-array fields:

```python
from llm_classification_finetuning.preprocess import ExternalBattleDeduplicator

result = ExternalBattleDeduplicator().deduplicate(
    external_rows, existing_rows, merge_votes=True
)
retained_rows = result.data
dedup_audit = result.audit
vote_counts = result.vote_counts
```

Both inputs need `id`, `prompt`, `response_a`, and `response_b`; `existing_rows`
also needs its saved `split` values. Vote merging additionally requires `label`
and `vote_id` on external rows. Pass only original Kaggle rows from both saved
partitions as `existing_rows`. Validation prompt exclusion precedes battle
matching. Independent votes are counted once in a common A/B direction; the
unique highest-count class becomes the final hard label. Highest-count ties
are excluded, while successful groups retain the smallest-ID representative's
content and orientation. Conflicting copies of one vote stop publication.
Statuses include `retained`, `merged_vote`, `duplicate_vote`, `duplicate_training`,
`excluded_validation_prompt`, `excluded_vote_tie`, and `excluded_model_context`.
Token statistics retain every candidate with `retained` and
`exceeds_model_context` flags. Length filtering follows voting, preserving
retained vote outcomes; `dedup_status` records the original deduplication status.
No row is excluded merely for exceeding the old length ceiling.
Conversation identity normalizes JSON encoding
without trimming text or changing case, and compares complete response arrays
independently of A/B order.

To apply the adopted source policy with this API, pass only Kaggle rows as
`existing_rows`, preserving their frozen split values, and combine LMSYS with
the other external datasets in `external_rows`. Passing the entire saved
Kaggle + LMSYS table as `existing_rows` would incorrectly give LMSYS precedence
over other external sources.

The optional `merge_votes=False` default retains the older first-occurrence API
for callers that only require battle filtering. The dedicated merger always
enables vote merging. Historical Kaggle/LMSYS snapshots remain available.

After any real-data rebuild, update `validation.validation_fraction` in
`config.yaml` and all complete presets to `11495 / total_rows`, then regenerate
both supplements with the command below. Their source fingerprints must match
the newly published real table.

### Ranked external training selection

The root and numbered presets set `data.external_rank_path` to
`data/processed/external_english_rarity_rank.csv`. The CSV supplies
`rank`, `id`, `source`, and `label`. The loader joins by ID to retained
external training rows in the merged Parquet, which supplies the conversations.
Each scope uses the sum of complete A and B branch tokens across its Kaggle
training rows as the external token budget. For `n` external rows, the number
of ties (`label=2`) is `n × Kaggle tie share`, rounded to the nearest integer
with halves rounded up. The remaining rows are A/B wins (`label=0` or `1`).
Each group contributes its first eligible rows in ascending `rank`. Selection
picks the `n` whose token total is closest to the budget, choosing the smaller
`n` when the differences are equal. Preparation stops if the eligible ranked
rows cannot reach the budget at that tie share or if an eligible ID's source
or label differs from the processed dataset.

Token lengths include both serialized chat branches, without padding or
truncation. The loader reuses `token_stats.parquet` when the merge summary
matches the saved dataset, tokenizer, and chat format. Otherwise it counts
tokens with the active model tokenizer and the shared length cache.

| Scope | Kaggle rows | Kaggle tokens | External rows | External ties | External tokens | Total training rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation training and standalone warmup | 45,982 | 35,856,134 | 6,658 | 2,060 | 35,865,470 | 52,640 |
| Full-data builds and their inline warmup | 57,477 | 44,873,727 | 8,934 | 2,765 | 44,865,610 | 66,411 |

Validation and standalone warmup evaluate on the same 11,495 frozen Kaggle
rows. Full-data builds use those rows for training. Selection runs before
building the final model inputs. The configured validation fraction describes
the full saved split. Set `data.external_rank_path: null` to train on
the complete saved partition. Optional tie supplements are applied separately
when configured.

### External non-tie downsampling

Executed on 2026-11-18 and published to
`data/processed/external_merged_downsampled/` (`summary.json`; historical snapshot not
retained in the current workspace). The figures below match the
[220,492-row real merge](../data/processed/external_merged/summary.json)
(reconstructed SHA-256
`9c1f041dc3cdfc63a8a924b9433d124da4905d93d5e40e4e03bcc03eeb993bc6`, 63,168 ties).
The target is the project's corrected Kaggle tie share of `17788 / 57477`.
Rounding the deletion count down with
`floor(220492 - 63168 / (17788 / 57477)) = 16382` yields **204,110** rows
(**192,615** train, **11,495** validation; reconstructed SHA-256
`c0e4ca20ef4d1c0839794acd54aa20db36a76ef57d4d250aed5bf54c85d1c618`) and a tie
share of `63168 / 204110` (30.94802%).

Use every original external training row with `label` 0 or 1 as a deletion
candidate. This covers Arena human, LMSYS, and Arena expert. Compute all
frequencies once from the full real table before filtering, including ties and
both splits. Count prompts by `group_id`. For branch frequency, use the complete
prompt and that side's full response array as a single identity, with the same
JSON normalization as battle deduplication. Pool A and B in the same counter
and sum the two branch counts for each candidate. Join the retained rows in
[`token_stats.parquet`](../data/processed/external_merged/token_stats.parquet)
by `id` for `turns`, `tokens_a`, and `tokens_b`.

Sort candidates by the following keys in order, using each key to break ties
in the previous keys.

1. Prompt frequency, highest first.
2. Sum of the two branch frequencies, highest first.
3. Single-turn rows (`turns == 1`) before multi-turn rows.
4. `tokens_a + tokens_b`, highest first.
5. Numeric `id`, lowest first, for deterministic selection.

Take the first 16,382 IDs. The simulation selects all 8,999 eligible rows whose
prompt occurs more than once. The other 7,383 rows have unique prompts and two
distinct branches with a combined frequency of 2. They are all single-turn
rows with at least 3,489 A+B tokens. The deletion totals by source are 12,270
Arena human, 3,365 LMSYS, and 747 Arena expert. By label, they are 8,107 A wins
and 8,275 B wins.

The simulation removes 7.43% of rows and 10.59% of A+B tokens. Median A+B length
falls from 1,129 to 1,091 tokens, P90 from 5,364 to 5,088, and P95 from 8,547 to
about 8,467. The multi-turn share rises from 24.79% to 26.72%.

| Longer branch (tokens) | Rows before | Rows after | Share removed | Tie share before → after |
| --- | ---: | ---: | ---: | ---: |
| ≤128 | 18,431 | 16,890 | 8.36% | 38.11% → 41.59% |
| 129–512 | 70,719 | 67,165 | 5.03% | 29.49% → 31.05% |
| 513–1,024 | 46,307 | 44,613 | 3.66% | 26.88% → 27.90% |
| 1,025–2,048 | 43,587 | 41,711 | 4.30% | 26.23% → 27.41% |
| 2,049–4,096 | 25,715 | 19,892 | 22.64% | 26.54% → 34.31% |
| 4,097–8,192 | 10,946 | 9,430 | 13.85% | 28.69% → 33.30% |
| >8,192 | 4,787 | 4,409 | 7.90% | 30.21% → 32.80% |

In the single-turn part of the 2,049–4,096-token band, the selection removes
5,806 of 8,348 non-ties (69.55%). The tie share rises from 24.84% to 52.05%,
which could make the model more likely to predict ties in that band. The
Arena-human-only simulation removes 84.20% of the same group's non-ties and
raises its tie share to 67.66%.

The execution recorded the source fingerprint (`9c1f041dc3cdfc63a8a924b9433d124da4905d93d5e40e4e03bcc03eeb993bc6`)
and all 16,382 ranked deletion IDs with their sort keys in
`data/processed/external_merged_downsampled/ranked_deletions.parquet`,
published the filtered rows in their original order to
`data/processed/external_merged_downsampled/`, and verified 192,615 training
rows, the same 11,495 frozen Kaggle validation IDs, all 57,477 original Kaggle
rows and 63,168 ties, and every expected source and length count above via
`TrainingInputLoader` and `summary.json` / `checksums.sha256`. The root and
numbered presets (`config.yaml` and `configs/1.yaml`–`7.yaml`) initially used
`data/processed/external_merged_downsampled/train_validation.parquet` with
`validation.validation_fraction: 0.05631767184361374` (`11495 / 204110`),
`data.train_supplement_path: null`, and `data.build_supplement_path: null`. The
presets now select ranked training rows from the full merge. The existing `external_merged` duplicate
supplement remains bound to the 220,492-row input table's fingerprint.

## Tie duplication

To reinforce real distinct-response tie examples during holdout validation
and standalone warmup without modifying the audited real-data table or touching
the frozen validation set, `data/generate_tie_duplicate.py` generates an
independent supplement containing one duplicate per selected eligible tie row
(`label=2`, `is_swapped=False`, `response_a != response_b`) with fresh unique
`id`s starting after the maximum real ID:

```powershell
uv run python data/generate_tie_duplicate.py --config config.yaml --scope all
```

`--scope train|build|all` defaults to `all`, generating every configured scope,
and `--target-count` defaults to `7343`.
Current presets temporarily set both `data.train_supplement_path` and
`data.build_supplement_path` to `null`. The counts below use the 220,492-row
`external_merged` snapshot with ranked selection disabled and `data.train_supplement_path` set to
`data/tie_duplicate/external_merged/train/tie_duplicates.parquet` and
`data.build_supplement_path: null`:

- `train` scope selects from all **59,188** distinct-response `split="train"`
  tie rows across **55,786** unique prompts (excluding **422** identical-A/B
  training tie rows out of 59,610 total), keeps at most one row per prompt
  (`group_id`), and picks the first **7,343** rows ordered by total prompt
  occurrences in `split="train"` ascending, multi-turn conversations first,
  complete `tokens_a + tokens_b` ascending (reusing pinned
  `token_stats.parquet` or `TokenizationCache`), and original `id` ascending.
- On the current snapshot, all **7,343** selected rows have `prompt_count = 1`
  and are multi-turn conversations (`tokens_a + tokens_b` from **82** to
  **3,474**, mean **1,681.85**; `arena_human` **5,008**, `kaggle` **1,517**,
  `lmsys` **788**, `arena_expert` **30**), increasing holdout/warmup training
  rows from **208,997** (59,610 ties, 28.52%) to **216,340** (66,953 ties,
  30.95%) while keeping the 11,495 validation rows untouched.
- `BuildSupplementLoader` validates schema metadata
  (`llm_classification_finetuning.row_duplicate`, `schema_version=2`,
  `kind="tie_duplicate"`), `source_sha256`, `max_per_prompt=1`, `target_count`,
  unique `group_id` values, `source_original_id` subset membership,
  validation-prompt isolation, and field parity with the real dataset.

## Scope-specific synthetic ties

The real-data merger preserves natural labels and counts. Current presets
temporarily exclude both synthetic ties and tie duplication
(`data.train_supplement_path: null`, `data.build_supplement_path: null`), and
the local synthetic-tie files under `data/synthetic_tie/external_merged/context40960/`
were removed. When generated from the complete real table, independent
self-response ties are sized to match the corrected
Kaggle-only ratio `17788/57477` (30.9480313865%) as closely as an integer
addition permits.

```powershell
uv run python data/generate_synthetic_tie.py --config config.yaml --scope all
```

`--scope train|build|all` defaults to `all`, generating every configured scope.
An explicit scope requires its corresponding path. The optional
`data.train_supplement_path` and `data.build_supplement_path` default to `null`.
When enabled, presets point to `data/synthetic_tie/external_merged/context40960/train/synthetic_ties.parquet`
and `data/synthetic_tie/external_merged/context40960/build/synthetic_ties.parquet` respectively;
both are currently `null`.
They are regenerated against the filtered real-data fingerprint and statistics.
They live in separate directories so staged publication cannot replace the other scope.
Staging directories inherit project permissions on Windows, preventing private
temporary-directory ACLs from blocking later sessions. The previous inaccessible
train/build directories are not used by current configurations.

| Scope | Real rows / ties | Added ties | Training rows / ties | Synthetic share of training | Resulting tie share |
| --- | --- | ---: | --- | ---: | ---: |
| Validation training and standalone warmup | 208,997 / 59,610 | 7,343 | 216,340 / 66,953 | 3.3941943237% | 30.9480447444% |
| Model build and inline build warmup | 220,492 / 63,168 | 7,342 | 227,834 / 70,510 | 3.2225216605% | 30.9479708911% |

Relative to the real rows before supplementation, the additions are 3.5134475614%
for train and 3.3298260254% for build.

For original count `N`, tie count `T` and target `p`, the ideal addition is
`(p*N - T)/(1-p)`. Compare the resulting tie-share errors of the adjacent
nonnegative integer counts using exact fractions; choose the smaller count on
an exact error tie. No additions are needed if the target is already met or exceeded.

Train candidates and historical win statistics use real `split="train"` rows
only, excluding frozen validation prompts. Build candidates use both real
partitions. Neither scope uses synthetic rows as source data. Group complete
normalized prompt JSON, exclude groups containing any real tie, and merge
identical normalized `(prompt, response)` candidates while retaining every source
occurrence. Historical scores are exact wins/occurrences fractions based on the
retained hard-labeled battles, with no additional vote or source weighting.

Each selection minimizes its prompt's current selected count, then complete
branch tokens, minimum source ID, and A before B. Counts start at zero.
Full-loss/full-win replacements require a strictly longer added branch;
when possible, the original single half-win adjustment completes the score.
If fractional histories prevent that step, choose an exact correction of one or
two same-prompt swaps. Prefer fewer swaps, then the smallest total token-length
change, then the existing candidate order. Apply negative score changes first
to avoid overshooting the target. Preserve the initially covered prompts and make the equal-weight
mean exactly `1/2`: score sums are `7343/2` for train and `3671` for build.
An infeasible adjustment fails with diagnostics rather than relaxing these rules.
Complete official-template lengths first reuse the real-data merger's statistics
when source, tokenizer/template identities and the statistics SHA-256 match.
Otherwise they use the shared content cache, without truncation.

A/B replies and model names both come from the minimum-ID source occurrence
(A wins a side tie). Each artifact assigns IDs from the maximum real ID plus one
in token/source order; artifacts are mutually exclusive. Train supplement rows
have `split="train"`; build rows have `split="build"`. Provenance records the
source row, side, original partition and dataset, historical counts and candidate ID.
Metadata schema v2 pins the real reconstructed SHA-256, scope, calculated count,
exact target and score sum, generation algorithm, and tokenizer/template identity.
The loader independently checks count, source membership, history and validation isolation.

Both SR2 profiles and standalone warmup select only the train supplement for
holdout training. Full-data builds select only the build supplement, including
inline warmup. The frozen 11,495-row validation set and the saved real table are
unchanged; `validation.validation_fraction` continues to describe the real table.
Supplement content enters training, batch-plan and warmup cache identities.
Matching content-based token and frozen-feature caches remain reusable.

Each directory also contains `candidates.parquet` (all occurrences in
`sources_json`), `selection.jsonl`, `replacements.jsonl`, `prompt_counts.csv`
(including zero selections), `summary.json`, and `checksums.sha256`. Files are
staged, read back and checked before publication. Large candidate audits stream
directly into bounded volumes; a failed run preserves its
diagnostics and the previous successful artifact. Large Parquet files follow
the shared manifest/parts policy. Only the supplement Parquet and its storage
components are required for training; retain the directory for the full audit.
Missing, corrupt, wrong-scope or source-mismatched configured supplements fail loading.

### Historical LMSYS33K supplement

The original build-only artifact lived at `data/synthetic_tie/lmsys33k/` (removed locally during the 2026-11-18 cleanup).
It has 1,012 ties for the 77,527-row snapshot, an exact score sum of `506`,
and a final tie share of `24306/78539` (30.9476820433%). Its schema v1 remains
loadable for its matching historical build, but cannot supply current train or
merged-data builds. It is not replaced or reused by the new generation command.

### Conversation serialization

SR2 uses `sr2-qwen3-chat-template-v1`. Each A/B branch becomes one complete list
of alternating `user` prompt and `assistant` response messages, rendered with
the configured tokenizer's `apply_chat_template(tokenize=False,
add_generation_prompt=False, continue_final_message=False)`. No system message
or tools are supplied. The complete output, including its trailing newline, is
encoded with `add_special_tokens=False`.

The [official SR2 template](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B/blob/main/chat_template.jinja)
uses ChatML role/control tokens and inserts a `<think>\n\n</think>\n\n` block before
the final assistant response when it has no reasoning content. This default
behavior is retained. Earlier assistant messages use the official template's
normal multi-turn handling. Missing or unusable templates fail explicitly.

Prompt and response values must be JSON arrays with the same nonzero turn count.
Prompts must be strings; responses may be strings or `null`. A null response is
an empty assistant message, retaining its position in the conversation. The
Parquet stores source arrays rather than materialized chat text; model
repositories render them with the same tokenizer used for encoding.

Current presets count tokens with `Skywork/Skywork-Reward-V2-Qwen3-4B` at revision
`fd958fef475f323f4e6b195930e3dd918485c668`. Switching from the old plain-text
serializer requires rebuilding the split to balance the new token counts; some
IDs can change partitions even when the fraction and random seed are unchanged.

SR2 rejects empty or over-limit branches and never truncates; see
[input limits](lora.md#hardware-and-input-limits). Historical frozen-model
behavior is recorded in [migration](migration.md#historical-frozen-model).

### Saved SR2 split (2026-10-27)

The default saved artifact was rebuilt from all 57,477 source IDs with the
official template above, validation fraction `0.2`, and seed `42`. It contains
114,954 augmented rows across 51,734 prompt groups, preserving source ID order.
Token balancing made nine whole-group exchanges. Compared with the previous
saved split, two original IDs moved to validation and two moved to training.

| Partition | Original rows | Augmented rows | Prompt groups | Canonical A+B tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 45,982 | 91,964 | 41,436 | 35,855,325 |
| Validation | 11,495 | 22,990 | 10,298 | 9,018,402 |

The validation row share is 19.999304% and token share is 20.097288%. Their
relative deviations from the 20% target are 0.003480% and 0.486438%, respectively,
both strictly below 0.5%. Before the tie correction below, the reconstructed
Parquet SHA-256 was
`2b06417150aea3d9320b0d40032feae154b6f072164fca09b05f58f64faf8f90`.
The previous artifact's SHA-256 was
`f2e9efb77dd2888609fb49c64b8b6884fd0deb05418e951151386667a64ab787`.
See [measured lengths](lora.md#measured-input-lengths) for the matching tokenizer
statistics and current preset ceiling.

#### Identical-response tie correction (2026-11-06)

The saved artifact was corrected in place for 27 original IDs whose complete
`response_a` and `response_b` fields match but whose labels were not tie: 22 train
IDs (10 A wins, 12 B wins) and five validation IDs (two A wins, three B wins).
Both orientations were corrected, changing 54 rows to `label=2` and one-hot
winner fields `(0, 0, 1)`.

The correction preserves all 114,954 rows, 57,477 original IDs, 51,734 prompt
groups, row order, column types, and train/validation membership. The row/token
statistics above remain unchanged. The raw CSV and historical model results are
unchanged. Full preprocessing now applies the correction before stratification,
so a future rebuild can assign different groups to the two partitions.

| Label | Augmented rows after correction |
| --- | ---: |
| `0` (A wins) | 39,689 |
| `1` (B wins) | 39,689 |
| `2` (tie) | 35,576 |

The reconstructed Parquet SHA-256 after that correction was
`f5273191b6b442e24d559835fe51e73e0fe5caeddf27700f9e04ab57ec643922`.

### A/B swap removal (2026-11-12)

All swapped rows were removed in place from both saved datasets. Original values,
row order, column types, prompt groups, and train/validation membership were
preserved. The repository-backed snapshot includes additions made after the
October baseline described above; this cleanup retained all of those original
rows without rebuilding either split.

| Dataset | Removed swapped rows | Remaining original rows | Train | Validation |
| --- | ---: | ---: | ---: | ---: |
| `data/processed/train_validation.parquet` | 59,533 | 59,533 | 47,626 | 11,907 |
| `data/processed/lmsys33k/train_validation.parquet` | 77,527 | 77,527 | 66,032 | 11,495 |

At that 2026-11-12 cleanup snapshot, both artifacts were below 100 MiB and used a single Parquet file with SHA-256 fingerprints:

- Repository-backed snapshot: `0591f630f6208081eeeffdd4776122ea3bb806a3aff7c9e0809e3eaedae38134`.
- Pre-supplement Kaggle + LMSYS snapshot (`77,527` rows): `0488ff2b22153bc5e1f05d0dad030906936ee0f3a8677e9ac646ebbb7ca33ecc`.

On 2026-11-19, `data/processed/lmsys33k/train_validation.parquet` was supplemented in place with 1,013 ranked real Arena ties to **78,540** total rows (**67,045** train, **11,495** validation; tie share 30.94856%), and is now stored in multipart form via `train_validation.parquet.manifest.json` (two `.part` files totaling `139,776,522` bytes; reconstructed SHA-256 `43eb5997d47e289babfec94f3732cee1762ca0dbd8392ed4dd1f3a9edea7b885`). See the [LMSYS33K data notes](../data/processed/lmsys33k/README.md).

Existing token and feature caches remain compatible when their branch content
and other identities match. SR2 already trained and predicted on original rows;
removing the stored counterparts does not change its training IDs or labels.

## Migration from five folds

Older fold configurations and datasets are not interpreted as a holdout.
See [migration](migration.md#five-folds-to-one-holdout) before regenerating data
or reporting historical runs. Continue with [model workflows](models.md).
