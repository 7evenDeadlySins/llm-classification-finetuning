# External Training Data Reference (Sources, Labels, and Integration Rules)

[Documentation index](../README.md) · [Candidate external datasets](external-training-data.md)

This document records the currently integrated external datasets from [external-training-data.md](external-training-data.md) — Arena human and Arena expert — along with the adopted processing rules and source notes. Integration does not imply proven competition gains.

The 2026-11-17 [source audit](data-source-audit-2026-11-17.md) identified significant distribution shifts and corrected two source assumptions: HelpSteer3's zero scores come from rounded annotator disagreement rather than direct Arena tie votes, and Arena expert filters prompts by topic rather than using credentialed expert voters or lower-noise labels. HelpSteer3 preference was subsequently removed.

## Adopted processing principles

**Data mixture and class proportions:** After cleaning, deduplication, and validation isolation, training selects the same number of rows and ties from the external English rarity ranking as the active Kaggle training scope. Ties and non-ties are selected separately in ascending `rank` order, with equal training weight per row. See [ranked external selection](data-preparation.md#ranked-external-training-selection).

* **Merged table preserves raw class ratios:** Presets draw from the [220,492-row real preference merge](data-preparation.md#merged-external-preferences) while keeping the 11,495 Kaggle validation rows frozen. Both supplement paths are set to `null`, disabling [tie duplication](data-preparation.md#tie-duplication) and [synthetic self-response ties](data-preparation.md#scope-specific-synthetic-ties). For the historical 204,110-row snapshot, see [external non-tie downsampling](data-preparation.md#external-non-tie-downsampling).
* **Synthetic tie adjustment (when enabled):** Preserves initial selections and full-win swaps. If `1/3` fractional histories prevent a final half-win swap, one or two exact-fraction same-prompt swaps are used instead (preferring fewer swaps, smaller total token-length change, and original candidate order; score-lowering steps run first while preserving prompt coverage, supplement count, and an exact `1/2` mean).

**Hard labels only:** No soft labels, sample weights, or weak-preference filtering are applied; each valid vote counts as one vote.

**Language and domain retention:** Non-English text is kept verbatim during merging under the same structural, deduplication, and validation-isolation checks, while training selection is restricted to the English ranking CSV. Complete branch lengths are counted without truncation: rows exceeding the old `13,866`-token ceiling but within the model's `max_position_embeddings=40960` are kept, and battles exceeding `40,960` on either branch are excluded and saved separately. Model and warmup `max_length` are set to `40960`.

* All Arena human categories (code, math, creative writing, instruction following, etc.) and all Arena expert `occupational_tags` are retained.

**Multimodal exclusion:** Any battle containing non-text content (e.g., images, audio) in either branch or in backfilled history is excluded in full rather than stripping media and keeping the original label. Pure-text content lists are concatenated in order; empty responses follow the rules below.

**Malformed structure and empty responses:** Unalignable records — unsupported roles (e.g., `system`), mismatched turn counts, or unrecoverable history — are dropped without rewriting roles or concatenating turns.

* **Single-side empty response:** Retained with its original hard label; the non-empty side is not auto-assigned as winner.
* **Both-sides empty or identical responses:** Final-turn empty responses on both sides, or identical `response_a == response_b` (without stripping whitespace), are corrected to `tie`, overriding the raw label.

**Complete context and battle deduplication:** Every valid vote retains its full conversation context; multi-turn or shared session history is not treated as a duplicate unless the actual battle repeats.

* Kaggle `prompt` and `response_a/b` arrays already store all turns; Arena follow-up votes (`evaluation_order > 1`) backfill prior history from `full_conversation`.
* `evaluation_order` tracks voting sequence within an `evaluation_session_id`, not turn count. Because models may be resampled after a vote, all valid evaluations in a session provide distinct training signal.
* Deduplication keys on complete prompt context and unordered A/B response pair (catching A/B swaps). Distinct response pairs for the same prompt are kept; `group_id` is only for prompt grouping and validation isolation.
* **Kaggle-only precedence:** When an external battle duplicates a Kaggle training row, Kaggle is kept. LMSYS, Arena human, and Arena expert have equal standing.

**Hard-vote merging across external duplicates:** Duplicate copies of the same raw vote count once; independent votes on the same battle are aligned to a common A/B direction and resolved by majority vote. Ties for the highest vote count drop the entire battle rather than converting it to `tie`. Retained battles emit one row with training weight `1`.

* Swapping A/B swaps A-win and B-win labels while leaving `tie` unchanged; identical or both-empty responses are corrected to `tie` before counting. All independent votes have equal weight.
* Example: two A-wins and one B-win retain A-win; one A-win and one tie drop the battle. The smallest internal ID supplies the content and A/B orientation without extra vote weight.
* Arena human/expert share `arena:{id}` vote identities; LMSYS uses `(question_id, judge, tstamp)`. Conflicting content or corrected labels under the same `vote_id` halt publication and save a conflict audit.
* External battles duplicating Kaggle training rows keep the Kaggle label without entering external voting; validation prompt isolation always applies first.

| Comparison scope | Condition | Action |
| --- | --- | --- |
| Within/across external datasets | Same prompt, different response pair | Keep all |
| External vs. Kaggle **train** | Same prompt, different response pair | Keep external sample |
| External vs. Kaggle **train** | Identical prompt and response pair (including A/B swap) | Keep Kaggle, drop external duplicate |
| Within/across external datasets (including LMSYS) | Identical prompt and response pair (including A/B swap) | Deduplicate identical `vote_id`; majority vote in aligned A/B order; drop highest-count ties; weight `1` |
| External vs. frozen Kaggle **validation** | Same prompt (`group_id`) | Exclude external sample to preserve prompt isolation |

## Integrated datasets

| Dataset | Size | Window / release | Coverage | Label mapping | License |
| --- | --- | --- | --- | --- | --- |
| [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | 135,634 rows | 2025-04-17 to 2025-07-24 | 126 languages (>50% English); 53 models | `winner`: `model_a` -> `0`, `model_b` -> `1`, `tie` / `both_bad` -> `2` | Prompts `CC-BY-4.0`; outputs per model terms |
| [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) | 5,128 rows | Released 2025 | 35 languages; 105 models; `occupational_tags` | `winner`: `model_a` -> `0`, `model_b` -> `1`, `tie` / `both_bad` -> `2` | Prompts `CC-BY-4.0`; outputs per model terms |

## Dataset-specific notes

### 1. `lmarena-ai/arena-human-preference-140k`

* **Distribution shift:** Uses the same 4-class Arena voting format, but 2025 models, languages, lengths, and multi-turn rates differ markedly from 2024 Kaggle data. See [source audit](data-source-audit-2026-11-17.md#2-training-mixture-vs-competition-data).
* **Content parsing & multimodal exclusion:** Message `content` in `conversation_a/b` is a list of dicts (e.g., `[{"type": "text", "text": ..., "image": null, "mimeType": null}]`). Pure-text entries are concatenated in order; non-text media anywhere in the battle or backfilled history excludes the row.
* **`evaluation_order > 1`:** `conversation_a/b` hold the current evaluation's conversation; prior turns live in `full_conversation` under `model_side_a` / `model_side_b`. Backfill preserves earlier turns (which may come from previously sampled models) and keeps the current vote. See the [dataset card](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k#overview).
* **Length checking:** Sequences up to 128 messages are tokenized with the pinned `Skywork/Skywork-Reward-V2-Qwen3-4B` tokenizer (`sr2-qwen3-chat-template-v1`); either branch exceeding `40960` tokens excludes the battle without truncation.

### 2. `lmarena-ai/arena-expert-5k`

* **Expert prompt tagging:** Uses DeepSeek-v3-0324 on prompts alone to assign `occupational_tags`; this does not imply credentialed expert annotators or lower label noise. See the [official blog](https://arena.ai/blog/arena-expert).
* **Key naming difference:** `full_conversation` uses `model_a` / `model_b` instead of `140k`'s `model_side_a` / `model_side_b`.
* **Source serialization:** Conversation fields are stringified NumPy `array(..., dtype=object)` representations (with empty lists as `array([], dtype=float64)` and occasional missing commas). The parser decodes only restricted container literals without `eval`; unparseable records or literal ellipses are excluded and audited.
* **Cross-dataset deduplication:** Overlaps in time and source with `140k`; shares `arena:{id}` vote identities and undergoes full context/pair deduplication, multimodal exclusion, and length filtering.

## Integration reminders

* **Validation isolation & IDs:** Keep the 11,495 IDs in [validation_ids.csv](../data/processed/lmsys33k/validation_ids.csv) frozen. Compute `group_id` (SHA-256 of canonical prompt JSON), exclude validation prompt matches, deduplicate battles, and write new rows to `split="train"` with `is_swapped=False`.
* **ID segmentation & config updates:**
  * Assign `id` outside Kaggle IDs and LMSYS (`2**32 + row_index`) using `segment * 2**32 + source_row_index`.
  * Update `validation.validation_fraction` in YAML to `11495 / total_rows` so [TrainingInputLoader](../src/llm_classification_finetuning/models/common/inputs.py) passes its `< 0.005` deviation check.
  * Regenerate any configured supplements after rebuilding the real merge. Do not run `uv run init` or `uv run preprocess` on merged datasets.
