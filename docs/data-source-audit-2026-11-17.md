# Training Data Source Audit (2026-11-17)

[Documentation index](../README.md) · [Current integration rules](recommended-external-data.md)

This audit examines the four frozen SR2 validation regressions against local real data, raw sources, synthetic supplements, and validation predictions. The added training data shows substantial distribution shifts relative to the competition set, with regressions concentrated in **distinct-response ties** and **multi-turn conversations**. The highest-priority suspects are the self-response tie supplement, 2025 Arena human (the largest training share), and HelpSteer3's preference-labeling semantics.

This audit inspects existing artifacts and sources only—no training, tests, or code/config/data changes were run. Causal attribution requires controlled ablation runs; the priorities below reflect observed differences rather than causal proof.

## 1. Consistent regression across the four latest runs

Compared against four November 15 frozen SR2 runs with matching `factor` values. Both groups used `swap_equivariant_abs_diff_v9`, original A/B orientation, the same 11,495 Kaggle validation rows, and identical frozen-head hyperparameters, with `max_length` raised from `13,866` to `40,960`. Because both the real-data path and train supplement changed, this comparison is not a single-source ablation.

| `factor` | Old result suffix | Old log loss | New result suffix | New log loss | Delta |
| --- | --- | ---: | --- | ---: | ---: |
| 0.95 | `20261115-205236` | 0.893858 | `20261117-061343` | 0.905146 | +0.011288 |
| 0.90 | `20261115-205332` | 0.893297 | `20261117-061644` | 0.905823 | +0.012527 |
| 0.85 | `20261115-205426` | 0.893130 | `20261117-061948` | 0.905861 | +0.012731 |
| 0.80 | `20261115-205527` | 0.893546 | `20261117-062258` | 0.905029 | +0.011484 |

All directory prefixes are `results/SR2-Qwen3-4B-`. Scores come from `metrics.json`; subset breakdowns are computed by joining `validation.csv` on ID (CSV float precision introduces ~`1e-8` rounding differences).

Mean log loss rose from **0.893457 to 0.905465** (+0.012007, ~+1.34%), and mean accuracy dropped from 58.3950% to 58.1187% (old training: 66,032 rows; new training: 251,415 rows).

| Validation subset | Rows | Mean log loss delta |
| --- | ---: | ---: |
| A wins | 4,010 | -0.012116 |
| B wins | 3,927 | -0.011386 |
| All ties | 3,558 | +0.065015 |
| Distinct-response ties | 3,516 | +0.067316 |
| Identical-response ties | 42 | -0.127612 |
| Single-turn | 9,932 | +0.006876 |
| Multi-turn | 1,563 | +0.044616 |
| Longer branch ≤512 tokens | 8,213 | +0.007549 |
| Longer branch 513–2,048 tokens | 3,118 | +0.023378 |
| Longer branch >2,048 tokens | 164 | +0.019094 |

Subsets may overlap. Mean tie recall fell from **42.8752% to 39.3479%**, while A-win and B-win recall rose from 65.2743%/65.4316% to 66.5212%/66.5457%. Mean predicted tie probability dropped from 30.6209% to 29.3926% (true validation tie share: 30.9526%).

Weighted by class share, A-win, B-win, and tie contributions to the overall loss delta are **-0.004227, -0.003890, and +0.020124** (sum: +0.012007). Multi-turn rows make up only 13.60% of validation but contribute ~+0.006067 (~50.5% of the total regression).

## 2. Training mixture vs. competition data

Figures below cover **real training rows** after cleaning, deduplication, and context filtering (excluding validation and synthetic supplements). Branch lengths use complete, untruncated counts from `token_stats.parquet` with the pinned SR2 tokenizer and chat template.

| Source | Train rows | Share of real train | Tie share | Multi-turn share | Mean branch tokens | Branch P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kaggle | 45,982 | 19.60% | 30.95% | 13.00% | 390 | 1,045 |
| LMSYS 33K | 24,227 | 10.32% | 29.30% | 12.25% | 266 | 677 |
| Arena human | 135,440 | 57.72% | 27.49% | 31.71% | 1,639 | 5,540 |
| Arena expert | 3,348 | 1.43% | 31.27% | 35.72% | 3,811 | 13,010 |
| HelpSteer3 preference | 25,657 | 10.93% | 5.60% | 48.29% | 1,070 | 3,094 |

Real training totals 234,654 rows and 61,047 ties (26.02%). Adding 16,761 self-response ties reduces Kaggle's row share to **18.29%** (vs. **69.64%** in the old 45,982/66,032 mix), with Arena human at **53.87%** and the synthetic supplement at **6.67%**. By A+B token volume across real training rows, Arena human accounts for **77.46%** and Kaggle **6.26%** (compute volume only; training loss weights all rows equally).

### Kaggle: target distribution and frozen baseline

Direct Chatbot Arena human preferences, including multi-turn sessions; model names are provided only in training. See the [Kaggle data page](https://www.kaggle.com/competitions/llm-classification-finetuning/data).

The frozen baseline has 57,477 rows (45,982 train, 11,495 validation), dominated by `gpt-4-1106-preview`, `gpt-3.5-turbo-0613`, `gpt-4-0613`, and `claude-2.1`. All fields and row order match the merged table exactly.

### LMSYS 33K: closest external source

33,000 Arena conversations collected **April–June 2023** across 20 models ([dataset card](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)). The 24,227 retained rows match Kaggle closely in length, turn count, and tie rate, with **76.70% of model occurrences** matching Kaggle model names exactly (without alias normalization).

Deduplication changed from prompt-level (20,050 rows) to full-battle-level (24,227 rows): 20,043 source IDs overlap with 0 label changes, 7 old IDs were dropped, and 4,184 new IDs were added.

### Arena human: largest source and largest shift

Collected **2025-04-17 to 2025-07-24**; 135,440 rows retained across 53 newer models (Claude 4, Gemini 2.5, Qwen3, o3, etc.) with **0 exact model-name overlap** against Kaggle.

Mean branch length is **4.20x** Kaggle's, and multi-turn share is 31.71%. Language tags show **52.48%** `en`, 10.20% `pl`, 6.83% `ru`, ~8.71% `und`, plus Chinese, German, etc. `is_code=True` covers 39,297/135,440 rows (29.01%). Surface formatting also shifts sharply: Markdown H1/H2 headers appear in 26.25% of Arena human pairs vs. 4.13% of Kaggle train, and code fences in 23.58% vs. 11.60%.

Additionally, **20.12%** of retained rows have `evaluation_order > 1`. Because Arena resamples models after each vote, backfilled `full_conversation` history can contain earlier responses from different models than the current vote ([dataset card](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)).

### Arena expert: prompt-filtered hard tasks

5,128 raw rows; 3,348 retained (1,753 were duplicate vote copies). Its 103 models have 0 exact overlap with Kaggle.

Arena Expert uses **DeepSeek-v3-0324** to tag **prompts** for expert-level reasoning or knowledge—it does not verify voter credentials or guarantee lower label noise ([official blog](https://arena.ai/blog/arena-expert), [dataset card](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k)). Previous wording claiming "high-quality expert annotations / lower noise" was removed from the reference docs.

Occupational tags (overlapping) include Software/IT (1,621; 48.42%), Math (924; 27.60%), and Natural/Social Sciences (775; 23.15%). Mean branch length is **9.77x** Kaggle's, with 56.54% `en` and 11.95% `zh`. At 1.43% of training rows, it is lower priority than Arena human for diagnosing the overall regression.

### HelpSteer3 preference: divergent objective and tie semantics

Not live Arena battles: prompts come from ShareGPT/WildChat, responses are regenerated from open models, and Scale AI / Translated annotators score preferences. Multi-turn samples share identical prior assistant history and compare only the final turn ([dataset card](https://huggingface.co/datasets/nvidia/HelpSteer3), [paper §2.1](https://arxiv.org/html/2505.11475v1#S2.SS1)).

**Key semantic mismatch at score `0`:** The paper averages three retained ratings and rounds—e.g., `[-1, -1, +1]` yields `overall_preference=0`—and explicitly **excludes these samples from reward-model training** because they reflect annotator disagreement rather than a direct tie choice. The paper also removes "Neither response is valid" items, whereas Arena keeps `both_bad` as `tie`.

Local checks:
- 14,789 exact vote duplicates were removed from 40,476 raw rows, leaving 25,657 rows.
- All **1,435 raw `overall_preference=0` rows** consist of opposing non-zero individual scores (±1, ±2, ±3; zero individual `0` ratings). Final ties total 1,437 (1,435 zero-scores + 2 identical-response corrections).
- 31.86% of retained rows have `abs(overall_preference)=1` but receive the same hard-label weight as strong preferences. Sign-to-A/B mapping matches the spec.
- Domains: General (11,802), Code (5,603), Multilingual (5,123), STEM (3,129); 1,087 of the 1,435 zero-scores are Multilingual.
- 48.29% are multi-turn with identical prior assistant turns on both branches (vs. only 64/5,976 multi-turn rows in Kaggle train).

## 3. Self-response ties restore total tie share but shift tie composition

The 16,761 added rows all have `response_a == response_b` and a mean branch length of **75.25 tokens** (median 66), drawn from Kaggle (5,602), Arena human (5,414), LMSYS (4,645), HelpSteer3 (1,091), and Arena expert (9).

| Training table | Distinct-response ties | Identical-response ties | Total tie share |
| --- | ---: | ---: | ---: |
| Old (66,032 rows) | 19,450 (29.46%) | 286 (0.43%) | 29.89% |
| New (251,415 rows) | 60,623 (24.11%) | 17,185 (6.84%) | 30.95% |

Identical-response ties account for **22.09% of all ties** in the new training set (vs. 1.45% in the old set and **1.18%**, 42/3,558, in frozen Kaggle validation), while the distinct-response tie share dropped from 29.46% to 24.11%.

```text
Total tie share matches competition: 30.95%
             ├─ Distinct-response ties: 24.11% of training rows
             └─ Identical-response ties: 6.84% of training rows

Observed validation deltas: identical-response ties improved; distinct-response ties regressed sharply
```

## 4. Verified integrity checks

- All 11 raw files match their `sources.json` SHA-256 checksums.
- All 57,477 Kaggle rows match in content, label, split, and order; all 11,495 validation IDs align exactly across predictions.
- Zero exact prompt-group overlap between frozen validation and either real external training rows or the train synthetic supplement.
- External voting merged 188,724 single-vote battles and 70 multi-vote battles (21 dropped for vote ties).
- All retained branches fit within 40,960 tokens (102 over-limit battles excluded; 1,584 rows above the old 13,866 ceiling retained, 0.68% of real training).
- Decoded CJK prompts in validation total 89 rows with a mean loss delta of `-0.000345`, ruling out Chinese prompts as the regression driver.

| Source | Pinned repository revision |
| --- | --- |
| `lmsys/chatbot_arena_conversations` | `1b6335d42a1d2c7e34870c905d03ab964f7f2bd8` |
| `lmarena-ai/arena-human-preference-140k` | `6322995ab34d7c2693e3f47dd13fa5caa0789a74` |
| `lmarena-ai/arena-expert-5k` | `171f77047d8d153c41b1f77047d8d153c41b1f83` |
| `nvidia/HelpSteer3` | `f6d145777bcbde96137596340fab89793acd1031` |

## 5. Recommended ablation order

Keep the frozen validation set, model, seed, and training settings fixed while changing one factor at a time, tracking overall log loss, distinct-response ties, and multi-turn rows:

1. **Isolate the self-response tie supplement:** Compare the same real merge with and without the 16,761 synthetic ties.
2. **Isolate Arena human:** Test removing the largest source (which drives model-era, language, length, and history shifts).
3. **Evaluate HelpSteer3 separately:** Re-evaluate hard-label sign mapping and zero-score-to-tie conversion against the competition objective.
4. **Evaluate Arena expert last:** Smallest training share (1.43%); verify whether hard/long prompts help.

## Evidence artifacts

- [Source and content measurements](data-source-audit-2026-11-17/measurements.json)
- [Validation prediction breakdown](data-source-audit-2026-11-17/validation_comparison.json)
- Merged data artifacts: `data/processed/external_merged/summary.json`, `token_stats.parquet`, `row_audit.parquet`, `vote_counts.parquet`; source lock: `data/raw/external/sources.json`
- Baseline and supplement: `data/processed/lmsys33k/train_validation.parquet`, `data/synthetic_tie/external_merged/context40960/train/synthetic_ties.parquet`
