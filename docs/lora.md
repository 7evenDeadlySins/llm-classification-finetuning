# SR2 LoRA training

[Documentation index](../README.md)

SR2 trains LoRA adapters and a three-class comparison head over `Skywork/Skywork-Reward-V2-Qwen3-4B` (`AutoModel`, omitting its scalar reward head) on one CUDA GPU without quantization. Each pair is serialized as separate A/B branches, pooled at the last real token, and trained on **original rows only**.

```shell
uv run model SR2-Qwen3-4B-lora
uv run model SR2-Qwen3-4B-lora --build --checkpoint-tag experiment
```

Requires `SR2_Qwen3_4B_lora` and `warmup` in YAML (namespace: `SR2_Qwen3_4B_lora` / `SR2_Qwen3_4B_lora_warmup`). See [commands](commands.md) and [configuration](configuration.md).

## Swap-equivariant head

For pooled vectors $A, B \in \mathbb{R}^D$ and classifier width $h$:

$$
\begin{aligned}
P_A &= \operatorname{GELU}(AL - BM + B_1), & P_B &= \operatorname{GELU}(BL - AM + B_1), \\
W_A &= P_A \odot m_P, & W_B &= P_B \odot m_P, \\
z_A &= W_A R, & z_B &= W_B R, \\
C_A &= \max(W_A, 0), & C_B &= \max(W_B, 0), \\
u_{\mathrm{eff}} &= s\operatorname{softplus}(u/s),\ s=0.01, &
z_{\mathrm{tie}} &= b_{\mathrm{tie}} + \bigl(C_A + C_B - t|C_A - C_B|\bigr) u_{\mathrm{eff}}.
\end{aligned}
$$

- **Parameters ($2Dh + 3h + 1$ trainable):** $L, M \in \mathbb{R}^{D \times h}$ (packed as one bias-free `Linear(D, 2h)`), $B_1 \in \mathbb{R}^h$ (`win_projection_bias`), $R \in \mathbb{R}^{h \times 1}$ (`win_readout`), raw $u$ (`[1, h]`), scalar `tie_bias`, and fixed FP32 buffer `t` (default $2\phi - 1 = \sqrt{5}$).
- **Initialization (`scaled_uniform_clamped_gelu_tie_v6`):** $L, M \sim U(-\sqrt{3}\sigma, \sqrt{3}\sigma)$ with $\sigma = 1/\sqrt{2D}$; $R$ with $\sigma = 0.1/\sqrt{h}$; $B_1 = 0$, `tie_bias` $= 0$; $U_0 \sim U(0.5\mu, 1.5\mu)$ with $\mu = 0.2/(3h)$ and $u = s\operatorname{invsoftplus}(U_0/s)$ where $\operatorname{invsoftplus}(x) = x + \log(-\operatorname{expm1}(-x))$.
- **Symmetry & precision:** $W_A, W_B$ share one dropout mask $m_P$ (disabled in eval). Swapping A/B exchanges $z_A, z_B$ and preserves $z_{\mathrm{tie}}$ (identical inputs yield $z_{\mathrm{tie}} = b_{\mathrm{tie}} + 2C_A u_{\mathrm{eff}}$). The head runs with `torch.autocast(..., enabled=False)`: in `bf16`, win projections cast inputs/weights to `bf16` and convert $z_A, z_B$ to FP32; the tie branch always runs in FP32, returning FP32 `[z_A, z_B, z_tie]`.
- **Markers:** Architecture `swap_equivariant_abs_diff_v10`, combination `cross_projected_clamped_win_sum_diff`, adapter/head schema v2, frozen-head schema v1, warmup schema v5. See [v10 migration](migration.md#sr2-v10-fixed-tie-coefficient).

## Hardware and input limits

- **No truncation:** Branches are right-padded independently without masks (causal attention protects real tokens; CPU-computed positions pool the last real token; KV cache disabled). Empty or `max_length`-exceeding branches fail explicitly.
- **Inputs:** Uses [SR2 chat-template serialization](data-preparation.md#conversation-serialization) (`add_special_tokens=False`, no system message).
- **Precision & attention:**

| Precision | Backbone / pooled CPU features | Trainable state and head computation |
| --- | --- | --- |
| `bf16` | BF16 residuals, checkpoint inputs, and features | FP32 params/gradients/AdamW moments; explicit BF16 win compute, FP32 tie/logits; PEFT input dtype upcasting disabled once after init |
| `fp16` | FP16 backbone; FP32 pooled features | FP32 head/state with `GradScaler` |
| `fp32` | FP32 | FP32 head/state |

RMSNorm variance, absolute values, Softplus, and final logits stay FP32. With `sdpa`, math fallback is disabled (only Flash, memory-efficient, and cuDNN kernels run; unsupported inputs fail explicitly).

## Measured input lengths

2026-10-27 SR2 Kaggle rebuild (`sr2-qwen3-chat-template-v1`, revision `fd958fef475f323f4e6b195930e3dd918485c668`, 57,477 original IDs): median branch **297**, P99 **2,231.94**, max branch **13,866** (ID `2995198009` A), canonical A+B total **44,873,727** tokens. Current presets use the [220,492-row external merge](data-preparation.md#merged-external-preferences) and set `max_length: 40960`. (Historical 2026-10-20 plain-text snapshot: median 283, P99 2,193, max 13,514, total 43,035,297.)

## Optimization and head warmup

| Stage | Trainable work | Validation / persistence |
| --- | --- | --- |
| Head warmup (cache miss) | Extract missing unique branches (adapter-free); train head only with epoch-based exponential decay (`warmup.learning_rate` → `warmup.build_min_lr` over `warmup.build_epochs`) | Eval full train (warmup-smoothed) & validation (unsmoothed) on shared features; save head/optimizer/RNG state |
| Head warmup (exact hit) | Restore head/optimizer/RNG | Inline LoRA skips all warmup work; standalone warmup recomputes both losses (`0` epochs) |
| Head warmup (fixed-rate lower-epoch hit) | Load/extract features, replay shuffle, restore state; train remaining epochs | Eval after each remaining epoch; save separate target cache |
| Joint training | Clear AdamW moments & step counters once at entry (preserving weights, param groups, `GradScaler`, RNG); train adapters + head with cosine decay | Fresh backbone extraction + head validation after each epoch (publishes final epoch) |

- **Updates & loss:** Warmup targets `warmup.training_batch_size` rows/update; joint targets `G = training_batch_size * gradient_accumulation_steps` rows/update. Short tails merge into the preceding full update; loss is normalized by actual update-group rows.
- **Counters:** `LoRATrainingHistory.executed_warmup_epochs` and `executed_joint_epochs` count only epochs executed in the current call (`ModelBuildResult.epochs` is their sum).

### Learning-rate decay

- **Head warmup & frozen SR2 builds:** For `N = build_epochs > 1`, `L = learning_rate`, `M = build_min_lr` (default `L / 5`): `lr = L * exp(-((ln(L) - ln(M)) / (N - 1)) * epoch_index)` for `epoch_index` in `0..N-1` (exact `L` at start, `M` at end; `N=1` uses `L`; `M=L` is fixed). Decaying caches require an exact target-epoch hit.
- **Frozen SR2 validation:** Uses `ReduceLROnPlateau` (`factor: 0.95`, `patience: 2`, `threshold=0`, `min_lr=0`, `eps=1e-8`) with staged `early_stopping` (`7`); see [scheduler rules](configuration.md#learning-rate-schedules).
- **LoRA joint training:** `U = max(1, rows // G)` updates/epoch, `N = epochs * U` total updates. Holds `learning_rate` for `H = min(lr_decay_delay_epochs, epochs) * U` updates, then decays via `lr = learning_rate * (min_lr_ratio + (1 - min_lr_ratio) * (1 + cos(pi * (i - H) / (N - H - 1))) / 2)` (`min_lr_ratio` default `0.125`). Advances once per consumed optimizer group (including FP16-skipped updates).

## Two-stage validation

Joint validation runs backbone extraction (`extraction_budget`) to temporary CPU features, then head inference (`head_inference_batch_size`), releasing features on return. Warmup evaluations reuse warmup's shared CPU features (`head_inference_batch_size` inline, `warmup.training_batch_size` standalone).

## Optional token-budget training batches

Joint micro-batch packing uses total padded A/B tokens `P = n * (L_A + L_B)`:

- **`gradient_checkpointing_threshold: 0`:** Fixed `training_batch_size` micro-batches, all gradient-checkpointed.
- **Positive threshold:** Compares three stable ascending sorts (`(A, B)`, `(B, A)`, `(A+B, A, B)`) divided into `G`-row groups, packing rows with `P <= threshold` via the better of suffix DP and best-fit (fewest micro-batches, then lowest `P`; ties keep best-fit). Oversized pairs run alone; batches with `P > threshold` use non-reentrant gradient checkpointing on both branches through backward. Base groups are cached; each epoch shuffles full groups (`random_state + joint_epoch`) and merges any short tail into the last group.
- **Second pass (`gradient_checkpointing_budget > 0`):** Repacks checkpointed singletons inside each finalized update under `P <= budget` (requires `budget > 2 * threshold` when `threshold > 0`, or `training_batch_size: 1` when `threshold: 0`). Never crosses update boundaries.

## Training progress

Three independent stages reset timers/counters at transitions and report `time: elapsed/estimated_total`:
1. **Warmup feature extraction** (~20% intervals): missing unique branch batches.
2. **Head warmup** (~20% intervals): `remaining_warmup_epochs * (W + T + V)` work units (skipped on exact inline hits; standalone hits run **Cached head evaluation** for `T + V` units).
3. **Joint training** (~10% intervals): `epochs * (J + E + V)` work units (`E = V = 0` on builds).

CPU prep, cache I/O, and publication are outside stage timers. See [shared progress](models.md#training-progress) and [timing metrics](reports.md#timing-metrics).

## Memory tuning

| OOM stage | Adjustment |
| --- | --- |
| Warmup extraction | Lower `warmup.extraction_budget` |
| Warmup head | Lower `warmup.training_batch_size` |
| Joint training (positive threshold) | Lower `gradient_checkpointing_threshold` and/or `gradient_checkpointing_budget` (or set threshold to `0`) |
| Joint training (threshold `0`) | Lower `training_batch_size`; adjust `gradient_accumulation_steps` |
| Joint-validation extraction | Lower `SR2_Qwen3_4B_lora.extraction_budget` |
| Head inference | Lower `head_inference_batch_size` |

Oversized single samples still run alone; never lower `max_length` to fix OOM.

## Checkpoints

`--build` prepares CPU inputs/plans and probes both output paths before CUDA setup, then atomically writes `models/SR2_Qwen3_4B_lora/adapter[_TAG].pt` (adapters + FP32 head + compatibility metadata, no base weights or optimizer state) after CPU readback validation.

### Merged backbone export

After saving `adapter[_TAG].pt`, build moves the backbone to CPU, runs `merge_and_unload(safe_merge=True)`, and exports `models/SR2_Qwen3_4B_lora/merged[_TAG]/` (plain `AutoModel` config, 5 GB-sharded Safetensors at build precision, tokenizer, `export_metadata.json` with `checkpoint_sha256`). Exports are validated locally in a staging directory and promoted with `.merged[_TAG]-backup` recovery. Export failures preserve the already-saved `adapter[_TAG].pt`.
