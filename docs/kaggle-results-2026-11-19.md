# Kaggle Test Results and Configuration Differences (2026-11-19)

[Documentation index](../README.md) · [Kaggle submission guide](kaggle-submission.md)

This note records Kaggle test scores and configuration differences for five full-data builds trained on `data/processed/lmsys33k/train_validation.parquet` (all `78,540` rows, no extra rank selection or supplements) using `Skywork/Skywork-Reward-V2-Qwen3-4B` (`bf16` / `sdpa`) from [l1.yaml](../configs/l1.yaml), [l2.yaml](../configs/l2.yaml), and [l3.yaml](../configs/l3.yaml).

## Results summary

| Model | Config | Key difference | Kaggle score |
| --- | --- | --- | ---: |
| `SR2-Qwen3-4B` | [l2.yaml](../configs/l2.yaml) | `build_min_lr = 8.1e-05` | 0.91887 |
| `SR2-Qwen3-4B` | [l3.yaml](../configs/l3.yaml) | `build_min_lr = 9.411817e-05` | 0.92105 |
| `SR2-Qwen3-4B-lora` | [l1.yaml](../configs/l1.yaml) | `min_lr_ratio = 0.0`, `lr_decay_delay_epochs = 0` (default) | 0.89724 |
| `SR2-Qwen3-4B-lora` | [l2.yaml](../configs/l2.yaml) | `min_lr_ratio = 0.25`, `lr_decay_delay_epochs = 0` (default) | 0.88924 |
| `SR2-Qwen3-4B-lora` | [l3.yaml](../configs/l3.yaml) | `min_lr_ratio = 0.0`, `lr_decay_delay_epochs = 1` | 0.88950 |

## Configuration comparison

All five runs share `max_length = 35130`, head `hidden_size = 128`, `dropout = 0.1`, `learning_rate = 0.0001`, `weight_decay = 0.01`, `random_state = 42`, and `label_smoothing = 0.0`. Differences are limited to model architecture and learning-rate decay:

- **Frozen `SR2-Qwen3-4B` (`655,745` trainable head parameters):** Trains for `build_epochs = 33` with exponential decay from `0.0001`. Lowering the final rate to `8.1e-05` (`l2.yaml`) scored `0.91887`, outperforming `9.411817e-05` (`l3.yaml`) at `0.92105`.
- **Joint `SR2-Qwen3-4B-lora` (`33,685,889` trainable parameters; `lora_rank = 16`, `lora_alpha = 32`, `lora_dropout = 0.1`):** Improves scores by ~`0.03` over frozen runs. After `22` head warmup epochs at a fixed `0.0001` rate, it runs `2` cosine-decayed joint epochs. Decaying immediately from epoch 1 to `0` (`l1.yaml`) scored `0.89724`; raising the floor to `2.5e-05` (`l2.yaml`, `min_lr_ratio = 0.25`) or delaying decay to epoch 2 (`l3.yaml`, `lr_decay_delay_epochs = 1`) improved the score to `0.88924` and `0.88950`, respectively.
