# Kaggle 测试结果与配置差异记录（2026-11-19）

[Documentation index](../README.md) · [Kaggle 提交指南](kaggle-submission.md)

本文记录五次全量构建模型提交 Kaggle 竞赛的测试分数与对应配置差异。五次训练均基于 `data/processed/lmsys33k/train_validation.parquet`（全量 `78,540` 行，未启用额外排序筛选或补充数据集），采用 `Skywork/Skywork-Reward-V2-Qwen3-4B` 基座与 `bf16` / `sdpa` 精度设置，分别由 [l1.yaml](../configs/l1.yaml)、[l2.yaml](../configs/l2.yaml) 与 [l3.yaml](../configs/l3.yaml) 生成。

## 测试结果与差异汇总

| 模型 | 配置文件 | 关键差异配置 | Kaggle 分数 |
| --- | --- | --- | ---: |
| `SR2-Qwen3-4B` | [l2.yaml](../configs/l2.yaml) | `build_min_lr = 8.1e-05` | 0.91887 |
| `SR2-Qwen3-4B` | [l3.yaml](../configs/l3.yaml) | `build_min_lr = 9.411817e-05` | 0.92105 |
| `SR2-Qwen3-4B-lora` | [l1.yaml](../configs/l1.yaml) | `min_lr_ratio = 0.0`，`lr_decay_delay_epochs = 0`（默认） | 0.89724 |
| `SR2-Qwen3-4B-lora` | [l2.yaml](../configs/l2.yaml) | `min_lr_ratio = 0.25`，`lr_decay_delay_epochs = 0`（默认） | 0.88924 |
| `SR2-Qwen3-4B-lora` | [l3.yaml](../configs/l3.yaml) | `min_lr_ratio = 0.0`，`lr_decay_delay_epochs = 1` | 0.88950 |

## 配置细节对比

五组构建在公共超参数上保持一致：`max_length = 35130`、分类头 `hidden_size = 128`、`dropout = 0.1`、`learning_rate = 0.0001`、`weight_decay = 0.01`、`random_state = 42` 且未启用 label smoothing。不同之处主要集中在模型结构与学习率衰减策略两个维度。

在模型结构上，冻结方案 `SR2-Qwen3-4B` 仅训练 `655,745` 个分类头参数，按 `build_epochs = 33` 从 `0.0001` 进行指数衰减；最终最小学习率取 `8.1e-05`（`l2.yaml`）时取得 `0.91887`，优于维持更高末期学习率 `9.411817e-05`（`l3.yaml`）的 `0.92105`。

相比之下，LoRA 联合微调方案 `SR2-Qwen3-4B-lora` 将分数整体推进约 `0.03`。该方案拥有 `33,685,889` 个可训练参数（`lora_rank = 16`、`lora_alpha = 32`、`lora_dropout = 0.1`），先以固定 `0.0001` 学习率完成 `22` 个 epoch 的 head warmup，再进入 `2` 个 epoch 的余弦衰减联合训练。当第 1 个 epoch 起即直接衰减至 `0`（`l1.yaml`）时，测试分数为 `0.89724`；通过提高终值下限至 `2.5e-05`（`l2.yaml` 的 `min_lr_ratio = 0.25`）或将衰减延迟至第 2 个 epoch（`l3.yaml` 的 `lr_decay_delay_epochs = 1`），线上表现分别改善至 `0.88924` 与 `0.88950`。
