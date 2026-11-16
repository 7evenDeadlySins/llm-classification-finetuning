# 推荐外部训练数据（带原生平局的人类偏好数据）

[Documentation index](../README.md) · [完整候选外部数据集参考](external-training-data.md)

本文档从 [external-training-data.md](external-training-data.md) 中单独提取**符合本项目标准、优先推荐考虑**的 3 个真实人类偏好数据集（均包含原生 `tie` 与多轮对话），并附上**仅供参考的使用注意事项**。

## 已确定的处理原则

**数据混合与类别比例**：清洗、去重并完成验证集隔离后，全部真实样本合并训练，按行等权；不按来源额外加权或重采样，也不采用按来源划分的两阶段训练。Kaggle 的重复对局保留优先级仅用于去重，不赋予其更高训练权重。

* **真实数据保留自然比例，训练时补齐 tie**：真实合并表保留全部自然类别比例；训练时追加独立自回复平局，使 tie 比例最接近 Kaggle 的 `17788/57477`（约 `30.9480%`）。不额外设置类别权重或重采样，全部真实及合成训练行等权。
* **恢复旧合成规则（2026-11-17）**：重新计算并分别生成 train 与 build 补充集，不复用旧的 1,012 条产物。验证训练和独立 warmup 仅从真实 train 取候选及统计历史胜率，补 16,759 条；全量 build 及其内置 warmup 从全部真实数据取候选，补 16,758 条。两份补充集互斥加载，历史胜率均值均须精确为 `1/2`；验证集不追加合成行。配置分别使用 `data.train_supplement_path` 与 `data.build_supplement_path`，详见[生成规则](data-preparation.md#scope-specific-synthetic-ties)。

合成时保留原有初选与全胜替换；若历史分数中的 `1/3` 等比例使最后一次半胜替换不可行，允许用一次或两次同 prompt 的精确分数替换完成调整。优先使用更少的替换，再比较总 token 长度变化和原有候选顺序；先执行降低分数的步骤，始终保留初始 prompt 覆盖与补充条数，最终均值仍须精确为 `1/2`。

**全部使用硬标签**：以 `response1` 为 A、`response2` 为 B，HelpSteer3 的 `overall_preference < 0` 映射为 A 胜（`label=0`），`> 0` 映射为 B 胜（`label=1`），`0` 映射为平局（`label=2`）。不按偏好强度生成软标签、加权或删除弱偏好样本。每条原始记录的 `overall_preference` 计一票，不拆分 `individual_preference`。

**语言与领域全部保留**：不按自然语言、任务领域或职业标签筛选样本。非英文内容保留原文，直接使用现有 tokenizer 与训练流程，不翻译；仍执行结构校验、去重与验证集隔离。完整统计分支长度，超出旧 `13866` token 上限的样本同样保留且不截断；配置的 `max_length` 随最终数据的真实最大分支长度更新。

* HelpSteer3 的 General、STEM、Code、Multilingual 四个领域全部纳入；其 Code 域的 `language` 表示编程语言，不能据此当作非英文样本排除。
* Arena human 的代码、数学、创意写作、指令遵循等类别，以及 Arena expert 的全部 `occupational_tags` 均保留，不据此过滤。

**多模态样本整条排除**：任一分支的当前对话或需要回填的历史中包含图片、音频等非文本内容时，排除整条样本，不通过删除非文本部分后保留文本来沿用原标签。纯文本消息的内容列表按原顺序提取、拼接文本；空回复按下述规则处理，不因文本为空而直接删除。

**异常结构与空回复**：删除无法严格对齐为 `user` / `assistant` 交替轮次的样本，包括含不支持的 `system` 等角色、轮次数不一致或无法还原完整上下文的样本；不改写角色或强行拼接消息修复结构。

* **单侧空回复**：保留样本及原始硬标签，不自动将非空侧判为胜方。
* **双侧空回复**：当前末轮两侧回复都为空时判平；完整 `response_a == response_b` 的样本也统一判平，覆盖原始偏好标签。空字符串与 `null` 回复按空回复处理，不去除空白字符。

**每条有效投票保留其完整上下文；多轮或共享历史本身不算重复，实际重复的对局才去重。**

* Kaggle 的 `prompt`、`response_a/b` 数组直接保存各轮内容；Arena 的后续评估则需结合 `full_conversation` 补齐此前历史。
* `evaluation_order` 是同一 `evaluation_session_id` 内的投票顺序，不是对话轮数。投票后可能重新抽取模型，因此同一会话中的不同评估均可提供有效训练信号，不因 `evaluation_order > 1` 而删除，也不限定每会话只留一条。
* 去重依据完整 prompt 上下文和 A/B 回答对，并识别 A/B 互换的重复；同 prompt 下的不同回答对全部保留。`group_id` 仅用于 prompt 分组和验证隔离，不能单独作为训练样本去重键。
* **重复对局仅 Kaggle 优先**：外部样本与 Kaggle 训练样本重复时保留 Kaggle。LMSYS 与 Arena human、Arena expert、HelpSteer3 等其他外部数据同等地位，不因已接入训练集或专家标注身份而获得来源优先级。

**外部重复对局的硬标签合并**：同一原始投票的重复副本只计一次；同一对局的不同独立投票统一 A/B 方向后，选择得票最多的类别。最高票并列时删除整组对局，不将其改判为平局。最终每个保留对局只有一条训练样本，训练权重为 `1`，不随投票数增加。

* A/B 互换时同步交换 A 胜和 B 胜标签，平局不变；同回复及双侧空回复在计票前修正为平局。各独立投票等权，不按来源或专家身份加权。
* 例如，A 胜两票、B 胜一票时保留 A 胜；A 胜一票、平局一票时删除对局。选取最小内部 ID 的记录保存对局内容及 A/B 方向，不赋予该记录的标签额外票权。
* Arena human/expert 共享 `arena:{id}` 投票身份；LMSYS 使用 `question_id`、`judge`、`tstamp` 组合。HelpSteer3 缺少投票 ID，以完整原始记录的 SHA-256 识别完全相同的副本。同一投票身份的对局内容或修正后标签矛盾时停止发布，并保存冲突审计。
* 外部样本与 Kaggle 训练样本重复时仍只保留 Kaggle 原标签，不参与外部计票；验证集 prompt 隔离规则保持不变。

| 比较范围 | 条件 | 处理方式 |
| --- | --- | --- |
| 外部样本内部、不同外部数据集之间 | 同 prompt，回答对不同 | 全部保留 |
| 外部样本与 Kaggle **训练集** | 同 prompt，回答对不同 | 保留外部样本 |
| 外部样本与 Kaggle **训练集** | 完整 prompt 和回答对均相同，包括 A/B 互换 | 保留 Kaggle，删除重复外部样本 |
| 外部样本内部、不同外部数据集之间（含 LMSYS） | 完整 prompt 和回答对均相同，包括 A/B 互换 | 同一投票副本只计一次；独立投票统一方向后取最多票类别，最高票并列则删除；保留对局的权重为 `1` |
| 外部样本与冻结的 Kaggle **验证集** | prompt 相同 | 排除外部样本，保持验证集按 prompt 隔离 |

## 推荐数据集概览

| 数据集 | 规模 | 采集/发布时间 | 覆盖范围 | 标签映射 | 授权协议 |
| --- | --- | --- | --- | --- | --- |
| [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | 135,634 行 | 2025-04-17 至 2025-07-24 | 126 种语言（英文 >50%）；53 个模型 | `winner`: `model_a` -> `0`，`model_b` -> `1`，`tie` / `both_bad` -> `2` | Prompts `CC-BY-4.0`；输出受各模型提供商条款约束 |
| [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) | 5,128 行 | 2025 年发布 | 35 种语言；105 个模型；含 `occupational_tags` | `winner`: `model_a` -> `0`，`model_b` -> `1`，`tie` / `both_bad` -> `2` | Prompts `CC-BY-4.0`；输出受各模型提供商条款约束 |
| [`nvidia/HelpSteer3`](https://huggingface.co/datasets/nvidia/HelpSteer3) (`preference` split) | 40,476 行（train 38,459 / val 2,017） | 2025-05 发布 | General、STEM、Code、Multilingual | 负分 -> A 胜；正分 -> B 胜；零分 -> 平局，全部为硬标签 | `CC-BY-4.0` |

## 各自的使用注意事项（仅供参考）

### 1. `lmarena-ai/arena-human-preference-140k`

* **优先度最高**：采集时间晚于 2024 年 Kaggle 竞赛与 `Skywork-Reward-V2` 发布时间，原生 4 类标签与任务分布最贴合。
* **内容解析与多模态排除**：`conversation_a` / `conversation_b` 中每条消息的 `content` 为字典列表（如 `[{"type": "text", "text": ..., "image": null, "mimeType": null}]`）。纯文本内容按原顺序提取拼接；当前对话或需回填的历史含实际非文本内容时整条排除，空回复沿用上述已确定规则。
* **处理 `evaluation_order > 1`**：`conversation_a/b` 保存当前评估的对话，可能包含多轮；此前评估的历史保存在 `full_conversation`（模型侧键名为 `model_side_a` / `model_side_b`）。按上述已确定原则为每条有效投票补全对应的 A/B 上下文，保留当前评估的标签；历史回复可能来自此前评估的其他模型。字段定义见[官方数据集说明](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k#overview)。
* **完整长度统计**：序列最长可达 128 条消息，用固定版本的 `Skywork/Skywork-Reward-V2-Qwen3-4B` tokenizer（`sr2-qwen3-chat-template-v1`）统计分支长度。超限样本不排除、不截断，模型及 warmup 的 `max_length` 更新为成品的真实最大分支长度。

### 2. `lmarena-ai/arena-expert-5k`

* **高质量专家标注**：噪声低于普通众包投票，适合作为 `140k` 的补充。
* **字段键名差异**：整体结构与 `140k` 相似，但 `full_conversation` 内的模型侧键名为 `model_a` / `model_b`（而非 `140k` 的 `model_side_a` / `model_side_b`），解析脚本需分别兼容。
* **源文件表示**：对话字段为包含 NumPy `array(..., dtype=object)` 的字符串表示，空内容还可能表示为 `array([], dtype=float64)`，外层元素间可能省略逗号。转换器仅解析受限容器语法，逐个还原字符串字面量，不执行表达式、不替换回复文本；空内容数组正常保留，源表示中的实际省略号或无法完整解析的记录排除并审计。
* **跨集去重**：与 `arena-human-preference-140k` 存在来源及时间重叠，同时使用时应识别重复来源 `id`，并按完整上下文与回答对识别重复对局；`group_id` 用于 prompt 分组与验证隔离，不单独作为训练样本重复的依据。同样执行纯文本提取、整条多模态样本排除、上下文回填与完整长度统计。

### 3. `nvidia/HelpSteer3`（`preference` split）

* **非 Arena 分布补充**：由 Scale AI 与 Translated 专业标注（注意 Code 域仅含英文 Prompt，Multilingual 域不含 Code Prompt），与 Arena 众包风格存在一定分布差异；本轮按上述原则与其他来源合并、按行等权训练。
* **多轮上下文展开**：数据为 `context`（OpenAI 消息列表，1–79 条）+ `response1` + `response2`。转换时需将 `context` 中末轮之前的 `assistant` 回复同步填入 `response_a` 与 `response_b` 的前置轮次，最后一轮分别填入 `response1` 与 `response2`。
* **角色与空回复清洗**：本项目 [ConversationSerializer](../src/llm_classification_finetuning/models/common/data.py) 不支持 `system` 角色且要求严格交替的 `user`/`assistant` 轮次，无法严格对齐的样本直接排除，不改写角色。`response1` / `response2` 单侧为空时保留原硬标签；双侧为空时判平。
* **合并原始 split**：自带的 5% `validation` 不可混入本项目的验证集，若用于训练应全部归入 `split="train"`。

## 接入本项目时的通用提醒（仅供参考）

* **冻结验证集与去重**：保持 [validation_ids.csv](../data/processed/lmsys33k/validation_ids.csv) 的 11,495 条 Kaggle 验证 ID 不变；按规范化 prompt JSON 计算 `group_id`（SHA-256），排除与冻结验证集 prompt 重合的新增样本。其余新增样本按完整 prompt 上下文与回答对内部及跨外部数据集去重，再与现有 Kaggle **训练集**及 LMSYS 训练样本按相同依据去重。同 prompt 的不同回答对全部保留，包括外部样本与 Kaggle 训练集同 prompt 的情况；新增数据只写入 `split="train"` 且 `is_swapped=False`。
* **同回复判平与无截断检查**：将 `response_a == response_b` 或双侧空回复的样本修正为 `label=2` 及整数 one-hot winner 字段。所有保留样本经 `sr2-qwen3-chat-template-v1` 完整编码并记录两支长度，不按旧上限排除样本；回复文本为空不代表聊天模板编码后的分支没有 token。
* **ID 分段与配置联动**：
  * `id` 需与现有数据（Kaggle ID 及 LMSYS 的 `2**32 + row_index`）分段隔离，避免整数主键冲突。
  * 追加训练集行数后，[TrainingInputLoader](../src/llm_classification_finetuning/models/common/inputs.py) 会校验实际验证集占比与配置的偏差（`< 0.005`），需同步更新 YAML 中的 `validation.validation_fraction` 为 `11495 / 新总行数`。
  * 旧补充产物强绑定 `lmsys33k` 的文件 SHA-256 与 77,527 行基线。新合并数据使用独立生成的 train/build 补充集；重新合并真实数据后需重新生成补充集。不要对合并数据集运行 `uv run init` / `uv run preprocess`。
