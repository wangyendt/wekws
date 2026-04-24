# 20260423 199K 头部微调 hardpos replay v1 实施方案

## 1. 背景

到当前为止，下面几条线都已经验证过，但都没有超过 `119.pt`：

- continuation / replay
- output KD
- hard positive replay
- sample weighting

其中最新结论最关键：

- `CTC only weighting` 的 `124.pt` 也只有 `93.03% / 98.82%`
- 说明问题不只是“训太久漂移”
- 更像是整网继续微调本身就容易破坏 `119.pt` 已有的表征和路径

## 2. 这一版的核心假设

当前最值得验证的是：

- backbone 已经学到的内容未必差
- 真正坏掉的更可能是输出层的路径组织能力
- 如果只动 HEAD，而不再动 backbone，可能更有机会保住 `119.pt` 的主体能力，同时把 hard positive 的输出路径往正确方向推一点

## 3. 实验设计

- 数据：`data_xlxl_0327_ctc_v1_clean_hardpos_replay_v2`
- 起点：`exp/fsmn_ctc_xlxl_distill_199k/119.pt`
- 训练范围：`finetune_trainable_scope=head_only`
- 训练家族：`replay-only`
- 训练时长：`finetune_epochs=10`
- 学习率：`finetune_lr=1e-4`
- 模型平均：`num_average=10`

实验目录：

- `exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1`

## 4. 通过标准

至少满足下面两条中的一条，才值得继续沿这条线加下一版：

- 原始 `test split` 明显优于 `hardpos_replay_from119_v2`
- repeated hard case 从 `1/3` 提升到至少 `2/3`

如果仍然只有 `93.x` 或 hard case 仍是 `1/3`，则说明“只动 HEAD”也不够，下一步要重新考虑更强的结构或目标约束，而不是继续在同一套路上微调超参。
