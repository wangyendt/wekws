# 20260423 199K 头部微调 hardpos replay v2 实施方案

## 1. 已知结论

`head-only hardpos replay v1` 是当前为止最有效的 199K 路线：

- `avg_10.pt = 98.20% / 99.74%`
- `0021 x3 = 3/3`

但仍有两个遗留问题：

- `0031 x3` 在 `avg_10.pt` 上仍只有 `1/3`
- 同一条 run 的 `129.pt` 虽然静态略低，但 `0031 x3` 已提升到 `2/3`

这说明：

- 继续沿 `head-only` 是对的
- 更晚 checkpoint 更有 hardcase 修复能力
- 当前最值得验证的是“再延长一点点 head-only finetune，能不能把 `0031` 从 `2/3` 推到 `3/3`，同时静态仍保在 `98%+`”

## 2. v2 设计

- 数据：`data_xlxl_0327_ctc_v1_clean_hardpos_replay_v2`
- 起点：`exp/fsmn_ctc_xlxl_distill_199k/119.pt`
- 训练范围：`finetune_trainable_scope=head_only`
- `finetune_epochs=20`
- `finetune_lr=1e-4`
- `num_average=5`

实验目录：

- `exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v2`

## 3. 通过标准

满足任一条即可认为 v2 比 v1 更值得保留：

- 静态维持在 `小雷小雷 >= 98.0%` 且 `小雷快拍 >= 99.7%`
- `0031 x3` 从 `1/3` 或 `2/3` 提升到 `3/3`

如果静态明显跌回 `97.x`，或 hardcase 没再提升，则说明 v1 已经基本到头，下一步需要转向更结构化的目标约束，而不是继续延长 head-only。
