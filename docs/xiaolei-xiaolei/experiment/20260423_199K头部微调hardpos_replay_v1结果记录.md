# 20260423 199K 头部微调 hardpos replay v1 结果记录

## 1. 实验动机

前面的 continuation、output KD、replay-only、sample weighting 都说明了一件事：

- 一旦继续动整个 199K 学生 backbone，模型很容易漂
- 当前最好学生 `119.pt` 的主体表征未必差
- 真正更像是输出路径和 hard positive 的组织方式有问题

所以这版的核心假设是：

- 只动 HEAD，不再改 backbone
- 用 hard positive replay v2 数据，把输出层往正确路径上推

## 2. 实验配置

- 起点：`exp/fsmn_ctc_xlxl_distill_199k/119.pt`
- 数据：`data_xlxl_0327_ctc_v1_clean_hardpos_replay_v2`
- 训练范围：`finetune_trainable_scope=head_only`
- `finetune_epochs=10`
- `finetune_lr=1e-4`
- `num_average=10`
- 实验目录：`exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1`

## 3. 结果

### 3.1 静态 / 原始 test split

`avg_10.pt`

- `小雷小雷 = 98.20%`
- `小雷快拍 = 99.74%`

对比：

- 优于 `119.pt = 96.70% / 99.61%`
- 明显优于 replay-only v1/v2
- 明显优于 output KD / sample weighting
- 已经逼近老师 `159.pt = 99.25% / 99.87%`

### 3.2 全量 streaming

`test_infer_stream_avg10_chunk300`

- `小雷小雷 = 98.20%`
- `小雷快拍 = 99.74%`

和静态完全一致。

### 3.3 repeated hard case

`avg_10.pt`

- `0031 x3`：学生 `1/3`，只触发 `segment 1`
- `0021 x3`：学生 `3/3`

这说明这版第一次真正修复了部分部署相关 hard case，但还没全修完。

### 3.4 同一条 run 内部的 checkpoint 差异

- `avg_10.pt = 98.20% / 99.74%`
- `avg_5.pt = 98.20% / 99.74%`
- `129.pt = 98.13% / 99.74%`

静态上：

- `avg_10` 和 `avg_5` 持平
- `129.pt` 只比它们低 `0.07` 个点（`小雷小雷`）

但 hard case 上：

- `avg_10` 的 `0031 x3 = 1/3`
- `avg_5` 的 `0031 x3 = 1/3`
- `129.pt` 的 `0031 x3 = 2/3`
- `129.pt` 的 `0021 x3 = 3/3`

这说明更晚的单 checkpoint 对 hardcase 更友好，而平均模型更偏向静态整体最优。

## 4. 结论

这版是当前最成功的 199K 修复实验。

关键价值有两个：

- 首次把 199K 学生重新拉回到接近老师的静态精度
- 首次在 repeated hard case 上出现明显修复，而不是全面 `1/3`

但还没有完全结束：

- `0031 x3` 还没到 `3/3`
- `avg_10` 和 `129.pt` 之间已经出现“静态最优 vs hardcase 更优”的 tradeoff

所以下一步不该回到旧路线，而应继续沿 `head-only` 做第二版，重点看更晚 checkpoint 能不能继续修 `0031`，同时保持静态不明显回落。
