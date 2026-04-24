# 20260424 199K 前缀起步 hardpos replay v1 结果记录

## 1. 实验目的

这版接在 `head-only hardpos replay v2` 后面，目标是验证一个更窄的假设：

- `0031 x3` 仍卡在 `1/3`
- 学生常见失败形态是 `雷 小 雷`
- 因此只强化“前缀起步失败 / 不以第一个小开头”的 hard positives

如果这个假设成立，应该至少看到 `0031 x3` 从 `1/3` 推到 `2/3`，同时原始 test split 不明显掉点。

## 2. 数据与训练配置

prefix-start 子集挖掘结果：

- 输入 hard positives：`826`
- 选择规则：`greedy_not_start`
- 选中样本：`513`
- 主模式：`雷 小 雷 = 454`
- 训练集：`data_xlxl_0327_ctc_v1_clean_prefixstart_replay_v1`
- replay：`513 * 20 = 10260` 条额外 train rows

训练配置：

- 起点：`exp/fsmn_ctc_xlxl_distill_199k/119.pt`
- 训练范围：`finetune_trainable_scope=head_only`
- `finetune_epochs=10`
- `finetune_lr=1e-4`
- `num_average=10`
- 实验目录：`exp/fsmn_ctc_xlxl_distill_199k_prefixstart_replay_head_only_from119_v1`

## 3. 结果

### 3.1 原始 test split

`avg_10.pt`

- `小雷小雷 = 98.05%`
- `小雷快拍 = 99.74%`

同一 run 的 `129.pt`

- `小雷小雷 = 98.05%`
- `小雷快拍 = 99.74%`

对比当前最好学生 `head-only hardpos replay v1/v2 = 98.20% / 99.74%`，这版 `小雷小雷` 轻微回落。

### 3.2 全量 streaming

`test_infer_stream_avg10_chunk300`

- `小雷小雷 = 98.05%`
- `小雷快拍 = 99.74%`

静态和 streaming 完全一致，没有隐藏流式收益。

### 3.3 repeated hardcase

`avg_10.pt`

- `0031 x3`：学生 `1/3`，只触发 `segment 1`
- `0021 x3`：学生 `3/3`

`129.pt`

- `0031 x3`：学生 `1/3`

这说明 prefix-start replay 没有修掉最关键的 `0031`。

### 3.4 gap=300ms 诊断

为了排除“零间隔拼接导致 decoder reset 不充分”的可能，又对 `head-only hardpos replay v1` 做了 `0031 x3 + gap=300ms` 诊断：

- 老师：`3/3`
- 学生 `avg_10.pt`：`1/3`
- 学生 `129.pt`：`2/3`

结论：加静音间隔只能让 `129.pt` 有所改善，但仍不到老师的 `3/3`。所以问题不只是无间隔拼接边界，而是学生路径本身仍不稳。

## 4. 结论

这版判定为负收益实验：

- `小雷小雷` 从 `98.20%` 回落到 `98.05%`
- `小雷快拍` 维持 `99.74%`
- `0031 x3` 没有改善
- 更晚 checkpoint 也没有改善

下一步不继续做 prefix-start 子集 replay。更合理的方向是构造 repeated-sequence 训练样本：把 hard positive 音频拼成 `小雷小雷 x3`，标签也扩展为三遍 `小 雷 小 雷`，让 CTC 训练目标直接覆盖连续多次触发路径。
