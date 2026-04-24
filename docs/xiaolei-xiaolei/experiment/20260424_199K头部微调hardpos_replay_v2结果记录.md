# 20260424 199K 头部微调 hardpos replay v2 结果记录

## 1. 实验动机

`head-only hardpos replay v1` 已经把 199K 学生明显拉回来了，但还剩一个核心悬念：

- `avg_10.pt` 静态已经到 `98.20% / 99.74%`
- `0021 x3` 已经修到 `3/3`
- 但 `0031 x3` 还只有 `1/3`
- 同一条 run 的 `129.pt` 对 `0031 x3` 已经到 `2/3`

所以这版想验证的是：

- 只把 `head-only` 的 finetune 再向后推一点
- 是否能在不掉静态精度的前提下，把 `0031` 再往上推

## 2. 实验配置

- 起点：`exp/fsmn_ctc_xlxl_distill_199k/119.pt`
- 数据：`data_xlxl_0327_ctc_v1_clean_hardpos_replay_v2`
- 训练范围：`finetune_trainable_scope=head_only`
- `finetune_epochs=20`
- `finetune_lr=1e-4`
- `num_average=5`
- 实验目录：`exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v2`

## 3. 结果

### 3.1 原始 test split

`avg_5.pt`

- `小雷小雷 = 98.20%`
- `小雷快拍 = 99.74%`

和 `v1 avg_10.pt` 完全一样，没有新增静态收益。

### 3.2 全量 streaming

`test_infer_stream_avg5_chunk300`

- `小雷小雷 = 98.20%`
- `小雷快拍 = 99.74%`

也和静态完全一致。

### 3.3 repeated hard case

`avg_5.pt`

- `0031 x3`：学生 `1/3`，只触发 `segment 1`
- `0021 x3`：学生 `3/3`

也就是：

- `0021` 继续保持修复
- `0031` 没有比 `v1 avg_10.pt` 更进一步

### 3.4 更晚 checkpoint 探针

`139.pt`

- 静态：`98.20% / 99.74%`
- `0031 x3 = 1/3`

这说明这版即使不做平均，也没有复现 `v1/129.pt` 那种“静态略低但 hardcase 更好”的 tradeoff。

## 4. 结论

这版可以明确判成“无新增收益”。

核心结论：

- 单纯把 `head-only` 从 `10` 拉到 `20` epoch，没有带来任何新的静态收益
- `0031` 也没有从 `1/3` 往上推
- `139.pt` 仍然没有好于 `avg_5.pt`

因此下一步不该继续做：

- 更长的 `head-only` epoch sweep
- 更晚 checkpoint sweep

更合理的方向是：

- 保留 `head-only` 这条正确主线
- 但把数据从“全部 hard positives”进一步收窄成“前缀起步失败 / 雷小雷塌缩”子集
- 用更定向的 replay 去打 `0031` 这类仍未修掉的模式
