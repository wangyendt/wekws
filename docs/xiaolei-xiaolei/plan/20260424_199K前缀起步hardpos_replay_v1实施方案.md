# 20260424 199K 前缀起步 hardpos replay v1 实施方案

## 1. 当前判断

到 `head-only hardpos replay v2` 为止，已经有两个结论比较稳定：

- `head-only` 是当前唯一真正有效的 199K 修复路线
- 但“把全部 hard positives 一起 replay”已经进入平台期

证据是：

- `v1 avg_10.pt = 98.20% / 99.74%`
- `v2 avg_5.pt = 98.20% / 99.74%`
- `v2 139.pt` 也没有比 `avg_5.pt` 更强
- `0031 x3` 仍卡在 `1/3`

这说明问题已经不再是：

- epoch 不够
- average checkpoint 方式不对

而更像是：

- 仍有一类 hard positive 的主失败模式没有被当前数据干预对准

## 2. 下一步假设

现有诊断一直在重复同一个信号：

- 学生容易塌成 `雷 小 雷`
- 或者虽然能打出后半段 `小/雷` 高峰，但前缀起步没立住

所以这版要验证：

- 只从 `826` 条 hard positives 里筛出“前缀起步失败 / 雷小雷塌缩”子集
- 再只对这类样本做更强 replay
- 同时继续保持 `head-only`

目标不是再抬整体召回，而是优先把 `0031` 这类仍未修掉的 prefix-start hardcase 推上去。

## 3. 数据策略

先对已有 hard positives 再做一次 student 诊断，再只保留更尖锐的 prefix-start 子集：

- 输入：`data_xlxl_0327_ctc_v1_clean_hardpos_replay_v2/hard_positive_rows.jsonl`
- 模型：`119.pt`
- 当前首选规则：
  - `greedy` 不以第一个 `小` 开头

原因：

- 用“任一前缀异常”去选会拿到 `701/826` 条，太宽
- 收紧到 `greedy_not_start` 后是 `513/826` 条，主模式几乎就是 `雷 小 雷`
- 这更接近 `0031` 当前还没修掉的失败形态

输出：

- `prefix-start` 子集 `rows.jsonl`
- 对应 `summary.json`
- 每条样本的 manifest，便于后续复盘

## 4. 训练配置

- 起点：`exp/fsmn_ctc_xlxl_distill_199k/119.pt`
- 数据：`data_xlxl_0327_ctc_v1_clean_prefixstart_replay_v1`
- 训练范围：`finetune_trainable_scope=head_only`
- `finetune_epochs=10`
- `finetune_lr=1e-4`
- `num_average=10`

实验目录：

- `exp/fsmn_ctc_xlxl_distill_199k_prefixstart_replay_head_only_from119_v1`

## 5. 通过标准

满足任一条即可认为值得保留：

- 原始 `test split` 维持 `小雷小雷 >= 98.0%` 且 `小雷快拍 >= 99.7%`
- `0031 x3` 从 `1/3` 提升到 `2/3` 或 `3/3`

如果静态显著掉点，或者 `0031` 仍然完全不动，则说明问题不只是 replay 样本选择，还需要更明确的路径约束。
