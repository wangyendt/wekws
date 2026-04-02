# 20260402 基于 clean 数据训练 2599 baseline 方案

## 1. 目标

基于当前全量 ASR 筛洗结果，先训练一版“只用可信数据”的 2599 维 baseline，目的不是一次性把数据清洗做到最终版，而是先回答一个更直接的问题：

- 当我们只保留当前最可信的一层样本时，baseline 训练效果会不会更稳定
- 相比原始 `data_xlxl_0327_ctc_v1`，仅去掉 ASR 明显不一致的样本后，模型是否会更干净
- 在不改模型头、不改词表、不改训练框架的前提下，单看数据清洗本身能带来多大收益

因此，这一版先刻意保持模型设置尽量不变，只替换训练数据。

## 2. 本轮使用哪些数据

### 2.1 数据来源

原始数据仍然是既有的 0327 数据集：

- `data_xlxl_0327_ctc_v1/train/data.list`
- `data_xlxl_0327_ctc_v1/dev/data.list`
- `data_xlxl_0327_ctc_v1/test/data.list`

ASR 筛洗结果在：

- `codex_artifacts/xlxl_asr_screen_train/`
- `codex_artifacts/xlxl_asr_screen_dev/`
- `codex_artifacts/xlxl_asr_screen_test/`

每个 split 都有：

- `clean.data.list`
- `ambiguous.data.list`
- `reject.data.list`

### 2.2 本轮“可信数据”的定义

本轮 baseline 只使用 `clean.data.list`，也就是：

- 正样本只保留 `clean_exact`
- 负样本只保留 `clean_negative`

不纳入训练的有两类：

- `ambiguous.*`
- `reject.*`

这样做的原因很简单：

- `ambiguous_other_target / ambiguous_target_overlap` 更像潜在错标
- `reject_positive_miss` 大量存在，需要人工听音才能进一步确认
- `ambiguous_extra_context` 更像切分问题，不适合在第一版“可信 baseline”里混进去

所以第一版 baseline 要尽量保守，优先提高标签纯度，而不是一开始就追求保留更多样本。

## 3. clean 数据规模

已经基于筛洗结果生成出新的可信数据目录：

- `data_xlxl_0327_ctc_v1_clean`

当前统计如下：

| split | 原始条数 | clean 条数 | 移除条数 | clean 时长(h) | clean 正样本 | clean 负样本 |
|---|---:|---:|---:|---:|---:|---:|
| train | 163560 | 159621 | 3939 | 122.1243 | 39933 | 119688 |
| dev | 9512 | 9309 | 203 | 7.1007 | 2196 | 7113 |
| test | 9096 | 8900 | 196 | 6.7099 | 2099 | 6801 |

可以看出，这一版并没有激进删数据，而是只去掉最不可信的那一小层：

- train 去掉 `3939` 条
- dev 去掉 `203` 条
- test 去掉 `196` 条

这个量级适合作为第一版 clean baseline。

## 4. 为什么这版仍然训练 2599 baseline

这次先不裁 head，不改成 top20，也不做蒸馏。原因是：

- 目标是验证“数据筛洗”本身的收益
- 如果同时改模型头、词表、训练配置，就很难判断收益到底来自数据还是模型结构变化
- 2599 baseline 是当前最稳定、最可对比的起点

因此这版训练保持：

- `num_keywords = 2599`
- `dict_dir = dict`
- `checkpoint_dict = dict`
- `checkpoint_strict = true`

也就是沿用原 baseline 体系，只把 `data_dir` 切到 `data_xlxl_0327_ctc_v1_clean`。

## 5. 本次需要跑的命令

以下命令默认都在目录 `examples/hi_xiaowen/s0` 下执行。

### 5.1 生成 clean 数据目录

如果你后面重新跑了 ASR 筛洗，或者想重新刷新 clean 数据集，执行：

```bash
python tools/build_xiaolei_screened_dataset.py --source-data-dir data_xlxl_0327_ctc_v1 --screen-root codex_artifacts --output-dir data_xlxl_0327_ctc_v1_clean --keep-source-manifest
```

### 5.2 校验 clean 数据目录

```bash
python tools/validate_xiaolei_ctc_data.py --data-dir data_xlxl_0327_ctc_v1_clean
```

### 5.3 查看 clean 数据规模摘要

```bash
python -c "import json; obj=json.load(open('data_xlxl_0327_ctc_v1_clean/summary.json', encoding='utf-8')); print(json.dumps({k:{'source_count':v['source']['count'], 'selected_count':v['selected']['count'], 'removed_count':v['removed_count'], 'selected_hours':v['selected']['duration_hours'], 'selected_pos':v['selected']['positives'], 'selected_neg':v['selected']['negatives']} for k,v in obj['splits'].items()}, ensure_ascii=False, indent=2))"
```

### 5.4 推荐训练命令：4 GPU 版

```bash
bash run_fsmn_ctc.sh --data_dir data_xlxl_0327_ctc_v1_clean --target_exp_dir exp/fsmn_ctc_xlxl_0327_clean_baseline_2599 --gpus 0,1,2,3 --dict_dir dict --checkpoint_dict dict --num_keywords 2599 --checkpoint_strict true 2 2
```

说明：

- 这里只跑 `stage 2 -> 2`，即直接训练
- 不重新做原始数据准备
- 不覆盖旧的 `exp/fsmn_ctc_baseline_4gpus`
- 新实验目录固定为 `exp/fsmn_ctc_xlxl_0327_clean_baseline_2599`

### 5.5 如果只有单卡，可用 1 GPU 版

```bash
bash run_fsmn_ctc.sh --data_dir data_xlxl_0327_ctc_v1_clean --target_exp_dir exp/fsmn_ctc_xlxl_0327_clean_baseline_2599_1gpu --gpus 0 --dict_dir dict --checkpoint_dict dict --num_keywords 2599 --checkpoint_strict true 2 2
```

### 5.6 训练完成后，如需直接跑 stage 3 评测

```bash
bash run_fsmn_ctc.sh --data_dir data_xlxl_0327_ctc_v1_clean --target_exp_dir exp/fsmn_ctc_xlxl_0327_clean_baseline_2599 --gpus 0 --dict_dir dict --checkpoint_dict dict --num_keywords 2599 --checkpoint_strict true 3 3
```

## 6. 这版 baseline 的判断标准

建议这版不要一上来就纠结最终指标是否已经最好，而是先看三件事：

- 训练过程是否更稳定，loss 曲线是否更干净
- dev/test 上是否比原始全量数据训练更好或至少不差
- 误检和漏检样本里，是否减少了明显的“脏标签带偏模型”的现象

如果这版 clean baseline 表现更好，下一步才值得继续讨论：

- 是否把部分 `ambiguous_extra_context` 加回去
- 是否在正样本侧引入人工复核通过的 `reject_positive_miss`
- 是否再往 top20 / distill 方向走

## 7. 当前结论

当前最合适的第一步不是继续手调模型，而是先跑这版：

- 数据：`data_xlxl_0327_ctc_v1_clean`
- 模型：原 2599 baseline
- 训练入口：`run_fsmn_ctc.sh`
- 核心变量只改一个：`--data_dir`

这样后面无论结果变好还是变差，结论都更清楚，因为变量控制得足够干净。
