---
name: KWS评估差原因定位
overview: 系统梳理为什么 `exp/fsmn_ctc_pinyin_403/test_32.pt/` 的 DET 指标远差于 baseline，并优先从打分/评估策略（`score_ctc.py`/`ctc_prefix_beam_search`/`compute_det_ctc.py`）入手定位与验证。
todos:
  - id: analyze-score-separation
    content: 用 `compute_det_ctc.py` 的 `keyword_filler_table` 统计正/负样本分数分布，证明是否存在严重重叠导致 FAR/FRR 失衡。
    status: completed
  - id: audit-beam-filter
    content: 定位 `score_ctc.py` + `ctc_prefix_beam_search` 的 token-set/topk/prob阈值过滤如何在 403 词表下放大误报。
    status: completed
  - id: ab-test-scoring-fixes
    content: 提出并A/B验证 1-2 个最小改动（取消/弱化 token-set 过滤、调 beam/阈值、加持续约束）来显著降低 FAR 同时控制 FRR。
    status: in_progress
  - id: fix-det-plot-and-units
    content: 修复/调整 `det.png` 绘图范围与 FRR 单位标注，避免“曲线空白/被裁剪”造成误判。
    status: in_progress
isProject: false
---

## 现象复盘（先把口径说清）

- 训练日志里你看到的高 `acc`（比如 `CV Batch 63/0 ... acc 83-85`）来自 [`wekws/utils/executor.py`](wekws/utils/executor.py) 调用 [`wekws/model/loss.py`](wekws/model/loss.py) 里的 `ctc_loss(... need_acc=True)`，本质是 **CTC 解码后的“序列识别准确率”**，不是 KWS 的 FAR/FRR。
- DET 评估来自 [`wekws/bin/score_ctc.py`](wekws/bin/score_ctc.py) 生成 `score.txt`（每条 utterance 给出 `detected/rejected` + 分数），再由 [`wekws/bin/compute_det_ctc.py`](wekws/bin/compute_det_ctc.py) 统计阈值曲线。两者优化目标不同，所以“某些 batch acc 高”并不必然带来“低 FAR/低 FRR”。

## baseline vs pinyin 的关键差异点（优先怀疑打分策略在小词表下更易误报）

- **输出词表规模差异**：baseline `output_dim=2599`（见 [`exp/fsmn_ctc_baseline/config.yaml`](examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline/config.yaml)），pinyin 实验 `output_dim=403`（见 [`exp/fsmn_ctc_pinyin_403/config.yaml`](examples/hi_xiaowen/s0/exp/fsmn_ctc_pinyin_403/config.yaml)）。
- **打分解码的“关键词 token 集过滤”**：`score_ctc.py` 调 `ctc_prefix_beam_search(score, ..., keywords_idxset)`（只让 beam search 看“关键词 token 集 + blank”）。在词表更小（403）时，负样本上“碰巧进 topk 的关键词 token”概率更高，更容易造成大量 `detected`，从而 **FAR 飙升**、阈值一抬又 **FRR 飙升**。
- 入口：[`wekws/bin/score_ctc.py`](wekws/bin/score_ctc.py) 中 `hyps = ctc_prefix_beam_search(..., keywords_idxset)`，以及用几何平均 `cand_score = prod_prob ** (1.0 / L)` 选 hit（见该文件 240 行后）。
- 过滤细节：[`wekws/model/loss.py`](wekws/model/loss.py) 的 `ctc_prefix_beam_search` 里只取 `topk(score_beam_size=3)`，并且还要求 `prob > 0.05`，再叠加 `idx in keywords_tokenset` 过滤（会放大“被迫用关键词 token 拟合负样本”的倾向）。

## 针对 `test_32.pt` 的定位验证步骤（不改训练，先用数据证明/证伪）

- **统计分数分布的可分性**（核心）：在 [`wekws/bin/compute_det_ctc.py`](wekws/bin/compute_det_ctc.py) 的 `keyword_filler_table` 里，分别拿到：
- `keyword_table`（正样本中被命中的分数）
- `filler_table`（负样本中误报的分数）
然后计算每个关键词的分位数/直方图/ROC-like 摘要，确认是否存在大量负样本分数与正样本重叠，导致“FAR 降下来就几乎全漏检”。
- **核对误报来源**：抽取 `filler_table` 中分数最高的若干条，回看它们在 `score.txt` 里对应的 `hit_keyword/hit_score`，确认是否是“过滤导致的伪匹配”。
- **同口径对比 baseline**：用同样的统计方式对 baseline 的 `score.txt`/stats（若有）做一遍，验证差异是否主要来自“词表规模 + token-set 限制”。

## 有针对性的改动与A/B验证（把 FAR/FRR 拉回正常形态）

- **改动 1（优先，最可能有效）**：给 `ctc_prefix_beam_search` 增加可配置项，弱化/取消 `keywords_tokenset` 过滤在 `score_ctc.py` 中的使用：
- 方案 A：先做全词表 beam search（不传 `keywords_tokenset`），拿到 `prefix_ids` 后再在结果序列里找关键词；
- 方案 B：保留 `keywords_tokenset`，但增大 `score_beam_size/path_beam_size`、降低或取消 `prob>0.05` 的硬阈值，并把这些参数暴露为 `score_ctc.py` CLI 参数，方便你快速 sweep。
- **改动 2（校准/鲁棒性）**：对 `hit_score` 做更稳健的归一化或加“持续帧数/跨度”约束（例如要求命中 token 的帧跨度 >= 某阈值），减少“瞬时尖峰”触发。
- **改动 3（展示修复，避免误导）**：`compute_det_ctc.py` 画图时当前 `xlim=5, ylim=35`，而你这次在 FAR<=5/h 区域 FRR（乘 100 后）往往 >35，导致 `det.png` 可能为空白；同时 stats 文件里写的是 rate（0-1）但 header 写 `%`，需要统一（不影响统计但影响解读）。

## 产出物（你能直接用来判断是否修复）

- 一份对比报告：baseline vs pinyin 在同一 FAR 点（例如 1/h、0.5/h、0.1/h）下的 FRR，以及正负样本分数分布重叠程度。
- 一套可复现实验：在 `score_ctc.py` 加参数后，跑 2-3 组配置，生成新的 `score.txt`/`stats.*.txt`/`det.png`，验证 FAR/FRR 是否回到接近 baseline 的形态。