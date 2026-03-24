# hi_xiaowen 流式与离线评测口径说明

> 更新时间：2026-03-24
> 适用代码：
> - `examples/hi_xiaowen/s0/infer_wav.py`
> - `examples/hi_xiaowen/s0/infer_wav_stream.py`
> - `examples/hi_xiaowen/s0/evaluate_infer_wav.py`
> - `examples/hi_xiaowen/s0/exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/*`

---

## 1. 目的

这份文档专门回答三个问题：

1. `hi_xiaowen` 当前到底有哪几种推理 / 评测口径。
2. 真流式后处理和之前单条 wav 离线推理，分别有哪些参数、默认策略和隐藏限制。
3. S3 模型在不同口径下为什么会出现不同精度，以及后续如何做 ablation study。

---

## 2. 先说结论

### 2.1 当前有三条容易混淆的链路

| 口径 | 前向方式 | 后处理方式 | 代表脚本 / 目录 | 是否真流式语义 |
|------|----------|------------|------------------|----------------|
| 单条 wav 离线推理 | 整段特征一次性前向 | 整段 `ctc_prefix_beam_search` | `infer_wav.py` | 否 |
| 旧流式评测口径 | 按 chunk 流式前向 | 先拼整段 probs，再做离线 beam search | `test_infer_stream_399_fix` / `test_infer_stream_399_tflite` | 否 |
| 真流式评测口径 | 按 chunk 流式前向 | 逐帧增量 beam search + 在线检测约束 | `test_infer_stream_399_py_post` / `test_infer_stream_399_cpp_post` | 是 |

### 2.2 S3 当前最该记住的精度

| 口径 | 结果目录 | 嗨小问 acc | 你好问问 acc | 说明 |
|------|----------|------------|--------------|------|
| 离线整段 | `test_infer_399` | 96.75% | 97.66% | 当前离线参考线 |
| 旧流式评测口径 | `test_infer_stream_399_fix` | 96.75% | 97.65% | 流式前向对齐，但后处理仍是离线 |
| 旧流式评测口径 + TFLite FP32 | `test_infer_stream_399_tflite` | 96.75% | 97.66% | 证明 streaming 前向 Torch/TFLite 基本对齐 |
| 真流式评测口径（Python） | `test_infer_stream_399_py_post` | 96.44% | 97.51% | 真流式基线 |
| 真流式评测口径（C++ pybind） | `test_infer_stream_399_cpp_post` | 96.44% | 97.51% | 与 Python 全量一致 |
| 真流式评测口径（C 风格 pybind） | `test_infer_stream_399_c_post` | 96.44% | 97.51% | 与 Python / C++ 全量一致 |

结论：

- `c_post`、`cpp_post` 和 `py_post` 的精度一致，说明当前 native decoder 替换没有引入额外掉点。
- 和你前几天记得的 `torch / tflite float` S3 结果相比，掉点出现在“后处理语义从离线切换成真流式”之后，而不是出现在 C++ 替换之后。

---

## 3. 三条链路分别在做什么

### 3.1 单条 wav 离线推理：`infer_wav.py`

核心流程：

1. 读整段 wav。
2. 提取整段离线 fbank。
3. 模型整段前向，得到整段 `T x V` 概率。
4. 对整段 `T x V` 直接调用 `ctc_prefix_beam_search`。
5. 在最优 hypothesis 里找关键词 token 子串。
6. 只做阈值判定，不做在线时长 / 间隔 / 状态 reset。

对应代码：

- CLI 参数定义：`infer_wav.py:119`
- 离线 decode：`infer_wav.py:576`
- 阈值加载：`infer_wav.py:680`
- 结果格式化：`infer_wav.py:710`

### 3.2 旧流式评测口径：`collect_streaming_probs(...) + decode_keyword_hit_with_token_info(...)`

核心流程：

1. 读整段 wav。
2. 按 chunk 做 streaming 前向和 cache 递推。
3. 把每个 chunk 产出的 frame probs 拼成整段 `T x V`。
4. 再把整段 `T x V` 丢给离线 `decode_keyword_hit_with_token_info(...)`。

对应 helper：

- `infer_wav_stream.py:491` `collect_streaming_probs(...)`
- `infer_wav.py:576` `decode_keyword_hit_with_token_info(...)`

这条路径的本质是：

- 前向是流式的。
- 后处理不是流式的。

所以它适合验证：

- streaming 前向有没有把 Torch 跑歪；
- TFLite step 模型和 Torch step 模型前向是否一致。

但它不适合回答“真实在线端侧后处理精度是多少”。

### 3.3 真流式评测口径：`collect_streaming_best_decode(...)`

核心流程：

1. 读整段 wav。
2. 按 chunk 流式提特征。
3. 每产出一帧 probs，就立刻做一次增量 beam update。
4. 每一帧都检查是否已经满足关键词匹配、阈值、时长、间隔等在线触发条件。
5. 全部音频送完后，返回历史最佳候选 `best_decode`。

对应 helper：

- `infer_wav_stream.py:504` `collect_streaming_best_decode(...)`
- `infer_wav_stream.py:409` `_step_decoder(...)`
- `infer_wav_stream.py:359` `_execute_detection(...)`
- `infer_wav_stream.py:432` `forward_chunk(...)`

这才是“真流式语义”。

---

## 4. 单条 wav 离线推理的参数与默认策略

### 4.1 CLI 参数：`infer_wav.py`

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--wav` | 必填 | 输入 wav |
| `--model` | `s3` | 模型别名 |
| `--checkpoint` | 空 | 显式 checkpoint / zip / tflite |
| `--model_dir` | 空 | 显式实验目录 |
| `--checkpoint_name` | 空 | 配合 `model_dir` 选 ckpt |
| `--config` | 空 | 显式 config |
| `--dict_dir` | 空 | 显式词典目录 |
| `--stats_dir` | 空 | 显式 stats 目录 |
| `--keywords` | `嗨小问,你好问问` | 关键词列表 |
| `--threshold_map` | 空 | 手动阈值覆盖 |
| `--target_fa_per_hour` | `1.0` | 从 stats 挑阈值时的 FA/h 目标 |
| `--pick_mode` | `legacy` | 阈值选择策略：`legacy` / `recall` / `robust` |
| `--frr_eps` | `0.001` | `robust` 模式 FRR 容差 |
| `--gpu` | `-1` | 设备选择 |
| `--disable_threshold` | `False` | 只看候选，不做最终阈值拒绝 |
| `--indent` | `2` | JSON 输出缩进 |

### 4.2 默认策略

1. 关键词阈值优先从 `stats.*.txt` 加载，不存在时才退回别名默认阈值。
2. `pick_mode=legacy` 时，阈值选择与 `analyze_exp_test_stats.py` 一致。
3. decode 使用整段 `ctc_prefix_beam_search`，不带在线时长 / 间隔约束。
4. 一旦在最优 hypothesis 中找到关键词 token 子串，就返回该候选。

### 4.3 这条链路没有的限制

下面这些限制，离线路径默认都没有：

- `min_frames`
- `max_frames`
- `interval_frames`
- stale beam reset
- 每帧 token `prob > 0.05` 的在线过滤
- 增量 beam 的每帧 top-k 剪枝语义

---

## 5. 真流式推理的参数与默认策略

### 5.1 CLI 参数：`infer_wav_stream.py`

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--wav` | 必填 | 输入 wav |
| `--model` | `s3` | 模型别名 |
| `--checkpoint` / `--model_dir` / `--checkpoint_name` / `--config` / `--dict_dir` / `--stats_dir` | 同离线版 | 模型与资源解析 |
| `--keywords` | `嗨小问,你好问问` | 关键词列表 |
| `--threshold_map` | 空 | 手动阈值覆盖 |
| `--target_fa_per_hour` | `1.0` | 通过 stats 自动挑阈值 |
| `--pick_mode` | `legacy` | 阈值策略 |
| `--frr_eps` | `0.001` | `robust` FRR 容差 |
| `--gpu` | `-1` | 设备选择 |
| `--disable_threshold` | `False` | 关闭最终阈值判定 |
| `--chunk_ms` | `300.0` | 每次送入的音频 chunk 时长 |
| `--score_beam_size` | `3` | 每帧 token 初筛 beam size |
| `--path_beam_size` | `20` | prefix beam 保留条数 |
| `--min_frames` | `5` | 关键词最短帧数 |
| `--max_frames` | `250` | 关键词最长帧数 |
| `--interval_frames` | `50` | 连续两次触发最小间隔 |
| `--use_cpp_decoder` | `False` | 使用 C++ pybind decoder |
| `--use_c_decoder` | `False` | 使用 C 风格 pybind decoder |
| `--indent` | `2` | JSON 输出缩进 |

### 5.2 当前默认在线策略

#### A. 每帧 token 过滤

当前 Python 流式 decoder 每一帧只会保留：

- `topk(score_beam_size)` 里的 token；
- 且 `prob > 0.05`；
- 且 token 必须属于 `keywords_idxset`。

对应代码：`infer_wav_stream.py:86`

这意味着它不会像离线 beam 一样对整段所有 token 概率做统一搜索。

#### B. prefix beam 截断

每一帧更新之后只保留前 `path_beam_size` 条 prefix。

对应代码：`infer_wav_stream.py:416`

#### C. 在线触发约束

命中时除了分数，还会检查：

- `threshold_map`
- `min_frames <= duration <= max_frames`
- `end_frame - last_active_pos >= interval_frames`

对应代码：`infer_wav_stream.py:377`

#### D. stale beam reset

如果当前最早 hypothesis 起点距离当前帧已经超过 `max_frames`，就把 decode state 清掉。

对应代码：`infer_wav_stream.py:456`

这在真实在线系统里很合理，但它和离线整段 beam search 明显不是同一语义。

#### E. best_decode 与 triggered 的区别

- `best_decode`：历史上见过的最佳候选
- `triggered`：是否满足在线触发条件并正式通过阈值

单条 wav 脚本默认在流式过程中一旦触发，可以提前停止；全量评测则通常把整段喂完后取最佳候选。

---

## 6. 批量评测脚本当前能调什么，不能调什么

### 6.1 `evaluate_infer_wav.py` 暴露给命令行的参数

当前 batch 脚本在 streaming 模式下，命令行能直接调的主要是：

- `--chunk_ms`
- `--use_cpp_decoder`
- `--use_c_decoder`
- `--threshold_map`
- `--target_fa_per_hour`
- `--pick_mode`
- `--frr_eps`

对应 CLI：`evaluate_infer_wav.py:41`

### 6.2 当前 batch 脚本里仍是硬编码的真流式参数

`evaluate_infer_wav.py` 在 worker 内创建 `StreamingKeywordSpotter(...)` 时，下面这些值还没有暴露到 CLI：

- `score_beam_size=3`
- `path_beam_size=20`
- `min_frames=5`
- `max_frames=250`
- `interval_frames=50`
- `disable_threshold=False`

对应代码：`evaluate_infer_wav.py:300`

因此：

- 单条 wav 的真流式参数可以动态调。
- 批量全量评测的真流式参数目前还不能动态调，除非改脚本。

---

## 7. S3 为什么会出现两种精度

### 7.1 不是 C++ 替换导致

已经验证：

- `test_infer_stream_399_py_post`
- `test_infer_stream_399_cpp_post`
- `test_infer_stream_399_c_post`

三者 `score.txt` 按 key 对齐后是 0 差异。

所以当前可以认为：

- Python 真流式 decoder
- C++ pybind 真流式 decoder
- C 风格 pybind 真流式 decoder

语义一致，当前精度差异不是 native decoder 实现引入的。

### 7.2 真正的分界点是“后处理语义切换”

从 git 历史看：

- `094cb13 feat(kws): add streaming torch inference and eval`
  - 已经是“流式前向 + 离线后处理”口径
- `045d6c0 Add streaming C++ CTC decoder and docs`
  - streaming 评测切到 `collect_streaming_best_decode(...)`
  - 即“真流式后处理”口径

所以如果你记得前几天 `torch / tflite float` 的 S3 精度接近 `96.75 / 97.66`，那对应的是旧口径，而不是今天的真流式口径。

### 7.3 目前可以确认的差异来源

下面几项都会让真流式结果偏离旧口径：

1. `min_frames / max_frames / interval_frames`
2. stale beam reset
3. 每帧 `prob > 0.05` 过滤
4. `score_beam_size=3` 的逐帧 top-k 剪枝
5. `path_beam_size=20` 的 prefix 剪枝
6. 增量 beam search 本身和“整段统一做一次 beam search”不完全等价

因此不能简单地说：

- “只要去掉时长限制，就一定恢复旧精度”

更准确的说法是：

- 去掉在线限制可能会恢复一部分；
- 但只要后处理仍是逐帧增量 decode，就未必 100% 等于旧口径。

### 7.4 当前已经观察到的现象

基于 `test_infer_stream_399_fix` vs `test_infer_stream_399_py_post`：

- 旧口径与真流式口径按 key 对齐后，预测结果共有 29 条发生变化。
- 其中“旧口径命中，但真流式拒绝”的样本有 19 条。
- 这 19 条里有 12 条跨度超过约 2.5s，强烈说明 `max_frames=250` 是重要因素。
- 另外还有少量样本发生关键词重分配，说明 beam 剪枝与在线状态机本身也在改结果。

---

## 8. 后续复现建议

### 8.1 如果你的目标是验证 streaming 前向没有跑歪

优先看：

- `test_infer_stream_399_fix`
- `test_infer_stream_399_tflite`

这两组更适合回答：

- Torch step 前向和 TFLite step 前向是否一致

但它们不回答“真实在线后处理精度”。

### 8.2 如果你的目标是评估真实在线端侧表现

优先看：

- `test_infer_stream_399_py_post`
- `test_infer_stream_399_cpp_post`
- `test_infer_stream_399_c_post`

并且后续所有调参都应以这三条真流式口径为基线。

---

## 9. 建议的 ablation study

下面建议按“先最可能、再较细节”的顺序做。

### A0. 当前真流式基线

保持现状：

- `score_beam_size=3`
- `path_beam_size=20`
- `min_frames=5`
- `max_frames=250`
- `interval_frames=50`
- Python / C++ / C 各跑一遍确认一致

目标：确认后续所有对比都从同一真流式基线出发。

### A1. 先放宽 `max_frames`

建议扫描：

- `250 -> 350 -> 450 -> 600`

理由：

- 当前已有证据表明不少旧命中新真流式拒绝样本是超长跨度。
- 这项最有机会立刻回收一部分精度。

### A2. 调整 / 关闭 stale reset

当前 stale reset 与 `max_frames` 绑定。

建议对比：

- 保持 reset
- 放宽 reset 阈值
- 完全关闭 reset

理由：

- 即使放宽 `max_frames`，过早 reset 仍可能让长尾关键词被截断。

### A3. 调整 `interval_frames`

建议扫描：

- `50 -> 20 -> 0`

理由：

- 对单次关键词命中影响通常小于 `max_frames`；
- 但会影响一段音频里多次候选时谁被保留。

### A4. 放大 beam

建议扫描：

- `score_beam_size: 3 -> 5`
- `path_beam_size: 20 -> 50`

理由：

- 如果当前是 beam 太窄导致 hypothesis 被过早剪掉，这一步会比放宽时长更有效。

### A5. 放宽每帧 token 过滤

当前 Python 路径里有：

- `prob > 0.05`
- token 必须属于 `keywords_idxset`

建议扫描：

- `0.05 -> 0.02 -> 0.0`

理由：

- 这一步会更接近离线 beam 的搜索空间；
- 但误检和算力也可能上升。

### A6. 重新评估 TFLite FP32 的真流式后处理

当前 `test_infer_stream_399_tflite` 仍是旧口径。

建议在 batch 脚本支持传递上述流式参数之后，再补跑：

- Torch + Python 真流式
- Torch + C++ 真流式
- Torch + C 风格 pybind 真流式
- TFLite FP32 + 真流式

这样才能做真正 apples-to-apples 的三方对比。

### A7. `chunk_ms` 敏感性

建议扫描：

- `300`
- `500`
- `1000`

理由：

- 理论上逐帧 decoder 对 `chunk_ms` 不该太敏感；
- 但特征拼接、cache 递推、端点对齐仍可能造成边界差异。

---

## 10. 推荐复现命令

### 10.1 单条 wav 离线口径

```bash
cd examples/hi_xiaowen/s0

python ./infer_wav.py \
  --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --wav speech_charctc_kws_phone-xiaoyun/example/kws_xiaoyunxiaoyun.wav
```

### 10.2 单条 wav 真流式口径

```bash
cd examples/hi_xiaowen/s0

python ./infer_wav_stream.py \
  --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --wav speech_charctc_kws_phone-xiaoyun/example/kws_xiaoyunxiaoyun.wav \
  --chunk_ms 300 \
  --score_beam_size 3 \
  --path_beam_size 20 \
  --min_frames 5 \
  --max_frames 250 \
  --interval_frames 50
```

### 10.3 全量旧口径

```bash
cd examples/hi_xiaowen/s0

python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus 0,1,2,3 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_fix
```

### 10.4 全量真流式口径

```bash
cd examples/hi_xiaowen/s0

python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus -1 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_py_post

python ./evaluate_infer_wav.py --model s3 
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt 
  --test_data data/test/data.list --gpus -1 --progress_every 2000 
  --streaming --chunk_ms 1000 --use_cpp_decoder --result_test_id test_infer_stream_399_cpp_post

python ./evaluate_infer_wav.py --model s3 
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt 
  --test_data data/test/data.list --gpus 0,1,2,3 --progress_every 2000 
  --streaming --chunk_ms 1000 --use_c_decoder --result_test_id test_infer_stream_399_c_post
```

---

## 11. 最后一句话

后续如果只允许保留一种“官方精度口径”，建议拆成两个指标同时维护：

1. **Streaming Frontend Parity**：看 `collect_streaming_probs + offline decode`，回答前向有没有跑歪。
2. **True Streaming Accuracy**：看 `collect_streaming_best_decode`，回答真实在线后处理精度。

这两个指标不要再混在同一张表里，否则后面很容易把“前向对齐”和“在线检测能力”误认为同一件事。
