# hi_xiaowen 实验记录

> **维护说明**：基线内容截至 **2026-03-02**；**2026-03-23** 起增补 **LiteRT 流式 TFLite** 与 **`evaluate_infer_wav` 全量索引**；**2026-03-24** 起补充 **真流式 decoder（Python / C++ pybind）** 对齐结论，并澄清旧 `test_infer_stream_*` 的后处理口径；**2026-03-25** 起补充 **C decoder lazy-node 内存压缩** 的全量回归，确认当前实现无精度回退；**2026-03-26** 起补充 **72KB 纯在线窄化版** 的全量回归与根因定位。  
> **路径约定**：未写绝对路径时，均相对仓库内 `examples/hi_xiaowen/s0/`。

---

## 怎么读这份文档

| 你想… | 直接去 |
|--------|--------|
| **快速知道什么有效、什么踩坑** | [§1 速览](#1-速览) |
| **查全量 infer 目录、模型路径、精度** | [§2 全量推理索引](#2-全量推理索引) |
| **按实验类型查数字表** | [§3 分主题实验表](#3-分主题实验表) |
| **方法 × 结论一页复盘** | [§4 复盘表](#4-复盘表) |
| **日志/脚本/配置在哪** | [§5 证据索引](#5-证据索引) |
| **v2 两条 test_299 完整训练命令** | [附录 A](#附录-a) |

---

## 1. 速览

### 1.1 有效尝试 ✅

| 方法 | 参数量 / 体积 | 精度（代表结果） | 结论 |
|------|----------------|------------------|------|
| **权重手术** | 756K → **392K** | test_79 约 **98%+** | 最稳的压缩手段 |
| **知识蒸馏（199K）** | **~200K** | test_229 约 **98%** | 可行，训练与超参敏感 |
| **蒸馏 v3 merged（S3）** | **~74K** | test_399 你 97.66% / 嗨 96.75% | 当前 **精度–参数** 较好折中 |
| **S3 真流式后处理（Py / C++ / C / lazy-node C）** | **~74K** | `test_infer_stream_399_{py,cpp,c}_post` 与 `test_infer_stream_399_c_post_lazy_nodes`：你 **97.51%** / 嗨 **96.44%** | **Python / C++ / C 风格 pybind 全量一致**；`lazy-node` 压缩版与旧 `c_post` 全量一致，较旧流式口径略降 |
| **S3 真流式后处理（C decoder 72KB 版）** | **decoder heap ~72KB** | `test_infer_stream_399_c_post_72k`：你 **97.48%** / 嗨 **96.42%** | 相对磁盘上的 `lazy_nodes` 目录有轻微回退，但需结合当前代码口径重新判断 |
| **INT8 PTQ（Torch/离线）** | 位数 ↓ | 229 → 229_int8 **轻微掉点** | 可接受 |
| **LiteRT 流式 FP32（S3）** | TFLite step | 与离线 infer **同量级** | 导出链路对齐好 |
| **LiteRT 流式 INT8（S3）** | 同上 | 相对 FP32 **明显回退** | 需继续优化 PTQ/校准等 |
| **LiteRT 流式 INT8（199K）** | 同上 | 嗨 98.05%/你 98.05%，几乎无损 | 199K 对 cache 量化容忍度高 |

> INT8 体积参考：量化日志示例 `793.1KB → 351.2KB`（TorchScript zip）。

### 1.2 失败或高风险 ❌

| 问题 | 现象 |
|------|------|
| 改小词表但 **HEAD 随机**（strict=false 等） | top2598/top440 **召回崩塌** |
| **简单裁剪**无权重手术 | 2598 等 **FRR≈100%** |
| 蒸馏 **早期/不当超参**（如 test_179） | 199K 嗨小问可低至 **~91%** |

### 1.3 核心洞察（HEAD）

- 迁移后的 **HEAD** 决定可分性与混淆抑制（如共享「问」字）。
- 改输出维度时务必 **保留/拷贝头部权重**，否则 CTC 易在「嗨小问 / 你好问问」间误判。
- 优先路线：**权重手术保头** 或 **蒸馏时显式复制/约束 HEAD**。

---

## 2. 全量推理索引

**公共设定**

- 脚本：`evaluate_infer_wav.py`；测试列表：`data/test/data.list`（**73459** utt）。
- 阈值：默认 **`legacy:fa<=target(1.0)`**（与各目录 `summary.json` 一致）。
- 未指定 `--result_test_id` 时，输出目录规则见 `evaluate_infer_wav.py` → `make_default_result_dir`。
- 常用命令出处：`docs/常用命令.txt` 约 **L136–L202**。

### 2.1 总表：输出目录 × 模型路径 × 精度

**S3 实验目录**：`exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/`（checkpoint 多为 `399.pt`）。

| 输出目录 | 模型路径（相对 `s0/`） | 流式 | 设备 | 嗨小问 acc | 你好问问 acc | `summary` |
|----------|-------------------------|------|------|------------|--------------|-----------|
| `test_infer_399` | `…/399.pt` | 否 | 多 GPU | **96.75%** | **97.66%** | ✅ |
| `test_infer_stream_399` | `…/399.pt` | 是 | 多 GPU | **92.80%** | **96.94%** | ✅ 旧流式路径，弱于 fix |
| `test_infer_stream_399_fix` | `…/399.pt` | 是 | 多 GPU | **96.75%** | **97.65%** | ✅ **旧口径**：流式前端 + `offline_ctc_prefix_beam` |
| `test_infer_stream_399_py_post` | `…/399.pt` | 是 | CPU | **96.44%** | **97.51%** | ✅ **真流式后处理**：`streaming_python_ctc_prefix_beam` |
| `test_infer_stream_399_cpp_post` | `…/399.pt` | 是 | CPU | **96.44%** | **97.51%** | ✅ **真流式后处理**：`streaming_cpp_ctc_prefix_beam`，与 Python 全量一致 |
| `test_infer_stream_399_c_post` | `…/399.pt` | 是 | CPU | **96.44%** | **97.51%** | ✅ **真流式后处理**：`streaming_c_ctc_prefix_beam`，与 Python / C++ 全量一致 |
| `test_infer_stream_399_c_post_lazy_nodes` | `…/399.pt` | 是 | 多 GPU | **96.44%** | **97.51%** | ✅ **真流式后处理**：`streaming_c_ctc_prefix_beam` 的 lazy-node 内存压缩版；与旧 `c_post` 全量一致 |
| `test_infer_stream_399_c_post_72k` | `…/399.pt` | 是 | 多 GPU | **96.42%** | **97.48%** | ⚠️ **真流式后处理**：72KB 纯在线窄化版；相对磁盘上的 `lazy_nodes` 结果略低 |
| `test_infer_stream_399_tflite` | `…/399_stream_litert_fp32.tflite` | 是 | CPU | **96.75%** | **97.66%** | ✅ **旧口径**：流式前端 + `offline_ctc_prefix_beam` |
| `test_infer_stream_399_tflite_int8` | `…/399_stream_litert_int8.tflite` | 是 | CPU | **91.39%** | **94.00%** | ✅ |
| `test_infer_stream_399_tflite_native_int8_calib200` | `…/399_stream_native_int8_calib200.tflite` | 是 | CPU | **94.66%** | **97.06%** | ✅ |

**199K 实验目录**：`exp/fsmn_ctc_distill_mini_align_20_test2/`（checkpoint `229.pt`）。

| 输出目录 | 模型路径（相对 `s0/`） | 流式 | 设备 | 嗨小问 acc | 你好问问 acc | `summary` |
|----------|-------------------------|------|------|------------|--------------|-----------|
| `test_infer_stream_229_tflite_int8` | `…/229_stream_litert_int8.tflite` | 是 | CPU | **98.05%** | **98.05%** | ✅ |

> 199K LiteRT 流式 INT8 精度几乎无损失（对比 PyTorch INT8 229：嗨 97.82%/你 97.97%），与 S3 INT8 显著回退形成对比。推测 199K 模型参数量大（250 维 linear_dim）、数值分布更宽裕，对 cache 量化误差容忍度更高。

**2026-03-24 口径澄清（S3 流式后处理）**

- `test_infer_stream_399_fix` 与 `test_infer_stream_399_tflite` 生成于 **2026-03-24 16:29** 这次 streaming decoder 改造提交（`045d6c0 Add streaming C++ CTC decoder and docs`）之前，`results.jsonl` 里记录的 `decode_mode` 都是 **`offline_ctc_prefix_beam`**。
- 旧逻辑是：先用 `collect_streaming_probs(...)` 把流式前向得到的整段概率拼起来，再调用 `infer_wav.py:decode_keyword_hit_with_token_info(...)` 做**离线** prefix beam 搜索。它验证的是 **streaming 前端 / Torch 与 TFLite 前向对齐**，不是“真流式后处理”。
- 新逻辑是：`evaluate_infer_wav.py --streaming` 调 `collect_streaming_best_decode(...)`，逐帧走 `StreamingKeywordSpotter._step_decoder(...)`。对应目录：
  - `test_infer_stream_399_py_post` -> `streaming_python_ctc_prefix_beam`
  - `test_infer_stream_399_cpp_post` -> `streaming_cpp_ctc_prefix_beam`
  - `test_infer_stream_399_c_post` -> `streaming_c_ctc_prefix_beam`
- 这三个新目录的 `score.txt` 按 `key` 对齐后 **0 差异**，说明当前 **Python / C++ / C 风格 pybind 真流式 decoder 全量一致**；本次掉点不是 native decoder 实现引入的。
- 它们相对旧 `fix/tflite float` 略低，核心原因是**评测语义变了**：真流式路径会额外施加 `min_frames=5`、`max_frames=250`、`interval_frames=50` 等在线约束，而旧离线后处理只看整段 prefix beam 最优路径，不施加这些约束。
- 一个代表性样本：`你 好 问 问` 在旧口径下会命中，`start=0.03s, end=4.17s, score≈0.784`；真流式路径拒绝。这个跨度约 **4.14s**，已明显超过 `max_frames=250`（约 **2.5s**）的在线约束。
- 全量对比里，旧 `fix` 命中但新真流式拒绝的样本有 **19** 条，其中 **12** 条跨度超过 **2.5s**；另有少量样本发生关键词重分配（如旧 `你好问问` -> 新 `嗨小问`），说明在线 beam 剪枝与时序约束共同改变了最终候选。
- 因此：
  - 若要证明 **Torch 与 TFLite streaming 前向** 一致，旧 `test_infer_stream_399_fix / tflite` 仍有参考价值。
  - 若要评估 **真实在线后处理精度**，应以 `test_infer_stream_399_py_post / cpp_post / c_post` 为准。

**2026-03-25：C decoder lazy-node 内存压缩全量回归**

- 新目录：`test_infer_stream_399_c_post_lazy_nodes`。命令与旧 `test_infer_stream_399_c_post` 唯一关键差异是底层 C decoder 改成了 lazy-node 物化版本。
- 全量 `summary.json` 与旧 `test_infer_stream_399_c_post` **完全一致**：嗨小问 `threshold=0.296, accuracy=96.44%, frr=3.56%, fa/h=0.99`；你好问问 `threshold=0.016, accuracy=97.51%, frr=2.49%, fa/h=0.50`。
- 更强证据：两次全量评测的 `results.jsonl` 的 `sha256` 都是 `08483e0ee587ed86b422afd03b9e038d982e226ff93becd964a6b4fe483df298`，`score.txt` 的 `sha256` 都是 `e996ba54843e0d4fec0251f13f9e14e1c446cbb063acfc146a474579c3c29a42`。
- 结论：截至 **2026-03-25**，这轮 lazy-node 内存压缩在 **73459** 条全量测试上没有观测到任何输出变化；当前证据支持它对现有 S3 真流式 C decoder 路径 **无精度回退**。

**2026-03-26：C decoder 72KB 纯在线窄化版全量回归**

- 新目录：`test_infer_stream_399_c_post_72k`。命令与 `lazy_nodes` 基线相比，关键差异是打开了 72KB 版 C decoder（`node_pool.token/frame` 收到 16 位，内部 duration / interval / stale reset 改成环形 frame 差值），并以 `HI_XIAOWEN_CTC_DECODER_C_EXTRA_CFLAGS="-DCTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES=0"` 重新编译加载。
- 全量 `summary.json` 显示轻微回退：嗨小问从 `threshold=0.296, accuracy=96.44%, frr=3.56%, fa/h=0.99` 变为 `threshold=0.294, accuracy=96.42%, frr=3.58%, fa/h=0.99`；你好问问从 `threshold=0.016, accuracy=97.51%, frr=2.49%, fa/h=0.50` 变为 `threshold=0.016, accuracy=97.48%, frr=2.52%, fa/h=0.48`。
- `results.jsonl` / `score.txt` 的 `sha256` 已变化：`lazy_nodes` 为 `08483e0ee587ed86b422afd03b9e038d982e226ff93becd964a6b4fe483df298` / `e996ba54843e0d4fec0251f13f9e14e1c446cbb063acfc146a474579c3c29a42`；`72k` 为 `ad0cdf951f480ec3c582a692ed2f6ee2660462fe6d4ecf57cee187ab7b1291c8` / `f359a589d07442bc6b333fd49f408f2986a4c23137bf0c412f77a721a92a6e8f`。
- 但这里面有两类差异要分开看：
  - 第一类是**预期差异**：大量 `results.jsonl` 变化仅体现在 `wake_time_sec / start_time_sec / end_time_sec`，属于此前 `frame -> sec` 修正后的时间语义变化，不代表检测结果变差。
  - 第二类才是**真实行为分叉**：只看 `triggered / keyword / score / start_frame / end_frame` 等检测相关字段，`73459` 条里共有 **15** 条发生变化，其中 **10** 条由“原先触发”变成“现在不触发”，另有 **4** 条为同关键词但分数和结束帧轻微变化，**1** 条为未触发样本的候选分数轻微变化。
- 这 **10** 条真实分叉高度集中在 `max_frames` 边界附近：旧结果多为 `start_frame=3`、`end_frame=255/258/261/270/273/276/279` 这类长跨度样本；72KB 版里相同样本要么被截到 `end_frame=252`，要么直接不再触发。
- 进一步单条重放显示：这些分叉**不一定是 72KB 窄化本身引入**。以 `0a14c32716afb98ddd1296fcf072528d`、`6a04368fc74709e31ac01e31a120547b` 为例，当前 **default C decoder** 与 **72KB C decoder** 都返回“不触发”，而当前 **Python streaming** 仍返回触发，说明当前差异更像是 **C decoder 与 Python/reference 口径不一致**，而不是 `72k` 相对“当前 default C”单独恶化。
- 更具体地说，当前 C decoder 在 `step_and_detect_next()` 中是**逐帧**调用 `ctc_decoder_c_maybe_reset_stale_beam(...)`；当首 token 在 `frame=3` 时，处理完 `frame=252` 后，下一帧索引将变成 `255`，此时 `255 - 3 = 252 > max_frames(250)`，beam 会在**进入 255 这一帧之前**被 reset。Python 路径则是在 `forward_chunk()` 的 chunk 尾部才做一次外部 stale reset，因此同类长尾候选还能继续跑到 `255/258/261/...` 并进入 `best_decode_result`。
- 随后补跑的 `test_infer_stream_399_c_post_current_default` 与 `test_infer_stream_399_c_post_72k` 的 `summary.json` **完全一致**：嗨小问 `threshold=0.294, accuracy=96.42%, frr=3.58%, fa/h=0.99`；你好问问 `threshold=0.016, accuracy=97.48%, frr=2.52%, fa/h=0.48`。进一步核对 artifacts，`results.jsonl` 都是 `ad0cdf951f480ec3c582a692ed2f6ee2660462fe6d4ecf57cee187ab7b1291c8`，`score.txt` 都是 `f359a589d07442bc6b333fd49f408f2986a4c23137bf0c412f77a721a92a6e8f`。
- 这意味着：`72KB` 版相对“当前 default C decoder”**没有独立增量回退**；`HI_XIAOWEN_CTC_DECODER_C_EXTRA_CFLAGS="-DCTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES=0"` 这个宏也**不是根因**。
- 因而 root cause 可以继续收口到更早的代码变化：`489429b torch2lite: internalize streaming decoder reset and fix frame timing`。其中真正影响检测行为的部分不是时间字段换算，而是 **把 stale reset 从 Python chunk 尾部外部判断，改成了 C decoder 在 `step_and_detect_next()` 里的逐帧内部判断**。
- 具体差异是：当前 C decoder 在处理完每一帧后，都会立刻按 `next_frame_index - first_hyp_start_frame > max_frames` 判断是否 reset；Python/reference 路径则是在整块 `forward_chunk()` 完成后才按 `self.total_frames - start > self.max_frames` 做一次外部 reset。对于 `start_frame=3`、`frame_step=3` 的长尾样本，C decoder 会在进入 `255` 这一帧之前 reset，而 Python 路径还能继续保留同一条候选到 `255/258/261/...` 并更新 `best_decode_result`。
- 当前最终结论应修正为：相对旧的 `lazy_nodes` 磁盘基线，今天看到的差异**主要不是 72KB 窄化引入的**，而是 **`489429b` 这轮 stale reset 内收后，C decoder 与 Python/旧评测口径发生了语义变化**。如果要恢复到旧 `lazy_nodes` 结果，优先应排查和调整的不是 16 位 `frame`，而是 stale reset 的时机与作用范围。

**其他**（暂无全量结果目录）

| 预期目录 | 模型路径 | 说明 |
|----------|-----------|------|
| `test_infer_399_litert_fp32` | `…/399_litert_fp32.tflite` | **整段静态** LiteRT，与 `399_stream_*` **不是**同一导出 |

### 2.2 LiteRT 流式 TFLite（S3）补充说明

- **导出**：`torch2lite/export_streaming_litert_tflite.py`；step 输入 **`1×1×400` + cache**（与端侧 chunk 一致）。
- **FP32**：`399_stream_litert_fp32.tflite`，元信息 `399_stream_litert_fp32.tflite.json`。
- **INT8（PT2E）**：`399_stream_litert_int8.tflite`；`399_stream_litert_int8.tflite.json` 中可见 `quant_mode=int8_pt2e`、`calib_data=data/train/data.list`、`num_calib=200`、`seed=20260323`。
- **结论**：
  - 旧目录 `test_infer_stream_399_tflite` 在 **`offline_ctc_prefix_beam`** 口径下与离线 **`test_399` 同量级**，说明导出后的 **streaming 前向** 基本对齐。
  - 但它**不能直接代表真流式后处理精度**；若要和 `test_infer_stream_399_py_post / cpp_post / c_post` apples-to-apples 对比，需要用当前脚本重新跑一遍 TFLite FP32 真流式后处理。
  - INT8 流式相对 FP32 **显著回退**（cache 累积量化误差等），后续可加大校准、QAT 或混合精度。

### 2.4 TFLite Native PTQ 流式（S3）补充说明

- **pipeline**：PyTorch → ONNX (`torch.onnx.export`) → TFLite INT8 (`onnx2tf`)；脚本 `torch2lite/quantize_tflite_native.py`。
- **INT8（200 calib）**：`399_stream_native_int8_calib200.tflite`（125KB，FP32 为 323KB）。
- **量化模式**：权重 INT8，输入/输出保持 float32（cache 天然 FP32，避免 LiteRT PT2E 的 cache 量化误差累积）。
- **结论**：Native PTQ 相比 LiteRT PT2E 明显改善（嗨 94.66% vs 91.39%，你 97.06% vs 94.00%），但仍未恢复到 FP32 水平（嗨差 2.1%，你差 0.6%）。待验证：增加校准数据量（1000 条）能否进一步缩小差距。

### 2.3 LiteRT 流式 TFLite（199K）补充说明

- **INT8（PT2E）**：`229_stream_litert_int8.tflite`；与 S3 相同导出脚本和校准配置（`num_calib=200`）。
- **结论**：199K LiteRT 流式 INT8 **精度几乎无损**（嗨 98.05%/你 98.05%），远优于 S3 INT8（嗨 91.39%/你 94.00%）。推测原因：199K 的 `linear_dim=250`（vs S3 的 `linear_dim=250`）和 `proj_dim` 更大，特征空间更宽裕，对 cache 量化误差容忍度更高；或 199K 整体参数量更大（199K vs 74K），量化粒度影响较小。

### 2.3 复现与统计命令

```bash
cd examples/hi_xiaowen/s0

# 离线全量（默认目录 test_infer_399）
python ./evaluate_infer_wav.py --model s3 --test_data data/test/data.list \
  --gpus 0,1,2,3 --progress_every 2000

# 流式 Torch（旧口径，streaming 前端 + offline 后处理）
python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus 0,1,2,3 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_fix

# 真流式后处理（Python / C++ / C 风格 pybind，对比应看这三组）
python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus -1 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_py_post

python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus -1 --progress_every 2000 \
  --streaming --chunk_ms 1000 --use_cpp_decoder --result_test_id test_infer_stream_399_cpp_post

python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus 0,1,2,3 --progress_every 2000 \
  --streaming --chunk_ms 1000 --use_c_decoder --result_test_id test_infer_stream_399_c_post

HI_XIAOWEN_CTC_DECODER_C_EXTRA_CFLAGS="-DCTC_DECODER_C_ENABLE_DEBUG_HYPOTHESES=0" \
python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus 0,1,2,3 --progress_every 2000 \
  --streaming --chunk_ms 1000 --use_c_decoder --result_test_id test_infer_stream_399_c_post_72k

# 流式 TFLite FP32 / INT8
python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399_stream_litert_fp32.tflite \
  --test_data data/test/data.list --gpus -1 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_tflite

python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399_stream_litert_int8.tflite \
  --test_data data/test/data.list --gpus -1 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_tflite_int8

# 199K 流式 INT8（跑完才有目录）
python ./evaluate_infer_wav.py --model distill199 \
  --checkpoint exp/fsmn_ctc_distill_mini_align_20_test2/229_stream_litert_int8.tflite \
  --test_data data/test/data.list --gpus -1 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_229_tflite_int8

# 整段静态 FP32 TFLite（默认 test_infer_399_litert_fp32）
python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399_litert_fp32.tflite \
  --test_data data/test/data.list --gpus -1 --progress_every 2000
```

查看汇总：

```bash
python analyze_exp_test_stats.py --exp-dir "exp/fsmn_ctc_distill_s3_a48_p24_l3_merged" --test-id "test_infer_stream_399_fix"
# 其它：把 --test-id 换成上表目录名即可
```

---

## 3. 分主题实验表

### 3.1 2599 baseline 与简单裁剪（无权重手术）

| 实验 | 关键词 | accuracy | frr | fa/h | 结论 |
|------|--------|---------:|-----|------|------|
| baseline_4gpus/test_2 | 你好问问 | 96.26% | 3.74% | 0.41 | 基线 |
| baseline_4gpus/test_2 | 嗨小问 | 96.96% | 3.04% | 0.93 | 基线 |
| top2598/test_2 | 你好问问 | 0.00% | 100.00% | 0.00 | 失败 |
| top2598/test_2 | 嗨小问 | 1.12% | 98.88% | 0.93 | 失败 |
| top440/test_2 | 你好问问 | 0.67% | 99.33% | 0.86 | 失败 |
| top440/test_2 | 嗨小问 | 1.00% | 99.00% | 0.99 | 失败 |
| top2599/test_2 | 你好问问 | 96.71% | 3.29% | 0.58 | 接近基线 |
| top2599/test_2 | 嗨小问 | 96.10% | 3.90% | 0.99 | 接近基线 |

**做法**：`gen_reduced_dict.py` 小词表；`checkpoint_strict=false`；无手术时 HEAD 易随机 → 崩塌。

### 3.2 权重手术（top20）

| 实验 | 关键词 | accuracy | frr | fa/h |
|------|--------|---------:|-----|------|
| baseline_4gpus/test_79 | 你好问问 | 98.05% | 1.95% | 0.28 |
| baseline_4gpus/test_79 | 嗨小问 | 98.38% | 1.62% | 0.64 |
| top20_weight_surgery/test_79 | 你好问问 | 98.63% | 1.37% | 0.32 |
| top20_weight_surgery/test_79 | 嗨小问 | 98.88% | 1.12% | 0.67 |

**参数量**：756,133 → **392,494**。`wekws/utils/checkpoint.py` 对 `out_linear2` 按 token 行拷贝。

### 3.3 知识蒸馏 199K

| 实验 | 关键词 | accuracy | frr | fa/h |
|------|--------|---------:|-----|------|
| distill_mini_align_20_test2/test_179 | 你好问问 | 95.31% | 4.69% | 0.63 |
| distill_mini_align_20_test2/test_179 | 嗨小问 | 91.50% | 8.50% | 0.67 |
| distill_mini_align_20_test2/test_229 | 你好问问 | 98.05% | 1.95% | 0.37 |
| distill_mini_align_20_test2/test_229 | 嗨小问 | 98.13% | 1.87% | 0.98 |

**参数量**：**199,760**（Backbone 196,940 + Head 2,820）。

### 3.4 蒸馏 v2（113K）与 test_299 对比

**架构**：重点跟踪 `a64_p32_l2`，约 **113,142** 参数；同系列还有 `a80_p48_l3`、`a64_p32_l3`（见 `docs/常用命令.txt`）。

| 实验 | 关键词 | accuracy | frr | fa/h | 备注 |
|------|--------|---------:|-----|------|------|
| distill_v2_a64_p32_l2/test_299 | 你好问问 | 97.30% | 2.70% | 0.70 | 从头 200+100 |
| distill_v2_a64_p32_l2/test_299 | 嗨小问 | 96.42% | 3.58% | 0.99 | |
| distill_v2_a64_p32_l2_ft_from199_lr1e4/test_299 | 你好问问 | 97.30% | 2.70% | 0.99 | 从 199.pt 继续 |
| distill_v2_a64_p32_l2_ft_from199_lr1e4/test_299 | 嗨小问 | 96.04% | 3.96% | 0.99 | 略差 |

**对比结论**：从 `199.pt` + 固定 `1e-4` 未优于从头到 299；你好问问 FA 变差。**完整训练命令见 [附录 A](#附录-a)。**

### 3.5 PTQ / ExecuTorch（199K）

| 实验 | 关键词 | accuracy | frr | fa/h |
|------|--------|---------:|-----|------|
| test_229 | 你好问问 | 98.05% | 1.95% | 0.37 |
| test_229 | 嗨小问 | 98.13% | 1.87% | 0.98 |
| test_229_int8 | 你好问问 | 97.97% | 2.03% | 0.45 |
| test_229_int8 | 嗨小问 | 97.82% | 2.18% | 0.99 |
| test_229_executorch_int8 | 你好问问 | 98.37% | 1.63% | 0.53 |
| test_229_executorch_int8 | 嗨小问 | 98.32% | 1.68% | 0.99 |

模型产物：Torch INT8 对应 `exp/fsmn_ctc_distill_mini_align_20_test2/229_int8.pt`（全量目录 `test_229_int8`）；ExecuTorch PT2E INT8 对应 `exp/fsmn_ctc_distill_mini_align_20_test2/229_executorch_int8.pte`（全量目录 `test_229_executorch_int8`）。
另：INT16 模拟、ExecuTorch PT2E 多 backend（xnnpack / portable / hifi4）已打通，见常用命令 PTQ / ExecuTorch 段。

### 3.6 蒸馏 v3 merged（S1–S5，`test_399`）

**公共设定**（详见 `docs/常用命令.txt` 约 L122–139）：teacher `exp/fsmn_ctc_top20_weight_surgery/79.pt`，`dict_top20`，`align 300 + finetune 100`，`layer_mapping="0:1,1:2,2:3"`，`merge_head=true`，`lr_scheduler=none` 等。

| 实验 | student_config | 参数量 |
|------|----------------|-------:|
| S1 | `s1_a64_p32_l3_merged` | ~96,836 |
| S2 | `s2_a56_p28_l3_merged` | ~85,484 |
| S3 | `s3_a48_p24_l3_merged` | ~74,132 |
| S4 | `s4_a40_p20_l3_merged` | ~62,780 |
| S5 | `s5_a32_p16_l3_merged` | ~51,428 |

配置证据：`conf/fsmn_ctc_student_s{1..5}_*_merged.yaml` 文件头注释。

**结果**（`analyze_exp_test_stats.py --test-id 399`，阈值 `legacy:fa<=target(1.0)`）：

| 实验 | 参数量 | 关键词 | thr | accuracy | frr | fa/h |
|------|-------:|--------|----|---------:|-----|------|
| s1/test_399 | 96,836 | 你好问问 | 0.000 | 97.58% | 2.42% | 0.54 |
| s1/test_399 | 96,836 | 嗨小问 | 0.338 | 96.86% | 3.14% | 0.99 |
| s2/test_399 | 85,484 | 你好问问 | 0.000 | 97.45% | 2.55% | 0.45 |
| s2/test_399 | 85,484 | 嗨小问 | 0.417 | 96.44% | 3.56% | 0.99 |
| s3/test_399 | 74,132 | 你好问问 | 0.016 | 97.66% | 2.34% | 0.53 |
| s3/test_399 | 74,132 | 嗨小问 | 0.272 | 96.75% | 3.25% | 0.99 |
| s4/test_399 | 62,780 | 你好问问 | 0.000 | 96.98% | 3.02% | 0.67 |
| s4/test_399 | 62,780 | 嗨小问 | 0.423 | 95.53% | 4.47% | 0.99 |
| s5/test_399 | 51,428 | 你好问问 | 0.000 | 96.83% | 3.17% | 0.93 |
| s5/test_399 | 51,428 | 嗨小问 | 0.520 | 93.40% | 6.60% | 0.98 |

**简评**：S1–S3 可用；**你好问问**最好 **S3**；**嗨小问**最好 **S1**，S3 极接近。**S3** 常视为 **精度–参数** 折中点。S4/S5 明显变差。与 v2 test_299 仅能做趋势对比（测试 id 不同）。

---

## 4. 复盘表

| 方法 | 做法要点 | 已做尝试 | 结论 |
|------|----------|----------|------|
| 简单裁剪词表 | 小 dict + 改 `num_keywords` | 2598、440、2599 | 2598/440 失败；2599 接近基线 |
| 权重手术 | `out_linear2` 按 token 拷贝 | top20 等 | **最稳**减参 |
| 蒸馏 199K | feature align + HEAD 策略 | 179、229 | 可达 ~98%，调参成本高 |
| 蒸馏 v2 | 更小 backbone | 从头、从 199 继续 | 可用，整体低于 199K |
| 蒸馏 v3 merged | 3 层 + merge_head，缩宽 | S1–S5 / test_399 | **S3 ~74K** 为较好折中 |
| PTQ | INT8/INT16、ExecuTorch | 229_int8 等 | 离线 INT8 损失小 |
| LiteRT 流式 | step TFLite FP32/INT8 | S3 全量已记 | FP32 对齐好；INT8 流式仍弱 |

---

## 5. 证据索引

| 类别 | 路径或说明 |
|------|------------|
| 常用命令 | `docs/常用命令.txt`（v2：~L26–29；v3 S1–S5：~L122–139；infer/LiteRT：~L136–202） |
| v2 日志 | `exp/fsmn_ctc_distill_v2_a64_p32_l2/logs/run_distill_stage_2_3_20260228_205342.log` |
| v2 从 199 继续 | `exp/.../fsmn_ctc_distill_v2_a64_p32_l2_ft_from199_lr1e4/logs/run_distill_stage_2_2_20260302_113805.log` |
| 参数量 log | baseline：`exp/fsmn_ctc_baseline_4gpus/logs/...`（756,133）；top20 surgery：`exp/fsmn_ctc_top20_weight_surgery/logs/...`（392,494）；199K：`exp/fsmn_ctc_distill_mini_align_20_test2/logs/...`（199,760） |
| 统计脚本 | `examples/hi_xiaowen/s0/analyze_exp_test_stats.py` |
| LiteRT 导出 | `examples/hi_xiaowen/s0/torch2lite/export_streaming_litert_tflite.py` |
| 全量 infer 结果 | 各 `exp/.../test_infer_*` 下 `summary.json`；索引见 **[§2](#2-全量推理索引)** |

---

## 附录 A

蒸馏 v2 **`test_299`** 两条实验的完整训练命令（与 `docs/常用命令.txt` 一致）。

### A.1 从头训练（200+100，`2 3` 带 Stage3 评测）

```bash
bash ./run_distill.sh \
  --teacher_checkpoint exp/fsmn_ctc_top20_weight_surgery/79.pt \
  --teacher_config exp/fsmn_ctc_top20_weight_surgery/config.yaml \
  --student_config conf/fsmn_ctc_student_v2_a64_p32_l2.yaml \
  --num_keywords 20 --dict_dir dict_top20 \
  --target_exp_dir exp/fsmn_ctc_distill_v2_a64_p32_l2 \
  --gpus "0,1,2,3" --seed 666 \
  --align_epochs 200 --finetune_epochs 100 \
  --head_lr_ratio 0.1 \
  --finetune_mse_weight_start 0.6 --finetune_mse_weight_end 0.15 \
  --layer_mapping "0:2,1:3" \
  --lr_scheduler plateau --scheduler_start_epoch 200 \
  2 3
```

**要点**：最小 v2 结构 `a64_p32_l2`；`layer_mapping=0:2,1:3`；plateau 从 epoch 200 起调度；产出 `test_299`。

### A.2 从 `199.pt` 继续（固定 lr，`2 2`）

```bash
bash ./run_distill.sh \
  --teacher_checkpoint exp/fsmn_ctc_top20_weight_surgery/79.pt \
  --teacher_config exp/fsmn_ctc_top20_weight_surgery/config.yaml \
  --student_config conf/fsmn_ctc_student_v2_a64_p32_l2.yaml \
  --num_keywords 20 --dict_dir dict_top20 \
  --target_exp_dir exp/fsmn_ctc_distill_v2_a64_p32_l2_ft_from199_lr1e4 \
  --checkpoint exp/fsmn_ctc_distill_v2_a64_p32_l2/199.pt \
  --align_epochs 200 --finetune_epochs 100 \
  --resume_lr 1e-4 --finetune_lr 1e-4 \
  --lr_scheduler none \
  --head_lr_ratio 0.1 \
  --finetune_mse_weight_start 0.6 --finetune_mse_weight_end 0.15 \
  --layer_mapping "0:2,1:3" \
  --gpus "0,1,2,3" --seed 666 \
  2 2
```

**要点**：从上一实验 `199.pt` 恢复；`resume_lr`/`finetune_lr`=1e-4；`lr_scheduler=none`；`test_299` 多为后续单独评估。

查看统计：`python analyze_exp_test_stats.py --test-id 299`（需与本仓库 exp 命名一致）。
