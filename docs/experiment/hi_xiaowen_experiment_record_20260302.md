# hi_xiaowen 实验记录

> **维护说明**：基线内容截至 **2026-03-02**；**2026-03-23** 起增补 **LiteRT 流式 TFLite** 与 **`evaluate_infer_wav` 全量索引**。  
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
| **INT8 PTQ（Torch/离线）** | 位数 ↓ | 229 → 229_int8 **轻微掉点** | 可接受 |
| **LiteRT 流式 FP32（S3）** | TFLite step | 与离线 infer **同量级** | 导出链路对齐好 |
| **LiteRT 流式 INT8（S3）** | 同上 | 相对 FP32 **明显回退** | 需继续优化 PTQ/校准等 |

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
| `test_infer_stream_399_fix` | `…/399.pt` | 是 | 多 GPU | **96.75%** | **97.65%** | ✅ **推荐** Torch 流式对照 |
| `test_infer_stream_399_tflite` | `…/399_stream_litert_fp32.tflite` | 是 | CPU | **96.75%** | **97.66%** | ✅ |
| `test_infer_stream_399_tflite_int8` | `…/399_stream_litert_int8.tflite` | 是 | CPU | **91.39%** | **94.00%** | ✅ |

**常用命令有写、但本仓库暂无全量结果目录**

| 预期目录 | 模型路径 | 说明 |
|----------|-----------|------|
| `test_infer_stream_229_tflite_int8` | `exp/fsmn_ctc_distill_mini_align_20_test2/229_stream_litert_int8.tflite` | `--model distill199`，需自行跑完全量 |
| `test_infer_399_litert_fp32` | `…/399_litert_fp32.tflite` | **整段静态** LiteRT，与 `399_stream_*` **不是**同一导出 |

### 2.2 LiteRT 流式 TFLite（S3）补充说明

- **导出**：`torch2lite/export_streaming_litert_tflite.py`；step 输入 **`1×1×400` + cache**（与端侧 chunk 一致）。
- **FP32**：`399_stream_litert_fp32.tflite`，元信息 `399_stream_litert_fp32.tflite.json`。
- **INT8（PT2E）**：`399_stream_litert_int8.tflite`；`399_stream_litert_int8.tflite.json` 中可见 `quant_mode=int8_pt2e`、`calib_data=data/train/data.list`、`num_calib=200`、`seed=20260323`。
- **结论**：FP32 流式与离线 **`test_399` 同量级**；INT8 流式相对 FP32 **显著回退**（cache 累积量化误差等），后续可加大校准、QAT 或混合精度。

### 2.3 复现与统计命令

```bash
cd examples/hi_xiaowen/s0

# 离线全量（默认目录 test_infer_399）
python ./evaluate_infer_wav.py --model s3 --test_data data/test/data.list \
  --gpus 0,1,2,3 --progress_every 2000

# 流式 Torch（推荐目录名）
python ./evaluate_infer_wav.py --model s3 \
  --checkpoint exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt \
  --test_data data/test/data.list --gpus 0,1,2,3 --progress_every 2000 \
  --streaming --chunk_ms 1000 --result_test_id test_infer_stream_399_fix

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

### 3.5 PTQ（199K，Torch）

| 实验 | 关键词 | accuracy | frr | fa/h |
|------|--------|---------:|-----|------|
| test_229 | 你好问问 | 98.05% | 1.95% | 0.37 |
| test_229 | 嗨小问 | 98.13% | 1.87% | 0.98 |
| test_229_int8 | 你好问问 | 97.97% | 2.03% | 0.45 |
| test_229_int8 | 嗨小问 | 97.82% | 2.18% | 0.99 |

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
