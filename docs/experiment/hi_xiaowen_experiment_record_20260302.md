# hi_xiaowen 实验记录（截至 2026-03-02）

## 0. 你关心的两个 test_299 实验：具体做了什么尝试

### 0.1 `exp/fsmn_ctc_distill_v2_a64_p32_l2/test_299`

对应命令（`docs/常用命令.txt`）：

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

这个实验的关键尝试：

1. 学生结构尝试路线里选择了最小的 `a64_p32_l2`（v2 系列里还有 `a80_p48_l3`、`a64_p32_l3`，见常用命令同段）。
2. 训练策略是两阶段：先 200 epoch feature alignment（HEAD 冻结），再 100 epoch finetune（HEAD 解冻）。
3. 用 `layer_mapping=0:2,1:3` 做跨层蒸馏对齐。
4. 学习率策略用 `plateau`，并且从 epoch 200 才开始调度（`scheduler_start_epoch=200`）。
5. `2 3` 表示跑训练并做 Stage 3 评测，所以直接产出了 `test_299`。

参数量证据（log 前 400 行可见）：

- `Student Model Parameters: Total params = 113,142`
- `Backbone=110,322`, `Head=2,820`

结果（`python analyze_exp_test_stats.py --test-id 299`）：

- 你好问问：`97.30%`，FRR `2.70%`，FA `0.70/h`
- 嗨小问：`96.42%`，FRR `3.58%`，FA `0.99/h`

---

### 0.2 `exp/fsmn_ctc_distill_v2_a64_p32_l2_ft_from199_lr1e4/test_299`

对应命令（`docs/常用命令.txt`）：

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

这个实验的关键尝试：

1. 不是从头训，而是**从上一实验的 `199.pt` 继续**（再蒸馏/微调）。
2. 固定学习率到 `1e-4`（`resume_lr=1e-4`, `finetune_lr=1e-4`）。
3. 关闭调度器（`lr_scheduler=none`），减少学习率变化因素。
4. `2 2` 只跑 Stage 2；测试 `test_299` 是后续单独评估拿到的。
5. 日志中可见恢复后在 epoch 200 进入 finetune（HEAD 解冻，head lr = `1e-5`）。

参数量证据：

- 与上一个实验结构相同，仍是 `113,142` 参数。

结果（`test_299`）：

- 你好问问：`97.30%`，FRR `2.70%`，FA `0.99/h`
- 嗨小问：`96.04%`，FRR `3.96%`，FA `0.99/h`

对比结论：

1. 这次从 `199.pt` 继续 + 固定 `1e-4`，没有带来明显收益。
2. 相比从头跑到 299 的版本，`嗨小问`略降，`你好问问`精度持平但 FA 变差。

---

## 1. 核心结论总结

### 1.1 有效尝试 ✅

| 方法 | 参数量变化 | 精度表现 | 结论 |
|---|---:|---|---|
| 权重手术（Weight Surgery） | 756,133 → 392,494 | 98%+（test_79） | ✅ 最可靠的压缩方法 |
| 知识蒸馏（mini-align） | 392,494 → 199,760 | 179轮偏低，229轮恢复到~98% | ⚠️ 可行但训练成本高、对训练策略敏感 |
| INT8 PTQ | 参数个数不变（199,760），位宽由 FP32→INT8 | 229: 98.05/98.13 → 229_int8: 97.97/97.82 | ✅ 量化损失可接受 |

> 备注：若按“等效 FP32 存储”估算，INT8 下约等价于 `~50K` FP32 参数体积；
> 量化日志里实际文件大小是 `793.1KB -> 351.2KB`（TorchScript zip）。

### 1.2 失败尝试 ❌

| 方法 | 问题 |
|---|---|
| 随机初始化 HEAD（`strict=false` 且输出层形状变化） | 关键词召回崩塌（top2598/top440 在 test_2 几乎全失效） |
| 简单裁剪（无权重手术） | 从 2599 裁到 2598 仍出现精度断崖，FRR 接近 100% |
| 早期蒸馏配置（如 test_179） | 199K 学生在不合适阶段/超参下表现不稳（嗨小问仅 91.50%） |

### 1.3 核心洞察

- 预训练/迁移后的 HEAD 是关键能力来源：它提供更尖锐且更可分的后验分布，尤其对共享 token（如“问”）的抑制混淆能力至关重要。
- 直接改输出维度若丢失头部权重（随机头），CTC 解码容易在“嗨小问/你好问问”之间误判。
- 因此路线应优先：`权重手术保头` 或 `蒸馏时显式复制/约束 HEAD`。

---

## 2. 当前已实现实验全景（有效/无效）

### 2.1 2599 baseline 与简单裁剪对比（无权重手术）

| 实验 | 关键词 | accuracy | frr | fa/h | 结论 |
|---|---|---:|---:|---:|---|
| baseline_4gpus/test_2 | 你好问问 | 96.26% | 3.74% | 0.41 | 基线正常 |
| baseline_4gpus/test_2 | 嗨小问 | 96.96% | 3.04% | 0.93 | 基线正常 |
| top2598/test_2 | 你好问问 | 0.00% | 100.00% | 0.00 | 失败 |
| top2598/test_2 | 嗨小问 | 1.12% | 98.88% | 0.93 | 失败 |
| top440/test_2 | 你好问问 | 0.67% | 99.33% | 0.86 | 失败 |
| top440/test_2 | 嗨小问 | 1.00% | 99.00% | 0.99 | 失败 |
| top2599/test_2 | 你好问问 | 96.71% | 3.29% | 0.58 | 接近基线 |
| top2599/test_2 | 嗨小问 | 96.10% | 3.90% | 0.99 | 接近基线 |

做法尝试：

1. `gen_reduced_dict.py` 重建小词表（top2598/top440）。
2. `run_fsmn_ctc.sh` 里 `checkpoint_strict=false` 以允许维度不匹配继续训练。
3. 但无权重手术时输出层出现随机初始化风险，导致召回崩塌。

### 2.2 权重手术：最稳定的压缩手段

| 实验 | 关键词 | accuracy | frr | fa/h | 结论 |
|---|---|---:|---:|---:|---|
| baseline_4gpus/test_79 | 你好问问 | 98.05% | 1.95% | 0.28 | 对照 |
| baseline_4gpus/test_79 | 嗨小问 | 98.38% | 1.62% | 0.64 | 对照 |
| top20_weight_surgery/test_79 | 你好问问 | 98.63% | 1.37% | 0.32 | 有效 |
| top20_weight_surgery/test_79 | 嗨小问 | 98.88% | 1.12% | 0.67 | 有效 |

做法尝试：

1. 训练时使用 `checkpoint_strict=false` + `dict_top20`。
2. 在 `wekws/utils/checkpoint.py` 中对 `out_linear2` 按 token 做行映射复制（weight surgery）。
3. 日志可见 `copied 20/20 rows`，即头部权重完整迁移。

参数变化：

- baseline `756,133` → top20 surgery `392,494`（约 -48%）。

### 2.3 知识蒸馏（199K 学生）

| 实验 | 关键词 | accuracy | frr | fa/h | 结论 |
|---|---|---:|---:|---:|---|
| distill_mini_align_20_test2/test_179 | 你好问问 | 95.31% | 4.69% | 0.63 | 早期一般 |
| distill_mini_align_20_test2/test_179 | 嗨小问 | 91.50% | 8.50% | 0.67 | 偏弱 |
| distill_mini_align_20_test2/test_229 | 你好问问 | 98.05% | 1.95% | 0.37 | 恢复到高精度 |
| distill_mini_align_20_test2/test_229 | 嗨小问 | 98.13% | 1.87% | 0.98 | 恢复到高精度 |

做法尝试：

1. Feature alignment 两阶段蒸馏（先 MSE 对齐，再 MSE+CTC）。
2. 学生 backbone 从 teacher 切片初始化，HEAD 从 teacher 拷贝并冻结/再解冻。
3. 通过继续训练和参数调整，从 179 提升到 229。

参数量：

- `199,760`（Backbone `196,940` + Head `2,820`）。

### 2.4 distill v2（113K 学生）与从 199.pt 再蒸馏

v2 架构尝试（来自常用命令）：

1. `a80_p48_l3`（配置注释参数约 164,768）
2. `a64_p32_l3`（约 129,776）
3. `a64_p32_l2`（约 113,142）

最终关注 `a64_p32_l2`：

| 实验 | 关键词 | accuracy | frr | fa/h | 结论 |
|---|---|---:|---:|---:|---|
| distill_v2_a64_p32_l2/test_299 | 你好问问 | 97.30% | 2.70% | 0.70 | 有效 |
| distill_v2_a64_p32_l2/test_299 | 嗨小问 | 96.42% | 3.58% | 0.99 | 有效但较 199K 略降 |
| distill_v2_a64_p32_l2_ft_from199_lr1e4/test_299 | 你好问问 | 97.30% | 2.70% | 0.99 | 持平/FA变差 |
| distill_v2_a64_p32_l2_ft_from199_lr1e4/test_299 | 嗨小问 | 96.04% | 3.96% | 0.99 | 略退化 |

结论：

- 113K 量级能保持中高精度，但相比 199K 还有可见回退。
- “从 199.pt 继续 + 固定 1e-4”这次没有比从头训到 299 更好。

### 2.5 PTQ 量化（INT8）

| 实验 | 关键词 | accuracy | frr | fa/h | 结论 |
|---|---|---:|---:|---:|---|
| distill_mini_align_20_test2/test_229 | 你好问问 | 98.05% | 1.95% | 0.37 | FP32 基线 |
| distill_mini_align_20_test2/test_229 | 嗨小问 | 98.13% | 1.87% | 0.98 | FP32 基线 |
| distill_mini_align_20_test2/test_229_int8 | 你好问问 | 97.97% | 2.03% | 0.45 | 轻微损失 |
| distill_mini_align_20_test2/test_229_int8 | 嗨小问 | 97.82% | 2.18% | 0.99 | 轻微损失 |

已做的量化尝试：

1. INT8、INT16 两条线都实现并可评估。
2. 校准样本量尝试（200、500）与不同校准数据集（train/dev）。
3. 额外打通了 ExecuTorch PT2E 导出与多 backend 选项（xnnpack/portable/hifi4）。

---

### 2.6 蒸馏 v3（merged HEAD + 3 层）S1-S5 横向结果

这 5 组实验做的事情本质一致，都是在蒸馏 v3 路线上继续压缩学生 backbone，只改变学生宽度；公共训练设置如下（见 `docs/常用命令.txt` 第 122-139 行）：

1. teacher 固定为 `exp/fsmn_ctc_top20_weight_surgery/79.pt`，即 top20 权重手术后的高精度老师。
2. 词表固定为 `dict_top20`，输出维度固定为 20。
3. 训练策略固定为 `align_epochs=300 + finetune_epochs=100`，比 v2 的 `200 + 100` 更长。
4. 蒸馏对齐固定为 `layer_mapping="0:1,1:2,2:3"`，对应 3 层学生去对齐老师后 3 层。
5. 学习率策略固定为 `finetune_lr=1e-4`、`head_lr_ratio=0.1`、`lr_scheduler=none`。
6. 结构上都启用了 `merge_head=true`，目标是把 HEAD 融入更紧凑的 3 层学生结构，而不是沿用 v2 的独立较大头部。

每个实验具体改了什么、参数量多少：

| 实验 | student_config | 具体改动 | 参数量 |
|---|---|---|---:|
| S1 | `s1_a64_p32_l3_merged` | 3 层 merged 结构里最大的学生，`input_affine_dim=64`、`proj_dim=32` | ~96,836 |
| S2 | `s2_a56_p28_l3_merged` | 在 S1 基础上缩小宽度到 `56/28` | ~85,484 |
| S3 | `s3_a48_p24_l3_merged` | 继续缩小宽度到 `48/24` | ~74,132 |
| S4 | `s4_a40_p20_l3_merged` | 继续缩小宽度到 `40/20` | ~62,780 |
| S5 | `s5_a32_p16_l3_merged` | 最小版本，宽度缩到 `32/16` | ~51,428 |

参数量证据来自对应配置文件头注释：

- `conf/fsmn_ctc_student_s1_a64_p32_l3_merged.yaml`
- `conf/fsmn_ctc_student_s2_a56_p28_l3_merged.yaml`
- `conf/fsmn_ctc_student_s3_a48_p24_l3_merged.yaml`
- `conf/fsmn_ctc_student_s4_a40_p20_l3_merged.yaml`
- `conf/fsmn_ctc_student_s5_a32_p16_l3_merged.yaml`

结果（`python analyze_exp_test_stats.py --test-id 399`）：

| 实验 | 参数量 | 关键词 | threshold | accuracy | frr | fa/h | 备注 |
|---|---:|---|---:|---:|---:|---:|---|
| distill_s1_a64_p32_l3_merged/test_399 | 96,836 | 你 好 问 问 | 0.000 | 97.58% | 2.42% | 0.54 | `legacy:fa<=target(1.0)` |
| distill_s1_a64_p32_l3_merged/test_399 | 96,836 | 嗨 小 问 | 0.338 | 96.86% | 3.14% | 0.99 | `legacy:fa<=target(1.0)` |
| distill_s2_a56_p28_l3_merged/test_399 | 85,484 | 你 好 问 问 | 0.000 | 97.45% | 2.55% | 0.45 | `legacy:fa<=target(1.0)` |
| distill_s2_a56_p28_l3_merged/test_399 | 85,484 | 嗨 小 问 | 0.417 | 96.44% | 3.56% | 0.99 | `legacy:fa<=target(1.0)` |
| distill_s3_a48_p24_l3_merged/test_399 | 74,132 | 你 好 问 问 | 0.016 | 97.66% | 2.34% | 0.53 | `legacy:fa<=target(1.0)` |
| distill_s3_a48_p24_l3_merged/test_399 | 74,132 | 嗨 小 问 | 0.272 | 96.75% | 3.25% | 0.99 | `legacy:fa<=target(1.0)` |
| distill_s4_a40_p20_l3_merged/test_399 | 62,780 | 你 好 问 问 | 0.000 | 96.98% | 3.02% | 0.67 | `legacy:fa<=target(1.0)` |
| distill_s4_a40_p20_l3_merged/test_399 | 62,780 | 嗨 小 问 | 0.423 | 95.53% | 4.47% | 0.99 | `legacy:fa<=target(1.0)` |
| distill_s5_a32_p16_l3_merged/test_399 | 51,428 | 你 好 问 问 | 0.000 | 96.83% | 3.17% | 0.93 | `legacy:fa<=target(1.0)` |
| distill_s5_a32_p16_l3_merged/test_399 | 51,428 | 嗨 小 问 | 0.520 | 93.40% | 6.60% | 0.98 | `legacy:fa<=target(1.0)` |

阶段性观察：

1. S1-S3 都维持在两个关键词 `96%+` 的可用区间，说明 merged HEAD + 3 层蒸馏路线在 `74K-97K` 参数段内是成立的。
2. `你 好 问 问` 上最好的是 S3（`97.66%`），且参数量比 S1 少约 `22.7K`；说明这条线上不是“越大越好”，中等宽度有更优甜点。
3. `嗨 小 问` 上最好的是 S1（`96.86%`），S3 非常接近（`96.75%`）；如果更看重综合平衡，S3 更像当前最佳折中点。
4. 从 S3 往下继续压到 S4/S5 后退化明显，尤其 S5 的“嗨小问”降到 `93.40%`，说明 `32/16` 这一档已经过窄。
5. 相比 `distill_v2_a64_p32_l2/test_299`（113,142 参数），S1 和 S3 都在更少参数下取得更好结果；其中 S3 约 `74K` 参数，已经明显优于 v2-113K。
6. 这些结果都来自 `test_399`，因此和前文 `test_299` 的 v2 结论只能做趋势参考；若要严格结论，仍建议补一轮同测试集对齐评测。

---

## 3. 方法-尝试-结果对应表（便于复盘）

| 方法 | 具体做法 | 已做尝试 | 当前结论 |
|---|---|---|---|
| 简单裁剪词表 | `dict_top2598/top440` + `num_keywords` 改小 | 2598、440、2599 | 2598/440 失败，2599 正常 |
| 权重手术 | `checkpoint.py` 对 `out_linear2` 做 token 行拷贝 | top2598_weight_surgery、top440_weight_surgery、top20_weight_surgery | top20 最稳，显著减参且高精度 |
| 蒸馏（199K） | 两阶段 feature alignment + HEAD 复制冻结/解冻 | test_179、test_229 | 可达 98% 左右，但调参成本高 |
| 蒸馏 v2（113K） | 更小 backbone（a64_p32_l2）+ 200/100 训练 | 从头训、从199继续训 | 可用但略低于 199K |
| 蒸馏 v3（S1-S5 merged） | 合并 HEAD + 3 层结构蒸馏，仅缩学生宽度 | s1-s5_a{64,56,48,40,32}_p{32,28,24,20,16}_l3_merged/test_399 | S1-S3 可用，其中 S3（74K）是当前精度/参数最佳折中 |
| PTQ 量化 | INT8/INT16 + 校准 + evaluate | 229_int8、执行器导出路线 | INT8 损失可接受 |

---

## 4. 证据索引（关键文件）

1. 常用命令：`docs/常用命令.txt`
   - v2 从头：第 26 行
   - v2 从 199 继续：第 29 行
   - v3 S1-S5：第 122-139 行
2. 两个目标日志：
   - `exp/fsmn_ctc_distill_v2_a64_p32_l2/logs/run_distill_stage_2_3_20260228_205342.log`
   - `exp/fsmn_ctc_distill_v2_a64_p32_l2_ft_from199_lr1e4/logs/run_distill_stage_2_2_20260302_113805.log`
3. 参数量证据：
   - baseline: `exp/fsmn_ctc_baseline_4gpus/logs/run_stage_2_2_20260205_104707.log`（756,133）
   - top20 surgery: `exp/fsmn_ctc_top20_weight_surgery/logs/run_stage_2_2_20260206_195135.log`（392,494）
   - distill 199K: `exp/fsmn_ctc_distill_mini_align_20_test2/logs/run_distill_stage_2_2_20260210_130428.log`（199,760）
   - distill v2 113K: 上述两个 v2 日志（113,142）
   - distill v3 S1-S5: `conf/fsmn_ctc_student_s{1,2,3,4,5}_a*_merged.yaml` 文件头注释（96,836 / 85,484 / 74,132 / 62,780 / 51,428）
4. 统计脚本：`examples/hi_xiaowen/s0/analyze_exp_test_stats.py`
