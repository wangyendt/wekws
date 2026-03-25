# CTC Decoder 流式唤醒检测器 — 堆内存分析、压缩方案与当前结论

> 项目: wekws_baseline / examples/hi_xiaowen/s0/torch2lite
> 文件: `ctc_decoder_c.h`, `ctc_decoder_c.cc`, `ctc_decoder_c_demo.c`
> 日期: 2026-03-25
> 会话: 01947081

## 1. 背景

流式 CTC beam search 唤醒词检测器（C 接口）部署在嵌入式平台（Hi3861 等），需要对堆内存进行审计和优化。本文记录两次讨论的完整分析过程和结论。

## 2. 堆内存全景分析

### 2.1 关键参数（demo 默认配置）

| 参数 | 值 | 含义 |
|---|---|---|
| `score_beam_size (B)` | 3 | 每帧选取的 top-k token 数 |
| `path_beam_size (P)` | 20 | beam search 保留的假设路径数 |
| `max_frames` | 250 | 关键词最大帧数 |
| `interval_frames` | 50 | 检测间隔帧数 |
| `max_prefix_len (M)` | 314 | = max(250+64, 64) |

### 2.2 中间计算量

```
cur_hyp_capacity (Cp)     = P = 20
next_hyp_capacity (Np)    = P × (B+1) = 80
prefix_slots_cur          = Cp × M = 6,280
prefix_slots_next         = Np × M = 25,120
node_pool_capacity        = Np × M × 2 = 50,240
```

### 2.3 Level 0 — 当前分配明细（64-bit 平台）

| # | 分配项 | 元素数 | 单元素 | 字节数 |
|---|---|---|---|---|
| 1 | `cur_hyps` | 20 | 56B | 1,120 |
| 2 | `next_hyps` | 80 | 56B | 4,480 |
| 3 | `cur_prefix_storage` | 6,280 | 4B | 25,120 |
| 4 | `next_prefix_storage` | 25,120 | 4B | 100,480 |
| 5 | `cur_node_storage` | 6,280 | 12B | 75,360 |
| 6 | `next_node_storage` | 25,120 | 12B | 301,440 |
| 7 | `cur_node_ref_storage` | 6,280 | 4B | 25,120 |
| 8 | `next_node_ref_storage` | 25,120 | 4B | 100,480 |
| 9 | `topk_probs + topk_indices` | 6 | 4B | 24 |
| 10 | `temp_prefix` | 314 | 4B | 1,256 |
| 11 | `node_pool` | 50,240 | 12B | **602,880** |
| 12 | `node_pool_tmp` | 50,240 | 12B | **602,880** |
| 13 | `node_ref_remap` | 50,240 | 4B | **200,960** |
| | **init 合计** | | | **2,041,600** |
| | `set_keywords` | | | **120** |
| | **总计** | | | **2,041,720 (~1.95 MB)** |

### 2.4 内存大户

| 分配 | 占比 | 用途 |
|---|---|---|
| `node_pool` + `node_pool_tmp` | 59% | 节点池 + 压缩临时缓冲 |
| `next_node_storage` | 14.8% | next 假设的物化节点 |
| `next_prefix_storage` | 4.9% | next 假设的前缀 token |

**三大户合计占 init 内存的 78.7%。**

## 3. 复用分析与安全优化

### 3.1 node_pool ×2 安全余量可以去掉

`ctc_decoder_c_init` 第 365 行:
```c
state->node_pool_capacity = state->next_hyp_capacity * state->config.max_prefix_len * 2;
```

每次帧结束 `compact_node_pool` 清理所有未被 cur_hyps 引用的节点。压缩后存活节点 ≤ `P × M = 6,280`，下一帧新增 ≤ `Np = 80`，峰值 ≈ 6,360，远小于 50,240。×2 余量完全多余。

### 3.2 node_pool_tmp → 复用 next_node_storage

`node_pool_tmp` 仅在 `compact_node_pool` 内使用。`compact_node_pool` 在 `finalize_frame` 末尾调用，此时 `next_hyps` 数据已拷贝到 `cur_hyps`，`next_node_storage` 不再被读取。去掉 ×2 后两者尺寸匹配（均为 `Np × M` 个 `CTCDecoderCTokenNode`），可直接复用。

### 3.3 node_ref_remap → 复用 next_node_ref_storage

同理，`node_ref_remap` 仅在 `compact_node_pool` 内使用，与 `next_node_ref_storage` 使用时段不重叠。去掉 ×2 后两者尺寸匹配（均为 `Np × M` 个 `int32_t`），可直接复用。

### 3.4 物化节点存储可去掉（如不需要 get_hypothesis）

`cur_node_storage` / `next_node_storage` 存储物化后的 `CTCDecoderCTokenNode`，仅在 `ctc_decoder_c_get_hypothesis()` 时需要。在流式唤醒检测场景下，检测逻辑直接通过 `node_pool[hyp->node_refs[i]]` 访问节点。如不需要 `get_hypothesis()` 接口，这两个缓冲区可省掉。

## 4. Level 1 — 安全复用方案（推荐）

改动：去掉 node_pool ×2 + `node_pool_tmp` 复用 `next_node_storage` + `node_ref_remap` 复用 `next_node_ref_storage`。

| 分配项 | Level 0 | Level 1 | 节省 |
|---|---|---|---|
| node_pool | 602,880 | 301,440 | -301,440 |
| ~~node_pool_tmp~~ | 602,880 | **0** (复用) | -602,880 |
| ~~node_ref_remap~~ | 200,960 | **0** (复用) | -200,960 |
| 其余 | 636,000 | 634,880 | -1,120 |
| **init 合计** | **2,041,600** | **936,320** | **-1,105,280** |
| **总计** | **~1.95 MB** | **~914 KB** | **-54%** |

**精度影响：零。** 纯代码结构优化，不改变任何算法语义。

## 5. 参数敏感度分析

在 Level 1 基线（936KB）上，总内存公式：

```
Total ≈ P·H·(B+2) + 4·M·(8·P·B + 13·P + 1) + 8·B
```

| 参数 | 灵敏度 (B=3, P=20, M=314) | 说明 |
|---|---|---|
| **M** (max_prefix_len) | **~2,964 B/unit** | 最大杠杆，几乎所有分配 ×M |
| P (path_beam_size) | ~46,752 B/unit | 线性影响 cur/next 容量 |
| B (score_beam_size) | ~202,088 B/unit | 通过 Np=P·(B+1) 间接影响 |

### 为什么 max_prefix_len=314 是过分配

当前逻辑把 **beam search 前缀容量** 和 **检测时长**（max_frames=250）绑定：
```c
max_prefix_len = max(max_frames + 64, 64)  // = 314
```

但两者是不同概念：

| 概念 | 含义 | 实际值 |
|---|---|---|
| `max_frames` | 关键词发音的最大帧数 | 250 帧 ≈ 2.5s |
| `max_prefix_len` | 假设中非 blank token 的**最大个数** | 314 ← **严重过大** |

在 KWS 场景下：
- 最长关键词 "你好问问" = **4 个 token**
- idxset 过滤仅 {3,6,9,11,16} 通过，绝大多数帧是 blank
- `interval_frames=50` 周期性触发检测/reset
- **实际上前缀几乎不可能超过 20-30 个 token**

M=64 已有 3×+ 安全余量，M=32 仍然安全。

## 6. 压缩方案总表

所有方案均在 Level 1（安全复用）基础上。

### 方案 A：L1 + M=64（推荐，精度零损失）

| 分配项 | Level 1 (M=314) | 方案 A (M=64) | 节省 |
|---|---|---|---|
| cur/next prefix_storage | 125,600 | 25,600 | 100,000 |
| cur/next node_storage | 376,800 | 76,800 | 300,000 |
| cur/next node_ref_storage | 125,600 | 25,600 | 100,000 |
| temp_prefix | 1,256 | 256 | 1,000 |
| node_pool | 301,440 | 61,440 | 240,000 |
| 其余 (hyps, topk) | 6,944 | 6,944 | 0 |
| **总计** | **936,440** | **195,440** | **741,000** |

**结果：~191 KB，比原始 2MB 缩减 90.5%**

精度风险：**零**。前缀超过 64 token = 64 个非 blank keyword token 连续累积而不触发检测，在 idxset=6 候选 + interval=50 约束下不可能。

### 方案 B：L1 + M=64 + P=10

| 分配项 | 方案 A | 方案 B | 节省 |
|---|---|---|---|
| cur_hyps | 1,120 | 560 | 560 |
| next_hyps (Np=40) | 4,480 | 2,240 | 2,240 |
| prefix/node/ref (cur) | 25,600 | 12,800 | 12,800 |
| prefix/node/ref (next) | 102,400 | 51,200 | 51,200 |
| node_pool (40×64) | 61,440 | 30,720 | 30,720 |
| scratch | 1,280 | 1,280 | 0 |
| **总计** | **195,440** | **97,920** | **97,520** |

**结果：~96 KB，比原始 2MB 缩减 95.2%**

精度风险：**极低**。5 个非 blank 候选 token 下，10 条 beam 绰绰有余。

### 方案 C：L1 + M=64 + P=10 + B=2

Np = 10×(2+1) = 30

| 分配项 | 方案 B | 方案 C | 节省 |
|---|---|---|---|
| next_hyps (Np=30) | 2,240 | 1,680 | 560 |
| prefix/node/ref (next) | 51,200 | 38,400 | 12,800 |
| node_pool (30×64) | 30,720 | 23,040 | 7,680 |
| topk | 24 | 16 | 8 |
| **总计** | **97,920** | **76,872** | **21,048** |

**结果：~75 KB，比原始 2MB 缩减 96.2%**

精度风险：**低但非零**。B=2 意味着每帧最多考虑 1 个非 blank token，偶尔错过第 3 高概率 token，但关键词 token 在连续多帧保持高概率，通常不影响最终检测。

## 7. 全景汇总

| 方案 | 堆内存 | vs 原始 | 精度风险 | 改动范围 |
|---|---|---|---|---|
| Level 0（当前） | ~1.95 MB | — | — | 无 |
| Level 1（安全复用） | ~914 KB | -54% | 零 | 改 1 行容量公式 + compact 换指针 |
| **A: L1 + M=64** | **~191 KB** | **-90.5%** | **零** | 设 config.max_prefix_len=64 |
| B: L1 + M=64 + P=10 | ~96 KB | -95.2% | 极低 | 调 path_beam_size |
| C: L1 + M=64 + P=10 + B=2 | ~75 KB | -96.2% | 低 | 调 score_beam_size |
| 极限 (L2 + M=32) | ~60 KB | -97% | 需验证 | 去物化 + M=32 |

## 8. 结论

**推荐方案 A**：Level 1 安全复用 + `max_prefix_len=64`。

- 一行配置改动 + 小幅代码重构，堆内存从 **1.95 MB → 191 KB**（-90.5%）
- 精度完全不变
- 原因：`max_prefix_len=314` 是为通用 ASR 设计的（前缀=完整转录），KWS 场景下前缀最多十几个 token，314 是数量级的浪费

如需进一步压到 ~96KB，方案 B（P=10）也基本安全，但建议用真实测试集验证。

## 9. 2026-03-25 修订结论（基于当前 190KB 代码）

本节结论覆盖上文中“方案 B 基本安全”这一类探索性表述，原因是后续已经基于真实代码路径补做了更严格的 smoke 对比。

### 9.1 当前仓库基线

当前源码已回到提交 `0c2b044`，也就是：

- Level 1 安全复用已生效
- `max_prefix_len = 64` 已生效
- `path_beam_size = 20`
- `score_beam_size = 3`

对应内存约为：

- decoder heap: `195,320 B`（约 `190.74 KiB`）
- 若连 demo 的 keyword/threshold 小块一起算：`195,472 B`（约 `190.89 KiB`）

### 9.2 基于当前 190KB 版本，真正“无损”的优化空间

这里的“无损”指：

- 不改变 beam search 搜索空间
- 不改变在线检测语义
- 不引入可观测的检测结果分叉

按收益排序，当前最值得做的有两类。

#### 9.2.1 最大的通用无损空间：去掉常驻 `hyp->nodes` 物化缓存，改成懒物化

当前代码固定分配了两块物化节点缓存：

- `cur_node_storage`
- `next_node_storage`

在当前 `P=20, M=64` 下，它们一共占：

- `15,360 + 61,440 = 76,800 B`

但在线唤醒主路径并不依赖这两块：

- 检测逻辑直接读取 `node_pool[hyp->node_refs[i]]`
- `get_best_decode()` 不依赖 `hyp->nodes`
- 真正会触发物化的，是 `ctc_decoder_c_get_hypothesis()` / pybind `get_hypotheses()` 这类调试与对齐接口

因此，更合理的实现是：

- 默认不分配 `cur_node_storage/next_node_storage`
- 仅在第一次调用 `get_hypothesis/get_hypotheses` 时，再临时分配并物化

这样做的收益是：

- decoder heap: `195,320 B -> 118,520 B`
- 节省 `76,800 B`（约 `75.0 KiB`）

这一步对在线检测结果可视为真正无损，代价只是调试接口从“常驻缓存”改成“按需构造”。

#### 9.2.2 项目绑定的无损空间：内部存储类型窄化

如果 decoder 只服务当前 hi_xiaowen 场景，而不是继续追求通用性，还可以进一步压：

1. `prefix_storage`: `int32_t -> uint16_t`
2. `node_ref_storage`: `int32_t -> uint16_t`
3. `CTCDecoderCTokenNode` 中的 `token/frame`: `int32_t -> uint16_t`

原因是当前上界非常小：

- token id 远小于 `65535`
- `node_pool_capacity = Np * M = 80 * 64 = 5120`
- `frame` 在当前配置下也远小于 `65535`

这类改动的收益量级大致是：

- `prefix_storage` 窄化：约省 `12.8 KiB`
- `node_ref_storage` 窄化：约省 `12.8 KiB`
- `token/frame` 窄化：还能继续省一截

但这类优化只对当前项目可视为无损，不再是“通用 C decoder 无损”。后续如果扩大词表、扩大帧范围、复用到其他模型，就必须重新审边界。

### 9.3 哪些量不应再称为“无损”

以下项目虽然还能继续降内存，但不应再归类为严格无损：

- `path_beam_size`
- `score_beam_size`
- `max_prefix_len`

原因：

- `path_beam_size` / `score_beam_size` 直接改变 beam search 搜索空间
- `max_prefix_len` 改变单条 prefix 的可达长度上限

特别是 `P=10` 这条路，虽然理论内存能到约 `96KB`，但后续用真实代码路径做的 1500 条 mixed smoke 已经出现可观测分叉，因此它更适合归类为“有风险的进一步压缩”，而不是“无损优化”。

### 9.4 更新后的建议顺序

如果目标是继续压内存，但不接受精度或行为风险，建议顺序如下：

1. 先做 `hyp->nodes` 常驻缓存懒分配
2. 如果还不够，再做 16-bit 内部存储窄化，并把边界检查写死
3. 只有在前两步仍不够时，才去碰 `path_beam_size / score_beam_size / max_prefix_len`

### 9.5 当前最终判断

基于当前 190KB 版本，仍存在一块很可观的、真正无损的优化空间：

- 首选目标是去掉常驻 `cur_node_storage/next_node_storage`
- 理论上可直接把 decoder heap 从约 `190.74 KiB` 压到约 `115.74 KiB`
- 若再叠加项目绑定的 16-bit 存储窄化，还能继续往下走

换句话说：

- `190KB -> 118KB` 这一步，有希望做到真正无损
- `190KB -> 96KB` 这一步，不能再简单表述为无损或近乎无损默认方案
