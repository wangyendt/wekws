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

## 9. 2026-03-25 修订结论（基于当前 online/debug 双模式实现）

本节覆盖上文中“懒物化（lazy allocate）仍在使用”的旧表述。当前代码已经进一步收敛为**调试模式专用 hypotheses** 设计：

- `online` 模式默认关闭 `get_hypothesis/get_hypotheses` 物化能力
- `debug` 模式显式打开时，在 `init` 阶段一次性分配调试缓存
- **不再存在运行中首次调用调试接口才补 malloc 的 lazy init 行为**

### 9.1 当前实现的模式划分

当前 `CTCDecoderCConfig` 新增：

- `enable_debug_hypotheses = 0`：默认 online 模式
- `enable_debug_hypotheses = 1`：显式 debug 模式

两种模式的边界如下：

| 模式 | `get_hypothesis/get_hypotheses` | `init` 后新增分配 | 用途 |
|---|---|---|---|
| `online` | 不可用（直接返回失败 / 抛错） | **无** | 端侧真实唤醒检测 |
| `debug` | 可用 | **无**（调试缓存已在 `init` 分好） | 单测、对齐、分叉排查 |

也就是说，当前实现已经满足：

- `ctc_decoder_c_init()` 完成后，逐帧 `advance_frame / execute_detection / get_best_decode / reset_beam_search` 路径不再触发新增堆分配
- `set_keywords()` / `set_thresholds()` 仍然属于初始化/配置阶段，会为关键词与阈值表分配存储，但这不属于运行期逐帧检测内存增长

### 9.2 当前两档内存

#### 9.2.1 Online 模式（关闭调试）

配置：

- `score_beam_size = 3`
- `path_beam_size = 20`
- `max_prefix_len = 64`
- `enable_debug_hypotheses = 0`

对应内存：

- decoder heap: `118,520 B`（约 `115.74 KiB`）
- 若连 demo 的 keyword/threshold 小块一起算：`118,672 B`（约 `115.89 KiB`）

#### 9.2.2 Debug 模式（打开调试 hypotheses）

在 online 模式基础上，`init` 时额外分配：

- `cur_node_storage = 15,360 B`

因此：

- decoder heap: `133,880 B`（约 `130.74 KiB`）
- 若连 demo 的 keyword/threshold 小块一起算：`134,032 B`（约 `130.89 KiB`）

### 9.3 190KB → 130KB → 118KB 分别去掉了什么

这一轮压缩现在可以拆成两步看。

#### 9.3.1 190KB → 130KB：去掉 `next_node_storage`

旧的 190KB 版本（`P=20, M=64`）里，常驻物化节点缓存有两块：

- `cur_node_storage = 15,360 B`
- `next_node_storage = 61,440 B`

当前 debug 模式只保留：

- `cur_node_storage = 15,360 B`

不再保留：

- `next_node_storage = 61,440 B`

因此：

- `195,320 B -> 133,880 B`
- 节省 `61,440 B`

这一步本质上是：

- 保留调试所需的当前帧 hypotheses 物化缓存
- 去掉 next hypotheses 的常驻物化缓存

#### 9.3.2 130KB → 118KB：关闭 `cur_node_storage`

在 debug 模式基础上，如果进一步切到纯 online 模式，则还能去掉：

- `cur_node_storage = 15,360 B`

因此：

- `133,880 B -> 118,520 B`
- 再节省 `15,360 B`

所以最终关系非常直接：

- **190KB -> 130KB**：去掉 `next_node_storage`
- **130KB -> 118KB**：再去掉 `cur_node_storage`

### 9.4 当前 online 模式下，哪些内存仍然是大头

在 `online` 模式下，主要常驻内存户已经变成：

| 分配项 | 字节数 | 说明 |
|---|---|---|
| `node_pool` | `61,440 B` | 当前最大的在线必需内存 |
| `next_prefix_storage` | `20,480 B` | next beam 前缀 token |
| `next_node_ref_storage` | `20,480 B` | next beam node 引用，同时兼作 remap 区 |
| `cur_prefix_storage` | `5,120 B` | current beam 前缀 token |
| `cur_node_ref_storage` | `5,120 B` | current beam node 引用 |
| `cur_hyps + next_hyps + topk/temp_prefix` | `5,880 B` | 小头 |

也就是说，真正还值得继续压的 online 内存，已经主要集中在：

1. `node_pool`
2. `next_prefix_storage`
3. `next_node_ref_storage`

调试缓存已经不是大头，且已经被模式化隔离。

### 9.5 当前设计的副作用与取舍

当前版本相较于旧 190KB 版本，关键取舍是：

- 优点：online 模式 `init` 后不再有 hypotheses 相关的新增分配，更适合端侧“初始化后禁止动态分配”的约束
- 优点：调试模式仍保留 `get_hypothesis/get_hypotheses`，单测和对齐工具不需要重写
- 代价：如果需要调试 hypotheses，必须显式打开 `enable_debug_hypotheses`，并接受多出约 `15 KiB` 的常驻内存

因此，当前推荐使用方式是：

- 端侧 release：`enable_debug_hypotheses = 0`
- 本地验证/对齐/排障：`enable_debug_hypotheses = 1`

### 9.6 当前最终判断

截至当前实现，结论应更新为：

- **已经没有 lazy init**
- 已经形成清晰的 `online/debug` 双模式边界
- `online` 模式下，decoder heap 已稳定收敛到约 **115.74 KiB**
- `debug` 模式下，decoder heap 为约 **130.74 KiB**
- 相对原始 190KB 版本，当前压缩的本质来源是：
  - 去掉 `next_node_storage`
  - 并在 online 模式下进一步去掉 `cur_node_storage`

换句话说：

- 若目标是端侧真实部署，应以 **118KB online 模式** 为准
- 若目标是本地调试与 hypothesis 对齐，应以 **134KB debug 模式** 为准
- 下一步若还要继续压缩，优先目标应转向 `node_pool / next_prefix_storage / next_node_ref_storage`，而不是再从调试缓存里找空间
