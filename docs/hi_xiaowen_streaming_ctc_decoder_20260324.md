# hi_xiaowen 流式 CTC Decoder 说明

> 更新时间：2026-03-24  
> 适用代码：
> - `examples/hi_xiaowen/s0/infer_wav_stream.py`
> - `examples/hi_xiaowen/s0/torch2lite/ctc_decoder.cc`
> - `examples/hi_xiaowen/s0/torch2lite/ctc_decoder_pybind.cc`

---

## 1. 先说结论

- `ctc_decoder.cc` **不是**直接接收整段 wav。
- 它也**不是**直接接收整段 `N x V` 一次性完成后处理。
- 它的核心接口是逐帧的 `advance_frame(frame_index, probs)`，其中

$$
\mathbf{p}_t \in \mathbb{R}^{V}
$$

表示第 $t$ 帧的 **CTC 概率向量**，而不是原始 logits。

- 在线推理时，模型每产生一帧概率，就喂给 decoder 一次。
- decoder 内部维护少量 beam state，而**不是**保存全部历史帧再做大滑窗。
- 是否唤醒，不是由单帧 `1 \times V` 直接映射出来，而是由“逐帧 beam 累积 + 关键词匹配 + 分数/时长/间隔判定”共同决定。

---

## 2. 输入输出是什么

### 2.1 模型输出

设模型第 $t$ 帧输出 logits 为

$$
\mathbf{z}_t \in \mathbb{R}^{V},
$$

其中 $V$ 是词表大小，例如 20。

进入 decoder 前，先做 softmax：

$$
\mathbf{p}_t = \operatorname{softmax}(\mathbf{z}_t), \quad
\sum_{i=0}^{V-1} p_t(i) = 1.
$$

因此 C++ decoder 实际接收的是：

$$
\mathbf{p}_t = [p_t(0), p_t(1), \dots, p_t(V-1)].
$$

在当前实现里：

- TFLite step 模型每次常见输入是 `1 x 1 x 400` 特征 + cache。
- 模型输出一帧 logits，转成一帧 probs。
- Python 再把这帧 probs 调给 `advance_frame(...)`。

### 2.2 Decoder 输出

逐帧调用后，decoder 有两类输出：

1. 当前时刻触发结果 `execute_detection()` / `step_and_detect()`

若触发，返回：

- `keyword`
- `candidate_score`
- `start_frame`
- `end_frame`

若未触发，返回 `None`。

2. 当前整段历史的最佳候选 `get_best_decode()`

即到当前时刻为止见过的最佳关键词候选及分数。

---

## 3. 在线不是“存全量再滑窗”

很多人容易把在线流程理解成：

1. 一直保存历史所有帧的 `1 x V` 概率；
2. 周期性拿一个滑窗；
3. 在滑窗内重新 decode；
4. 看有没有唤醒词。

这**不是**当前实现。

当前实现更接近下面这个状态机：

1. 第 $t$ 帧到来，输入 $\mathbf{p}_t$；
2. 用 $\mathbf{p}_t$ 更新 beam hypotheses；
3. 立刻检查当前 hypotheses 是否已经包含关键词 token 序列；
4. 如果满足阈值、时长、间隔，就立即触发；
5. 然后继续吃第 $t+1$ 帧。

所以在线时保存的核心不是完整历史矩阵，而是：

- 当前 beam 内若干条 hypothesis；
- 每条 hypothesis 的 token prefix；
- prefix 中每个 token 对应的帧位置和概率；
- 上次触发位置 `last_active_pos`；
- 历史最佳结果 `best_decode`。

这就是“历史被压缩进状态里”的含义。

---

## 4. Beam State 在维护什么

对每个 hypothesis，decoder 维护：

$$
h = (\pi, p_b, p_{nb}, \mathcal{N}),
$$

其中：

- $\pi$：当前 token prefix；
- $p_b$：以 blank 结束的概率；
- $p_{nb}$：以 non-blank 结束的概率；
- $\mathcal{N}$：和 prefix 对齐的 token 节点序列，每个节点记录

$$
n_i = (\text{token}_i,\ \text{frame}_i,\ \text{prob}_i).
$$

decoder 每来一帧，只保留 top path beam，避免状态无限增长。

---

## 5. 每一帧是怎么更新的

设当前帧为 $t$，当前候选 token 为 $c$，其概率为 $p_t(c)$。

### 5.1 blank 扩展

若 $c = \epsilon$（当前代码里 blank id 为 0），则：

$$
p_b'(\pi) \leftarrow p_b'(\pi) + p_t(\epsilon)\bigl(p_b(\pi) + p_{nb}(\pi)\bigr).
$$

prefix 不变。

### 5.2 重复字符扩展

若 $c$ 等于 prefix 最后一个 token：

- 从 non-blank 延续到同一 prefix：

$$
p_{nb}'(\pi) \leftarrow p_{nb}'(\pi) + p_t(c)\, p_{nb}(\pi)
$$

- 从 blank 转到扩展后的 prefix：

$$
p_{nb}'(\pi + c) \leftarrow p_{nb}'(\pi + c) + p_t(c)\, p_b(\pi)
$$

### 5.3 普通非 blank 扩展

若 $c \neq \epsilon$ 且 $c$ 不等于最后一个 token，则：

$$
p_{nb}'(\pi + c) \leftarrow p_{nb}'(\pi + c) + p_t(c)\bigl(p_b(\pi) + p_{nb}(\pi)\bigr).
$$

更新后按

$$
p(\pi) = p_b(\pi) + p_{nb}(\pi)
$$

排序，只保留 path beam 内最优的若干条 hypothesis。

---

## 6. 怎么从逐帧概率变成“是否唤醒”

### 6.1 关键词匹配

设目标关键词对应 token 序列为

$$
\mathbf{k} = [k_1, k_2, \dots, k_M].
$$

对当前 hypothesis 的 prefix $\pi$，检查 $\mathbf{k}$ 是否是其一个连续子序列。

若不是，则该 hypothesis 当前不能触发。

若是，设匹配起点偏移为 $o$，则对应节点区间为：

$$
\mathcal{N}[o], \mathcal{N}[o+1], \dots, \mathcal{N}[o+M-1].
$$

### 6.2 分数计算

令这些节点的保留概率分别为

$$
q_1, q_2, \dots, q_M,
$$

则当前关键词候选分数定义为：

$$
s = \sqrt{\prod_{i=1}^{M} q_i }.
$$

这就是当前实现里的 `candidate_score`。

### 6.3 时长约束

设起止帧为

$$
t_s,\ t_e,
$$

则 duration 为

$$
d = t_e - t_s.
$$

只有当

$$
d_{\min} \le d \le d_{\max}
$$

时，才允许触发。

### 6.4 阈值与触发间隔

对关键词 $w$，其阈值记作 $\tau_w$。  
若未关闭 threshold，则要求

$$
s \ge \tau_w.
$$

同时还要满足和上一次触发的间隔约束。设上次触发结束帧为 $t_{\text{last}}$，则要求

$$
t_e - t_{\text{last}} \ge \Delta,
$$

其中 $\Delta$ 为 `interval_frames`。

### 6.5 最终布尔判定

因此“是否唤醒”本质上是下面这个布尔条件：

$$
\mathbb{1}_{\text{wake}} =
\mathbb{1}\Bigl[
\mathbf{k} \subset \pi
\ \land\
 s \ge \tau_w
\ \land\
 d_{\min} \le d \le d_{\max}
\ \land\
 (t_e - t_{\text{last}} \ge \Delta)
\Bigr].
$$

所以单帧 `1 x V` 并不会直接变成一个布尔值，而是要先进入 decoder 的历史状态累积之后，才能得到最终的唤醒判断。

---

## 7. 在线时应该怎么调用

在线推理的正确流程是：

```python
decoder = StreamingCTCDecoder(...)

abs_frame = 0
while True:
    chunk_wave = get_next_audio_chunk()
    feats = fbank.accept(chunk_wave)
    probs = model(feats).softmax(-1)  # shape [N, V]

    for i in range(probs.size(0)):
        result = decoder.step_and_detect(abs_frame, probs[i], disable_threshold=False)
        abs_frame += downsampling
        if result is not None:
            on_wakeup(result)
            decoder.reset_beam_search()
```

要点：

- 模型可以按 chunk 产出 `N x V`；
- 但 decoder 仍然是对 `N` 行逐行处理；
- 在线场景不需要保存全部历史 `N x V`；
- 唤醒后通常用 `reset_beam_search()` 重启搜索。

---

## 8. reset 的语义

当前 C++ 版本有两个 reset：

### 8.1 `reset()`

全量清空，包括：

- 当前 beam 状态；
- `last_active_pos`；
- `best_decode`。

适合“整段会话重新开始”。

### 8.2 `reset_beam_search()`

只清空当前 beam state，不清：

- `last_active_pos`
- `best_decode`

适合“本次已触发，重新开始搜下一个关键词”。

流式在线场景通常优先使用 `reset_beam_search()`。

---

## 9. 离线和在线的关系

离线和在线本质上跑的是同一个逐帧算法。

差别只在于：

- 离线：把整段音频所有帧一次性或分批全部喂完；
- 在线：边来边喂，喂到某一帧就可以提前触发。

因此离线可以看成在线算法跑到音频结尾的特例。

---

## 10. 199K 模型能不能复用这个 C++ decoder

结论：**可以复用，前提是后处理语义一致。**

当前仓库里的 199K 模型 `distill199`：

- 仍然是 CTC 输出；
- 仍然使用 `dict_top20`；
- 关键词 token 化仍然通过同一套 `build_keyword_token_info(...)`；
- 流式入口仍然走同一个 `StreamingKeywordSpotter`。

因此只要满足下面这些条件，199K 就可以直接复用同一个 C++ decoder：

1. blank id 仍然是 0；
2. 模型输出仍然是每帧一个 vocab 概率向量；
3. 关键词 token 编码方式不变；
4. `score_beam_size / path_beam_size / min_frames / max_frames / interval_frames / threshold_map` 与 Python 对照路径保持一致。

此时，C++ decoder 的结果应当与 Python 版本**理论上一致**，因为两边执行的是同一套逐帧 beam search 与关键词判定规则。

但这里要区分两件事：

- **理论兼容性**：199K 没有结构性障碍，可以直接复用；
- **实验一致性**：仍然应该跑一次 Python vs C++ 的 A/B 全量评测来确认“几乎完全一致”。

更稳妥的验证顺序：

1. `distill199 + Python decoder`
2. `distill199 + C++ decoder`
3. 对比 `score.txt`
4. 对比 `results.jsonl`
5. 对比最终 `summary.json`

如果这些结果一致或只存在浮点末位差异，就说明 199K 也已经完成了后处理替换。

补充说明：当前仓库里真正接上 C++ decoder 的是 `infer_wav_stream.py` / `evaluate_infer_wav.py --streaming --use_cpp_decoder` 这条流式链路；非流式离线分支目前仍然没有单独切到 C++。

---

## 11. 当前最实用的判断标准

如果你在做迁移验收，不要只看“能不能触发”，而要看下面四层是否一致：

1. 单帧 / 单 wav smoke test 的 `candidate_score`
2. Python/C++ 在逐帧 hypothesis 上是否一致
3. 全量 `score.txt` / `results.jsonl` 是否一致
4. 阈值挑选后的 `summary.json` 是否一致

只有这四层都对齐，才能认为“后处理 C++ 化没有掉点”。
