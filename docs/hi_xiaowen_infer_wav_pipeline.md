# `infer_wav.py` 离线唤醒推理链路说明

本文面向 `examples/hi_xiaowen/s0/infer_wav.py`，按真实代码调用顺序说明一条 `wav` 是如何一步一步变成最终 JSON 结果的。

重点覆盖：

- 函数调用链
- 每一步输入/输出的数据结构
- 每一步张量的 shape
- 关键公式：Kaldi Fbank、CMVN、Context Expansion、FSMN、Softmax、CTC Prefix Beam Search、最终阈值判定

本文默认以 `s3` 模型为例，即：

- checkpoint: `examples/hi_xiaowen/s0/exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/399.pt`
- config: `examples/hi_xiaowen/s0/exp/fsmn_ctc_distill_s3_a48_p24_l3_merged/config.yaml`
- dict: `examples/hi_xiaowen/s0/dict_top20`

## 1. 总调用链

`main()` 的主流程可以概括为：

```text
parse_args
  -> to_abs_path
  -> parse_keywords_arg
  -> resolve_model_paths
  -> load_config
  -> load_model
  -> build_input_features
       -> load_wav_and_resample
       -> extract_fbank_features
       -> apply_context_expansion
       -> apply_frame_skip
  -> run_model_forward
       -> KWSModel.forward
            -> GlobalCMVN
            -> preprocessing
            -> FSMN.forward
            -> classifier
            -> activation
       -> softmax
  -> decode_keyword_hit
       -> build_keyword_token_info
       -> decode_keyword_hit_with_token_info
            -> ctc_prefix_beam_search
            -> is_sublist
  -> load_threshold_map
       -> parse_threshold_map
       -> parse_stats_file
       -> pick_best_row
  -> get_time_resolution_sec
  -> format_result
  -> print(json)
```

从数据形态上看，主链路是：

```text
wav 文件
  -> waveform: (1, L)
  -> fbank: (T_raw, 80)
  -> context 扩展后: (T_ctx, 400)
  -> frame_skip 后: (T, 400)
  -> batch 化后: (1, T, 400)
  -> 模型输出 logits: (1, T, 20)
  -> softmax 概率: (T, 20)
  -> CTC 候选 token 序列: 长度 m 的 token id 列表
  -> 关键词命中结果: dict
  -> 阈值判定后的最终 JSON: dict
```

其中：

- `L` 是音频采样点数
- `T_raw` 是 Fbank 帧数
- `T_ctx` 是 context expansion 后的帧数
- `T` 是 frame skip 后进入模型的帧数
- `m` 是 beam search 得到的 token 序列长度，不固定

## 2. 默认 `s3` 配置与实际维度

默认 `s3` 的 `config.yaml` 里，和推理 shape 直接相关的配置是：

- 采样率：`16000`
- `num_mel_bins=80`
- `frame_length=25 ms`
- `frame_shift=10 ms`
- `context_expansion.left=2`
- `context_expansion.right=2`
- `frame_skip=3`
- `input_dim=400`
- `hidden_dim=48`
- `output_dim=20`
- FSMN 主干：
  - `input_affine_dim=48`
  - `linear_dim=250`
  - `proj_dim=24`
  - `num_layers=3`
  - `left_order=10`
  - `right_order=2`
  - `merge_head=true`

因此默认 `s3` 的模型前向维度链路是：

```text
(1, T, 400)
  -> CMVN                    -> (1, T, 400)
  -> preprocessing = none    -> (1, T, 400)
  -> in_linear1 400->48      -> (1, T, 48)
  -> in_linear2 48->250      -> (1, T, 250)
  -> ReLU                    -> (1, T, 250)
  -> 3 x FSMN block          -> (1, T, 250)
  -> merged head 250->20     -> (1, T, 20)
  -> classifier identity     -> (1, T, 20)
  -> activation identity     -> (1, T, 20)
  -> softmax(dim=2)          -> (1, T, 20)
  -> cpu()[0]                -> (T, 20)
```

这里 `output_dim=20` 对应 `dict_top20/dict.txt` 中可预测的 20 个 token id。默认关键词的 token 序列是：

- `嗨小问 -> [16, 11, 3]`
- `你好问问 -> [6, 9, 3, 3]`

## 3. 入口参数层

### 3.1 `parse_args()`

作用：解析命令行参数。

输入：

- 无显式张量输入

输出：

- `argparse.Namespace`

关键字段：

- `wav: str`
- `model: str`
- `checkpoint: str`
- `model_dir: str`
- `config: str`
- `dict_dir: str`
- `stats_dir: str`
- `keywords: str`
- `threshold_map: str`
- `target_fa_per_hour: float`
- `pick_mode: str`
- `gpu: int`
- `disable_threshold: bool`

这一层还没有 shape，主要是把文本参数规范化，交给后续函数使用。

### 3.2 `parse_keywords_arg(raw_keywords)`

作用：把逗号分隔的关键词字符串转成列表。

输入：

- `raw_keywords: str`

输出：

- `List[str]`

例如：

```text
"嗨小问,你好问问" -> ["嗨小问", "你好问问"]
```

## 4. 路径与配置解析

### 4.1 `resolve_model_paths(args)`

作用：把别名 `s3`、手工传入的 `--checkpoint`、`--model_dir` 等统一解析成实际文件路径。

输入：

- `args: Namespace`

输出：

- `model_info: Dict[str, Optional[Path]]`

字段包括：

- `alias`
- `checkpoint`
- `config`
- `dict_dir`
- `stats_dir`

这一层仍然不是张量处理，但它决定了后面：

- 模型权重从哪里加载
- 特征参数从哪里读取
- 词表从哪里读取
- 阈值从哪里读取

### 4.2 `load_config(config_path)`

作用：读取 YAML 配置，并在 `resolve_cmvn_path()` 中把相对路径的 `cmvn_file` 变成绝对路径。

输入：

- `config_path: Path`

输出：

- `configs: Dict`

对默认 `s3` 而言，最关键的是：

- `dataset_conf`
- `model.cmvn`
- `model.backbone`
- `model.output_dim`

## 5. 音频到模型输入：`build_input_features()`

`build_input_features(wav_path, configs)` 是离线推理里最重要的前处理总入口。

它内部按顺序调用：

1. `load_wav_and_resample`
2. `extract_fbank_features`
3. `apply_context_expansion`
4. `apply_frame_skip`
5. `unsqueeze(0)` 增加 batch 维

### 5.1 `load_wav_and_resample(wav_path, target_sr)`

作用：读取音频、转单声道、必要时重采样。

输入：

- `wav_path: Path`
- `target_sr: int`

内部调用：

- `torchaudio.load(str(wav_path))`
- `torchaudio.functional.resample(...)`

输入数据形态：

- 音频文件

输出：

- `waveform: torch.Tensor`

shape：

- 原始读取后：`(C, L0)`
  - `C` 是通道数
  - `L0` 是原始采样点数
- 如果多通道，则做均值：
  - `waveform.mean(dim=0, keepdim=True)`
  - 输出变成 `(1, L0)`
- 如果重采样到 16 kHz：
  - 输出变成 `(1, L)`

其中：

- `L ~= duration_sec * 16000`

### 5.2 `extract_fbank_features(waveform, dataset_conf)`

作用：把时域波形转成 Kaldi 风格 Fbank 特征。

输入：

- `waveform: torch.Tensor`, shape `(1, L)`

输出：

- `feats: torch.Tensor`, shape `(T_raw, 80)`

内部关键代码逻辑：

1. `waveform = waveform * (1 << 15)`
2. `kaldi.fbank(...)`

第一步的意义：

- `torchaudio.load()` 的浮点波形通常在 `[-1, 1]`
- 乘 `2^15` 是把它映射到接近 16-bit PCM 的数值尺度，符合 Kaldi 的习惯

默认 `s3` 下的 Fbank 参数：

- `sample_frequency = 16000`
- `frame_length = 25 ms`
- `frame_shift = 10 ms`
- `num_mel_bins = 80`

对应的样本点数是：

- 帧长：`N_win = 0.025 * 16000 = 400`
- 帧移：`N_hop = 0.010 * 16000 = 160`

在 Kaldi 默认 `snip_edges=True` 的前提下，原始帧数近似是：

$$
T_{raw} = \left\lfloor \frac{L - N_{win}}{N_{hop}} \right\rfloor + 1
$$

也就是：

$$
T_{raw} = \left\lfloor \frac{L - 400}{160} \right\rfloor + 1
$$

例如 1 秒音频：

- `L = 16000`
- `T_raw = floor((16000 - 400) / 160) + 1 = 98`
- 所以 Fbank shape 是 `(98, 80)`

#### Fbank 核心公式

给定时域信号 `x[n]`，第 `t` 帧加窗后做短时傅里叶变换：

$$
X_t[k] = \sum_{n=0}^{N_{win}-1} x[n + tN_{hop}] w[n] e^{-j 2 \pi kn / N_{win}}
$$

得到功率谱：

$$
P_t[k] = |X_t[k]|^2
$$

再通过 Mel 滤波器组做加权求和，第 `m` 个 Mel bin 的能量为：

$$
E_t[m] = \sum_k H_m[k] P_t[k]
$$

最终取对数得到 log Mel filterbank：

$$
F_t[m] = \log(E_t[m] + \epsilon)
$$

于是整段特征可写为：$F \in \mathbb{R}^{T_{raw} \times 80}$。

#### 一个实现细节

虽然 `config.yaml` 里 `fbank_conf.dither=1.0`，但 `infer_wav.py` 的 `extract_fbank_features()` 实际硬编码传的是：

- `dither=0.0`
- `energy_floor=0.0`

也就是说，单条离线推理脚本并没有直接使用配置里的 `dither` 值。

### 5.3 `apply_context_expansion(feats, dataset_conf)`

作用：把当前帧和左右邻居帧拼接成更高维的上下文特征。

输入：

- `feats: torch.Tensor`, shape `(T_raw, 80)`

默认 `s3` 配置：

- `left = 2`
- `right = 2`

输出：

- `feats_ctx: torch.Tensor`, shape `(T_ctx, 400)`

这里单帧的上下文拼接公式是：

$$
c_t = [f_{t-2}; f_{t-1}; f_t; f_{t+1}; f_{t+2}] \in \mathbb{R}^{5 \times 80}
$$

拼接后维度为：$400 = 80 \times (2 + 1 + 2)$。

因此：$c_t \in \mathbb{R}^{400}$。

整段输出是：$C \in \mathbb{R}^{T_{ctx} \times 400}$。

代码里最后会执行：

```text
return feats_ctx[: num_frames - right]
```

因此时间维会减掉 `right` 帧：$T_{ctx} = T_{raw} - right$。

在默认 `s3` 下：$T_{ctx} = T_{raw} - 2$。

例如 1 秒音频：

- `T_raw = 98`
- `T_ctx = 96`
- shape 从 `(98, 80)` 变成 `(96, 400)`

边界处理方面：

- 左边界使用复制后的上下文填充
- 右边界通过最后的裁剪去掉 `right` 帧，避免无效 roll 结果泄漏到输出

### 5.4 `apply_frame_skip(feats, dataset_conf)`

作用：每隔 `frame_skip` 帧取一帧，降低时间分辨率和计算量。

输入：

- `feats: torch.Tensor`, shape `(T_ctx, 400)`

默认 `s3` 配置：

- `frame_skip = 3`

输出：

- `feats_skip: torch.Tensor`, shape `(T, 400)`

代码是：

```python
feats[::frame_skip, :]
```

因此时间维长度是：$T = \left\lceil \frac{T_{ctx}}{3} \right\rceil$。

例如 1 秒音频：

- `T_ctx = 96`
- `T = ceil(96 / 3) = 32`
- shape 变成 `(32, 400)`

### 5.5 `build_input_features()` 最终输出

`build_input_features()` 末尾执行：

```python
return feats.unsqueeze(0)
```

所以最终模型输入是：$X \in \mathbb{R}^{1 \times T \times 400}$。

也就是：

- batch size 固定为 `1`
- 时间帧数是 `T`
- 特征维度是 `400`

对于 1 秒音频，默认 `s3` 下大约是：

```text
(1, 32, 400)
```

## 6. 模型前向：`run_model_forward()`

### 6.1 总体输入输出

`run_model_forward(model, feats, device, is_jit)` 的输入输出为：

输入：

- `feats: torch.Tensor`, shape `(1, T, 400)`

输出：

- `probs: torch.Tensor`, shape `(T, 20)`

注意这里函数名叫 `run_model_forward`，但它返回的并不是原始 logits，而是：

```python
logits.softmax(2).cpu()[0]
```

也就是说：

1. 先得到模型输出 `logits`: `(1, T, 20)`
2. 再对词表维 `dim=2` 做 softmax
3. 再去掉 batch 维，得到 `(T, 20)`

关于 cache 还有两个实现细节：

- 如果加载的是 `.pt` 模型，离线脚本直接调用 `model(feats)`，不显式传入 cache
- 如果加载的是 `.zip` JIT 模型，离线脚本会额外传入 `empty_cache = torch.zeros(0, 0, 0)` 作为占位缓存

### 6.2 `KWSModel.forward()`

`KWSModel.forward(x, in_cache)` 的主链路是：

```text
x
  -> global_cmvn
  -> preprocessing
  -> backbone
  -> classifier
  -> activation
```

#### 6.2.1 Global CMVN

输入：

- `x: (1, T, 400)`

输出：

- `x_cmvn: (1, T, 400)`

公式：$\hat{x}_{t,d} = (x_{t,d} - \mu_d) \cdot \sigma_d^{-1}$。

其中：

- `\mu_d` 是第 `d` 维均值
- `\sigma_d^{-1}` 是第 `d` 维标准差倒数

也就是逐维做：

1. 减均值
2. 乘逆标准差

shape 不变。

#### 6.2.2 preprocessing

默认 `s3` 配置里：

- `preprocessing.type = none`

因此：

- 输入 `(1, T, 400)`
- 输出仍然 `(1, T, 400)`

### 6.3 `FSMN.forward()`

默认 `s3` 主干是 `FSMN`。

输入：

- `input: (1, T, 400)`

输出：

- `x7: (1, T, 20)`
- `out_cache`

这里离线推理只关心 `x7`，`out_cache` 主要给流式场景使用。

对默认 `s3` 而言：

- `proj_dim = 24`
- `num_layers = 3`
- `padding = (left_order - 1) * left_stride + right_order * right_stride = 11`

因此如果 cache 被真正使用，拼接后的 `out_cache` shape 是：

```text
(B, proj_dim, padding, num_layers) = (1, 24, 11, 3)
```

#### 6.3.1 输入仿射层

第一层：

- `in_linear1: 400 -> 48`
- shape: `(1, T, 400) -> (1, T, 48)`

第二层：

- `in_linear2: 48 -> 250`
- shape: `(1, T, 48) -> (1, T, 250)`

然后：

- `ReLU`
- shape 保持 `(1, T, 250)`

#### 6.3.2 单个 FSMN block 的 shape

每个 block 是：

```text
LinearTransform(250 -> 24)
  -> FSMNBlock(24 -> 24, left_order=10, right_order=2)
  -> AffineTransform(24 -> 250)
  -> ReLU
```

所以单层 block 的维度变化是：

```text
(1, T, 250)
  -> (1, T, 24)
  -> (1, T, 24)
  -> (1, T, 250)
  -> (1, T, 250)
```

默认 `s3` 一共有 `3` 层 block，所以整个主干中间部分是：

```text
(1, T, 250)
  -> block1 -> (1, T, 250)
  -> block2 -> (1, T, 250)
  -> block3 -> (1, T, 250)
```

#### 6.3.3 FSMN 核心公式

FSMNBlock 先把 `250` 维投影到 `24` 维，再在时间维做深度可分离的记忆卷积。可把其核心记忆项理解为：

$$
\tilde{h}_t
= h_t
+ \sum_{i=0}^{L-1} a_i \odot h_{t-i}
+ \sum_{j=1}^{R} c_j \odot h_{t+j}
$$

其中：

- `h_t \in \mathbb{R}^{24}` 是投影后的当前帧表示
- `L = left_order = 10`
- `R = right_order = 2`
- `a_i, c_j \in \mathbb{R}^{24}` 是逐通道记忆滤波器参数
- `\odot` 表示逐元素乘法

在代码实现中，这个记忆项通过 `Conv2d(groups=self.dim)` 完成，本质上是对每个通道单独沿时间维做卷积。

离线推理时，FSMN 不改变时间维长度，所以：

- 输入 `(1, T, 24)`
- 输出仍是 `(1, T, 24)`

#### 6.3.4 输出头

默认 `s3` 使用：

- `merge_head = true`

因此没有两层输出头，而是直接：

- `out_linear2: 250 -> 20`

输出 shape：

```text
(1, T, 250) -> (1, T, 20)
```

随后：

- `classifier = identity`
- `activation = identity`

所以模型主体最终输出 logits：$Z \in \mathbb{R}^{1 \times T \times 20}$。

### 6.4 Softmax

`run_model_forward()` 里执行：

```python
probs = logits.softmax(2)
```

因此每一帧都会在词表维做归一化：$p_t(v) = \frac{e^{z_t(v)}}{\sum_{u=0}^{19} e^{z_t(u)}}$。

输出：$P \in \mathbb{R}^{T \times 20}$。

其中第 `t` 帧满足：$\sum_{v=0}^{19} p_t(v) = 1$。

## 7. 关键词解码：`decode_keyword_hit()`

这一部分负责把 `(T, 20)` 的逐帧类别概率，变成“有没有命中关键词、命中了哪个关键词、分数是多少、起止帧在哪里”。

### 7.1 `build_keyword_token_info(keywords, dict_dir)`

作用：把文本关键词转成 token id 序列。

输入：

- `keywords: List[str]`
- `dict_dir: Path`

输出：

- `keywords_token: Dict[str, Dict[str, Tuple[int, ...]]]`
- `keywords_idxset: set`

默认例子：

```text
"嗨小问"   -> (16, 11, 3)
"你好问问" -> (6, 9, 3, 3)
```

`keywords_idxset` 初始为 `{0}`，再加入所有关键词 token id，因此默认会变成：

```text
{0, 3, 6, 9, 11, 16}
```

这里的 `0` 是 blank。

这一步很重要，因为后面的 beam search 会只关注：

- blank
- 关键词相关 token

从而降低搜索空间。

### 7.2 `decode_keyword_hit_with_token_info(probs, keywords, keywords_token, keywords_idxset)`

输入：

- `probs: torch.Tensor`, shape `(T, 20)`
- `keywords: List[str]`
- `keywords_token`
- `keywords_idxset`

输出：

- `decode_result: Dict[str, object]`

字段为：

- `candidate_keyword`
- `candidate_score`
- `start_frame`
- `end_frame`

### 7.3 `ctc_prefix_beam_search()`

内部调用：

```python
hyps = ctc_prefix_beam_search(probs[:utt_len], utt_len, keywords_idxset)
```

其中：

- `utt_len = int(probs.size(0)) = T`

因此 beam search 的输入是：$P \in \mathbb{R}^{T \times 20}$。

输出 `hyps` 是一个列表，每个元素是：

```text
(prefix_ids, path_score, nodes)
```

其中：

- `prefix_ids`: tuple[int, ...]
  - 长度为 `m`
- `path_score`: float
- `nodes`: List[dict]
  - 长度也是 `m`
  - 每个元素含有：
    - `token`
    - `frame`
    - `prob`

也就是：

- 输入是逐帧概率矩阵 `(T, 20)`
- 输出是若干条候选 token 路径，每条路径长度 `m` 不固定

#### 7.3.1 CTC Prefix Beam Search 核心递推

代码维护每个前缀 `l` 的两类概率：

- `P_b^t(l)`: 时刻 `t` 结束于 blank 的前缀概率
- `P_{nb}^t(l)`: 时刻 `t` 结束于 non-blank 的前缀概率

对 blank 的更新是：

$$
P_b^t(l) \mathrel{+}= \left(P_b^{t-1}(l) + P_{nb}^{t-1}(l)\right) p_t(\text{blank})
$$

对重复 token 的更新是：

$$
P_{nb}^t(l) \mathrel{+}= P_{nb}^{t-1}(l) p_t(s)
$$

对从 blank 扩展出重复 token 的更新是：

$$
P_{nb}^t(l + s) \mathrel{+}= P_b^{t-1}(l) p_t(s)
$$

对新 token 扩展的更新是：

$$
P_{nb}^t(l + s) \mathrel{+}= \left(P_b^{t-1}(l) + P_{nb}^{t-1}(l)\right) p_t(s)
$$

最后每条前缀的总分是：

$$
P^t(l) = P_b^t(l) + P_{nb}^t(l)
$$

代码里每一帧还会做两级裁剪：

1. `score_beam_size=3` 的 token 级裁剪
2. `path_beam_size=20` 的路径级裁剪

而且由于传入了 `keywords_idxset`，只有 blank 和关键词相关 token 会保留。

### 7.4 `is_sublist(prefix_ids, label)`

作用：检查某个关键词 token 序列是否是 beam 候选路径中的连续子序列。

输入：

- `prefix_ids`: 长度 `m`
- `label`: 长度 `k`

输出：

- `offset: int`
  - 找到则返回起始位置
  - 否则返回 `-1`

例如：

```text
prefix_ids = [16, 11, 3]
label     = [16, 11, 3]
offset    = 0
```

### 7.5 候选关键词得分公式

一旦 `prefix_ids` 中匹配到关键词 `label`，代码会取匹配区间内每个 token 对应的 `prob` 相乘：$s = \prod_{i=\text{offset}}^{\text{offset}+k-1} p_i$。

然后返回：$\text{candidate\_score} = \sqrt{s}$。

注意这里是代码真实逻辑：

```python
hit_score = math.sqrt(score)
```

也就是说：

- 不管关键词长度是 3 个 token 还是 4 个 token
- 最终都统一取平方根

因此它不是严格意义上的 `k` 次几何平均，而是“匹配 token 概率乘积后再开平方”。

### 7.6 `decode_result` 的 shape 和含义

`decode_keyword_hit_with_token_info()` 返回的不是张量，而是一个字典：

```python
{
    "candidate_keyword": str | None,
    "candidate_score": float | None,
    "start_frame": int | None,
    "end_frame": int | None,
}
```

其中：

- `candidate_keyword` 是 beam 命中的第一个关键词
- `candidate_score` 是上面定义的分数
- `start_frame`, `end_frame` 来自 `nodes` 中匹配 token 的帧号

## 8. 阈值加载：`load_threshold_map()`

解码阶段只负责找候选关键词和分数，是否真正触发，还要看阈值。

### 8.1 输入输出

输入：

- `args`
- `model_info`
- `keywords: List[str]`

输出：

- `threshold_map: Dict[str, Optional[float]]`

例如默认 `s3` 下，最终通常会得到：

```text
{
  "嗨小问": 0.272,
  "你好问问": 0.016
}
```

### 8.2 阈值来源优先级

优先级从高到低是：

1. `--threshold_map` 手工覆盖
2. `stats_dir/stats.<keyword>.txt`
3. `FALLBACK_THRESHOLDS`

### 8.3 `parse_stats_file(stats_file)`

每个 stats 文件会被解析成：

- `List[StatRow]`

其中每个 `StatRow` 有：

- `threshold: float`
- `fa_per_hour: float`
- `frr: float`

这一步没有张量 shape，属于表格数据解析。

### 8.4 `pick_best_row(rows, target_fa, pick_mode, frr_eps)`

默认 `infer_wav.py` 用的是：

- `target_fa_per_hour = 1.0`
- `pick_mode = legacy`

`legacy` 模式下的选择规则是：

1. 先筛出 `fa_per_hour <= target_fa` 的候选
2. 在候选中按 `(frr, fa_per_hour, threshold)` 最小排序
3. 取最优行

也就是优先：

1. 低拒真率
2. 更低 FA/h
3. 更低阈值

默认 `s3` 的实际选择结果是：

- `嗨小问 -> threshold = 0.272`
- `你好问问 -> threshold = 0.016`

## 9. 时间分辨率：`get_time_resolution_sec()`

该函数根据配置计算每一帧对应多少秒：

$$
\text{time\_resolution\_sec}
= \frac{\text{frame\_shift\_ms} \times \text{frame\_skip}}{1000}
$$

默认 `s3` 下：

- `frame_shift_ms = 10`
- `frame_skip = 3`

所以（默认 `s3`）：

$$
\text{time\_resolution\_sec} = \frac{10 \times 3}{1000} = 0.03
$$

也就是说模型输出的每一帧大约对应原音频时间轴上的 `30 ms`。

## 10. 最终结果组装：`format_result()`

### 10.1 输入

输入参数包括：

- `wav_path`
- `model_info`
- `threshold_map`
- `decode_result`
- `time_resolution_sec`
- `disable_threshold`

其中关键的中间结果是：

- `candidate_keyword`
- `candidate_score`
- `start_frame`
- `end_frame`

### 10.2 时间换算

如果 beam search 找到了候选关键词，则：

$$
\text{start\_time\_sec} = \text{start\_frame} \times 0.03
$$

$$
\text{end\_time\_sec} = \text{end\_frame} \times 0.03
$$

### 10.3 最终触发判定

如果没有命中候选关键词：

- `triggered = False`

如果命中了候选关键词：

- 先取该关键词对应阈值 `threshold`
- 若 `disable_threshold=True`，则直接触发
- 否则按下面的规则判断：

$$
\text{triggered} =
\left(\text{candidate\_score} \ge \text{threshold}\right)
$$

触发后：

- `keyword = candidate_keyword`
- `wake_time_sec = end_time_sec`

否则：

- `keyword = None`
- `wake_time_sec = None`

### 10.4 最终输出结构

最终 `print(json.dumps(...))` 输出的是一个字典：

```python
{
    "wav": str,
    "model_alias": str,
    "checkpoint": str,
    "config": str,
    "dict_dir": str,
    "stats_dir": str | None,
    "triggered": bool,
    "keyword": str | None,
    "wake_time_sec": float | None,
    "score": float | None,
    "threshold": float | None,
    "start_frame": int | None,
    "end_frame": int | None,
    "start_time_sec": float | None,
    "end_time_sec": float | None,
    "candidate_keyword": str | None,
    "candidate_score": float | None,
    "threshold_map": Dict[str, Optional[float]],
    "time_resolution_sec": float,
}
```

这就是从 `wav` 到最终结果的最后一层输出。

## 11. 一条 1 秒音频的完整 shape 示例

假设输入音频：

- 单声道
- 采样率 16 kHz
- 时长 1.0 秒

则整条链路大致是：

```text
wav 文件
  -> waveform                : (1, 16000)
  -> fbank                   : (98, 80)
  -> context expansion       : (96, 400)
  -> frame skip              : (32, 400)
  -> unsqueeze batch         : (1, 32, 400)
  -> CMVN                    : (1, 32, 400)
  -> in_linear1              : (1, 32, 48)
  -> in_linear2              : (1, 32, 250)
  -> FSMN block x 3          : (1, 32, 250)
  -> output head             : (1, 32, 20)
  -> softmax                 : (1, 32, 20)
  -> squeeze batch           : (32, 20)
  -> CTC beam search         : 若干条长度 m 不定的 token 序列
  -> keyword match           : candidate_keyword / candidate_score
  -> threshold compare       : triggered / keyword / wake_time_sec
```

## 12. 几个容易忽略的实现细节

- `infer_wav.py` 在特征提取时把 `waveform` 乘了 `2^15`，这是为了匹配 Kaldi 风格输入尺度。
- `extract_fbank_features()` 实际写死了 `dither=0.0`，不是直接照搬 `config.yaml` 中的 `dither=1.0`。
- `build_keyword_token_info()` 会把关键词拆成逐字序列，因此它依赖 `dict.txt` 中字符级 token 定义。
- `ctc_prefix_beam_search()` 只保留 blank 和关键词相关 token，这不是“完整 ASR 解码”，而是面向关键词检测的定向搜索。
- `candidate_score` 不是简单的最大帧概率，也不是标准 `k` 次几何平均，而是“匹配 token 概率乘积后统一开平方”。
- 最终 `wake_time_sec` 使用的是 `end_frame * time_resolution_sec`，表示关键词末 token 的时间位置。
- 即使 `candidate_keyword` 非空，也不代表最终触发；还必须通过 `threshold` 比较。

## 13. 总结

`infer_wav.py` 的本质可以概括为：

1. 把 `wav` 变成 `Fbank + context + frame_skip` 的帧级特征；
2. 用 `FSMN-CTC` 输出逐帧 token 概率；
3. 用定向 `CTC prefix beam search` 搜索关键词相关 token 序列；
4. 通过子序列匹配得到候选关键词、分数和起止帧；
5. 再结合 `stats.*.txt` 里选出的阈值，得到最终的 `triggered / keyword / wake_time_sec`。

如果只看默认 `s3`，它的核心张量流就是：

```text
(1, L)
  -> (T_raw, 80)
  -> (T_ctx, 400)
  -> (1, T, 400)
  -> (1, T, 20)
  -> (T, 20)
  -> 关键词候选 dict
  -> 最终 JSON dict
```

这是理解 `infer_wav.py`、排查误触发/漏检、或者改写成流式版本时最核心的一条主线。
