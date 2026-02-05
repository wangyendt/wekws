# Hi XiaoWen 数据流与评测说明

本文整理 `examples/hi_xiaowen/s0` 训练/评测/单条推理的数据流、关键形状、词表/字典差异与关键词匹配逻辑。面向当前配置 `exp/fsmn_ctc_baseline_4gpus/config.yaml`（`input_dim=400`，`output_dim=2599`）。

## 1) 为什么单条推理输出在 `train_79` 目录？

`local/infer_wav.sh` 会从 `data/metadata.db` 查询这条 wav 的 `split`（train/dev/test），如果你没有显式指定 `--dataset`，它就用数据库里的 `split` 当输出目录名：

```232:245:examples/hi_xiaowen/s0/local/infer_wav.sh
split_from_db=$(python - "${meta_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("split",""))
PY
)

if [ -z "${dataset}" ]; then
  dataset="${split_from_db:-single}"
fi

result_dir="${checkpoint_dir}/${dataset}_${checkpoint_basename}"
```

因此你的 wav 在 DB 里是 `train`，输出目录就变成 `exp/.../train_79`。

## 2) WAV → 特征 → 模型输入：每一步的 shape 与含义

以下描述与 `exp/fsmn_ctc_baseline_4gpus/config.yaml` 一致。

### 2.1 读取与重采样

- **原始 wav**：`torchaudio.load()` 输出 `waveform`，shape 为 `(C, L)`，`C=1`，`L` 是采样点数。
- **重采样**：如配置为 16 kHz，则 `L ≈ duration * 16000`。

### 2.2 Fbank 特征

配置：
```1:36:examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline_4gpus/config.yaml
dataset_conf:
  fbank_conf:
    num_mel_bins: 80
    frame_shift: 10
    frame_length: 25
```

**输出特征**：

- shape：`(T, 80)`
  - `T` ≈ `floor((L - frame_length) / frame_shift) + 1`
  - `frame_shift=10ms`，因此 `T` 大约是每秒 100 帧。

### 2.3 Context Expansion（左/右拼接）

配置：
```1:10:examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline_4gpus/config.yaml
context_expansion_conf:
  left: 2
  right: 2
```

逻辑（简化）：

- 将每一帧扩展为 `[t-2, t-1, t, t+1, t+2]` 拼接。
- **维度从 80 → 80 * (2+1+2) = 400**
- 右侧会“截掉” `right` 帧，所以时间步会减少 `right`：

```24:53:wekws/dataset/init_dataset.py
feats_ctx = feats_ctx[:, :feats_ctx.shape[1] - right]
sample['feats_lengths'] = sample['feats_lengths'] - right
```

**输出 shape**：

- `(T - right, 400)`  (这里 `right=2`)

### 2.4 Frame Skip

配置：
```1:24:examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline_4gpus/config.yaml
frame_skip: 3
```

逻辑：

- 每 3 帧取 1 帧，时间长度变为 `ceil((T - right) / 3)`：

```55:68:wekws/dataset/init_dataset.py
feats_skip = sample['feats'][:, ::skip_rate, :]
sample['feats_lengths'] = ceil(feats_lengths / skip_rate)
```

**输出 shape**：

- `(T', 400)`，其中 `T' = ceil((T - right) / 3)`

### 2.5 Batch + Padding

数据进入 `DataLoader` 后，会做 batch 和 padding：

- **输入 batch**：`(B, T_max, 400)`
- **有效长度**：`feats_lengths` shape `(B,)`

逻辑见：
```314:371:wekws/dataset/processor.py
def padding(data):
  padded_feats = pad_sequence(sorted_feats, batch_first=True, padding_value=0)
```

## 3) 模型输入/输出 shape

模型配置：
```35:58:examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline_4gpus/config.yaml
model:
  input_dim: 400
  output_dim: 2599
```

KWSModel 前向（FSMN）：

- **输入**：`(B, T', 400)`
- **输出 logits**：`(B, T', 2599)`

代码位置：
```65:90:wekws/model/kws_model.py
def forward(...):
  x = self.preprocessing(x)
  x, out_cache = self.backbone(x, in_cache)
  x = self.classifier(x)
  x = self.activation(x)
  return x, out_cache
```

> 结论：**模型就是输入 `N×400`，输出 `N×2599`，其中 N = `T'`（由 wav 时长决定，不能随意指定）**。

## 4) CTC 解码与关键词检测

### 4.1 模型输出 → 概率

`score_ctc.py` 内部：
```307:317:wekws/bin/score_ctc.py
logits_raw, _ = model(feats)  # (B, T, V)
probs = logits_raw.softmax(2)
probs_i = probs_cpu[i][:utt_len]  # (T, V)
```

### 4.2 CTC Prefix Beam Search

解码函数：
```206:313:wekws/model/loss.py
def ctc_prefix_beam_search(
    logits: torch.Tensor,  # (T, V)
    logits_lengths: ...
):
```

输出 `hyps`：

- `prefix_ids`: token id 序列，长度 **m**（不固定）
- `nodes`: 每个 token 对应的帧位置和概率

### 4.3 关键词匹配逻辑

`score_ctc.py` 中做**关键词子序列匹配**：

- 先把关键词转成 token id 序列
- 对 `prefix_ids` 做子序列查找
- 命中则记录起止帧与得分

```316:350:wekws/bin/score_ctc.py
hyps = ctc_prefix_beam_search(probs_i, utt_len, keywords_idxset)
for one_hyp in hyps:
  prefix_ids = one_hyp[0]
  for word in keywords_token.keys():
    lab = keywords_token[word]['token_id']
    offset = is_sublist(prefix_ids, lab)
    if offset != -1:
      hit_keyword = word
      ...
```

**最终结果不是完整转写文本**，而是：

```
<key> detected <keyword> <score>
```

## 5) Greedy/Beam 解码输出的 shape 与含义

如果开启 `--decode_greedy` 或 `--decode_beam`，`score_ctc.py` 会生成解码文本：

```168:184:wekws/bin/score_ctc.py
def _ctc_greedy_decode(token_ids, blank_id=0) -> List[int]:
  # 逐帧 argmax -> 去重 -> 去 blank

def _ids_to_text(ids, id2tok):
  toks = [id2tok.get(i, f'<unk:{i}>') for i in ids]
  return (" ".join(toks)).strip(), toks
```

- **Greedy decode**：逐帧 argmax → 去重/去 blank → 得到长度 `m` 的 token id 序列。
- **Beam decode**：CTC 前缀束搜索得到最优 token 序列，长度同样是 `m`（不固定）。

**注意**：`m` 取决于语音内容与 CTC 合并规则，不是固定值。  
**解码结果是 token 序列**，再通过 `dict.txt` 映射为文本。

如果某个 token id 在 `dict.txt` 中不存在，会显示为 `<unk:id>`。

## 6) dict.txt 为何是 2787 行，而不是 2599？

你当前 `dict.txt` 来自 `mobvoi_kws_transcription/tokens.txt`：

- `tokens.txt` 行数是 **2787**
- `dict.txt` 行数也是 **2787**

这是在训练脚本里从 ASR 转写资源生成的：

```148:152:examples/hi_xiaowen/s0/run_fsmn_ctc.sh
awk '{print $1, $2-1}' mobvoi_kws_transcription/tokens.txt > dict/dict.txt
sed -i 's/& 1/<filler> 1/' dict/dict.txt
```

而模型的 `output_dim=2599`（来自 `config.yaml`）：

```35:58:examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline_4gpus/config.yaml
  input_dim: 400
  output_dim: 2599
```

这说明：**词表（dict）比模型输出维度更大**。  
模型实际只能输出 `0..2598` 的 token id，超出范围的 token 永远不会被预测到。

### 6.1 “ASR 输出了字典外的符号怎么办？”

不会发生“输出字典外 token”的情况，因为模型输出维度就是 vocab 大小。  
如果解码时遇到字典中缺失的 id，会显示为 `<unk:id>`（见 `_ids_to_text` 实现）。

### 6.2 训练/评测时遇到“字典外文本”怎么办？

`score_ctc.py` 的 tokenizer 使用 `dict.txt` 并指定 unk 为 `<filler>`：

```230:233:wekws/bin/score_ctc.py
tokenizer = CharTokenizer(
  f'{args.dict}/dict.txt',
  f'{args.dict}/words.txt',
  unk='<filler>',
  split_with_space=True)
```

也就是说：**数据中的未知字符会被映射为 `<filler>`**，不会导致崩溃，但会影响准确率。

## 7) 关键词检测最终输出是什么？

在 `score_ctc.py` 中，关键词检测不是“完整识别文本”，而是：

1. CTC beam search 生成候选 token 序列 `prefix_ids`
2. 用关键词 token 序列做子串匹配
3. 命中则写入 `score.txt`：

```
<utt_id> detected 嗨小问 0.532
```

否则输出：

```
<utt_id> rejected
```

**所以最终结果不是完整转写文本，而是：**
- 是否检测到关键词
- 关键词类别
- 置信度与起止帧

如果你打开 `--decode_greedy/beam`，会额外输出文本序列，但这只是辅助信息。

## 8) 常见误解澄清

- **“模型输入 N×400，N 可以随意指定？”**  
  不行。`N` 是由音频时长和 `frame_shift + context_expansion + frame_skip` 共同决定的。

- **“模型输出 N×2599，2599 是字典行数？”**  
  不是。`2599` 是模型输出维度。`dict.txt` 可以更大，但超出维度的 token 不会被预测。

- **“CTC decode 输出 m×1？”**  
  可以理解为 **长度为 m 的 token 序列**。m 不固定。
