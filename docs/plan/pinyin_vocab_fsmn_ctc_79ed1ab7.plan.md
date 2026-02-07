---
name: pinyin_vocab_fsmn_ctc
overview: 在保持 FSMN+CTC 训练/评估脚本可用的前提下，把现有 2599 字表输出改成更小的拼音音节词表（非中文转录统一映射为 <filler>），并支持从阿里 base.pt 迁移加载（跳过不匹配输出层），为后续把模型压到 100–200K 做准备。
todos:
  - id: token_mode_make_list
    content: 给 tools/make_list.py 增加可选 text_mode=pinyin_syllable，并实现“中文转拼音、非中文-><filler>”输出
    status: completed
  - id: dict_build_and_vocab_size
    content: 在 run_fsmn_ctc.sh 增加拼音模式开关：生成 dict/dict.txt+words.txt，并自动计算 N 传给 --num_keywords
    status: completed
  - id: checkpoint_partial_load
    content: 增强 wekws/utils/checkpoint.py：支持 strict=False 并自动跳过 shape 不匹配层；train.py 增加开关来启用
    status: completed
  - id: keyword_convert_eval
    content: 修改 score_ctc.py 与 compute_det_ctc.py：根据字典自动决定关键词按汉字或按拼音 token 化
    status: completed
  - id: shrink_followup
    content: 提供后续把模型压到100–200K的结构瘦身参数建议与实验路径（基于fmsn_ctc.yaml）
    status: completed
isProject: false
---

## 目标与约束

- **目标**：把 `output_dim` 从 2599 降到更小的 N（拼音音节词表），先显著缩小输出层；同时为后续进一步瘦身到 **100–200K** 做结构/量化铺垫。
- **约束**：继续使用 **FSMN+CTC**；尽量保持 [`examples/hi_xiaowen/s0/run_fsmn_ctc.sh`](/home/wayne/work/code/project/ffalcon/wekws/examples/hi_xiaowen/s0/run_fsmn_ctc.sh) 与 [`examples/hi_xiaowen/s0/eval_checkpoint.sh`](/home/wayne/work/code/project/ffalcon/wekws/examples/hi_xiaowen/s0/eval_checkpoint.sh) 仍可直接跑通（只需增加可选开关/自动检测）。

## 现状关键耦合点（为何需要改）

- 训练时 `output_dim` 来自 `--num_keywords`（实际是“词表大小”）：`wekws/bin/train.py` 将其写入 `configs['model']['output_dim']` 并构建模型。
- FSMN 的输出层在 backbone 内部：`wekws/model/kws_model.py` 里 `FSMN(..., output_dim)`，最终层 `out_linear2` 的参数量约为 `140×output_dim + output_dim`。
- 目前 `load_checkpoint` 是严格加载：`wekws/utils/checkpoint.py` 里 `model.load_state_dict(checkpoint)`，当 `output_dim` 变小会因 shape 不匹配直接报错。
- 文本到 token ids：`tools/make_list.py` 生成 `data/*/data.list` 的 `txt` 字段；`CharTokenizer(split_with_space=True)` 期望输入是“空格分隔 token 序列”。
- 评估/解码：`wekws/bin/score_ctc.py` 和 `wekws/bin/compute_det_ctc.py` 用 `CharTokenizer` 把 **keywords** 转 token ids（当前做法是按汉字拆分）。

## 方案设计（你已确认）

- **token 方案**：拼音音节 token（`pypinyin.Style.NORMAL`，无声调）。
- **非中文转录**：统一映射为单个 `<filler>` token（避免英文转录导致词表膨胀）。

## 需要改/新增的内容（按影响面从小到大）

### A. 数据/字典生成（让训练 label 变成拼音 token 序列）

- **新增/改造** `tools/make_list.py`（保持默认行为不变）：
- 增加可选参数（例如 `--text_mode char|pinyin_syllable`，默认 `char`）。
- `pinyin_syllable` 模式下：
- 中文字符：用 `pypinyin.lazy_pinyin` 转为拼音音节 tokens。
- 非中文（英文单词、数字、符号等）：整段映射为 `<filler>`（避免序列过长）。
- 输出仍是空格分隔 token 序列，供 `CharTokenizer(split_with_space=True)` 直接消费。
- **在** [`examples/hi_xiaowen/s0/run_fsmn_ctc.sh`](/home/wayne/work/code/project/ffalcon/wekws/examples/hi_xiaowen/s0/run_fsmn_ctc.sh) **Stage 1** 里，调用 `tools/make_list.py` 时加上该模式开关（仅在你启用拼音模式时）。
- **字典文件**：
- 生成 `dict/dict.txt`：至少包含 `<blk> 0`、`<filler> 1`、`<silence> 2`（或等价特殊符号），以及所有拼音音节 tokens。
- 生成 `dict/words.txt`：供 `CharTokenizer` 使用（保持与 `dict.txt` 第一列一致即可）。
- 让 `run_fsmn_ctc.sh` 自动从 `dict/dict.txt` 计算 `N` 并设置 `--num_keywords=N`（避免手工改 2599）。

### B. 训练加载阿里 `base.pt`（输出层不匹配也能迁移）

- **改造** [`wekws/utils/checkpoint.py`](/home/wayne/work/code/project/ffalcon/wekws/wekws/utils/checkpoint.py)：
- 给 `load_checkpoint` 增加 `strict` 参数（默认 `True` 保持兼容）。
- 当 `strict=False`：自动过滤掉 **shape 不匹配** 的 key（主要是 `backbone.out_linear2.linear.weight/bias`），只加载可匹配的层。
- **改造** [`wekws/bin/train.py`](/home/wayne/work/code/project/ffalcon/wekws/wekws/bin/train.py)：
- 增加训练参数开关（例如 `--checkpoint_strict true|false`，默认 `true`）。
- 当你在拼音小词表上用 `base.pt` 初始化时，传 `--checkpoint_strict false`，实现“迁移 backbone + 随机初始化新输出层”。

### C. 评估脚本兼容（keywords 仍传中文，但内部按字典自动转拼音）

- **改造** [`wekws/bin/score_ctc.py`](/home/wayne/work/code/project/ffalcon/wekws/wekws/bin/score_ctc.py) 与 [`wekws/bin/compute_det_ctc.py`](/home/wayne/work/code/project/ffalcon/wekws/wekws/bin/compute_det_ctc.py)：
- 增加一个“关键词转 token”的统一函数：
- 若检测到 `dict/dict.txt` 里主要是拼音 token（基本不含中文），则把 `--keywords` 的中文先转为拼音音节 tokens，再 `tokenizer.tokenize(' '.join(tokens))`。
- 否则保持现有汉字逐字拆分逻辑。
- 这样 `eval_checkpoint.sh` 的 `--keywords "嗨小问,你好问问"` **无需改** 或只需极小改动。

### D. 参数量与 100–200K 目标的现实差距（给下一步提供路线）

- 仅改 `output_dim` 能减少的参数主要来自 FSMN 最后一层：
- 现状 `140×2599+2599≈366k` 参数。
- 若 N≈200：输出层约 `28.2k`，仅这一项减少约 `338k`。
- 所以从 700k 级别下降后，大概率仍在 **300–400k** 量级。
- 要到 **100–200K**，建议后续做两类动作（可以分阶段）：
- **结构瘦身**：在 [`examples/hi_xiaowen/s0/conf/fsmn_ctc.yaml`](/home/wayne/work/code/project/ffalcon/wekws/examples/hi_xiaowen/s0/conf/fsmn_ctc.yaml) 里逐步下调 `linear_dim/proj_dim/input_affine_dim/output_affine_dim/num_layers`，并配合蒸馏或更长训练。
- **部署侧压缩**：ONNX + INT8（已有 `wekws/bin/static_quantize.py`），可进一步降体积/加速。

## 验证与回归（保持原脚本可用）

- 兼容性目标：
- `bash run_fsmn_ctc.sh ...` 在默认 `char` 模式下行为不变。
- 启用拼音模式时：仍能完成 Stage 1 训练、Stage 3 `eval_checkpoint.sh` 评估与 `compute_det_ctc.py` 出图。
- 最小验证点：
- 训练能启动且 `count_parameters(model)` 明显下降。
- `score_ctc.py` 能正确把中文关键词映射为拼音 token 序列并命中（至少在正样本上有 detected）。