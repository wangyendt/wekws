# Hi XiaoWen 脚本用法汇总

本文汇总 `examples/hi_xiaowen/s0` 相关脚本与 `tools/analyze_asr_vocab.py` 的用途与使用方法。

## 1) 训练主脚本

### `examples/hi_xiaowen/s0/run_fsmn_ctc.sh`
**用途**：完整数据准备、训练、评测、导出。

**常用用法**：
- 训练：  
  `bash run_fsmn_ctc.sh 2 2`
- 指定实验目录：  
  `bash run_fsmn_ctc.sh --target_exp_dir exp/my_exp 2 2`
- 指定 GPU：  
  `bash run_fsmn_ctc.sh --gpus "0,1,2,3" 2 2`
- 自动按 `num_keywords` 生成词表：  
  `bash run_fsmn_ctc.sh --target_exp_dir exp/fsmn_ctc_top1000 --gpus "0,1,2,3" --dict_dir dict_top1000 --num_keywords 1000 --dict_auto_build true --dict_sorted_file examples/hi_xiaowen/s0/dict/model_vocab_freq_asr_sorted.txt --checkpoint_strict false 2 2`
- Stage 1.5（构建 metadata.db）：  
  `bash run_fsmn_ctc.sh 1.5 1.5`

**关键参数**：
- `--target_exp_dir`：实验输出目录
- `--gpus`：GPU 列表（逗号分隔）
- `--dict_dir`：词表目录（包含 `dict.txt`/`words.txt`）
- `--num_keywords`：输出维度，需与词表大小一致
- `--dict_auto_build`：训练前自动生成词表（true/false）
- `--dict_sorted_file`：词频排序文件路径（用于自动生成词表）
- `--checkpoint_strict`：是否严格加载 checkpoint（改词表时建议 `false`）
- `stage/stop_stage`：控制流程阶段

## 2) 评测与单条推理

### `examples/hi_xiaowen/s0/evaluate.sh`
**用途**：评测模型在 train/dev/test 上的检出率、DET 曲线。

**用法**：
```bash
bash evaluate.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/61.pt --dataset test --gpu 0
```

**注意**：
- 评测时需使用与训练一致的词表目录：`--dict_dir <词表目录>`

**关键参数**：
- `--checkpoint`（必填）：模型权重
- `--dataset`：`train|dev|test`，默认 `test`
- `--gpu`：使用的 GPU（多卡时取第一个）

**输出**：  
`<checkpoint_dir>/<dataset>_<checkpoint_name>/`  
包含 `score.txt` 和 `det_*.png`。

---

### `examples/hi_xiaowen/s0/analyze_exp_test_stats.py`
**用途**：汇总 `exp/**/test_N` 目录下的 `stats.*.txt`，输出两个唤醒词的准确率/误检率表格。

**用法**：
```bash
python analyze_exp_test_stats.py --test-id 2
python analyze_exp_test_stats.py --test-id 79
```

**常用参数**：
- `--test-id`：指定 `test_N` 中的 N
- `--target-fa-per-hour`：阈值选择上限（单位：次/小时），例如 12 小时一次可设 `1/12`
- `--gen-stats`：当缺少 `stats.*.txt` 时，尝试用 `score.txt` 生成
- `--label-file`：用于生成 stats 的 `data.list`，默认 `data/test/data.list`

**输出说明**：
- `accuracy`、`frr` 为百分比（保留 2 位小数）
- `fa_per_hour` 为“每小时误检次数”（次/小时），**不**是百分比

---

### `examples/hi_xiaowen/s0/infer_wav.sh`
**用途**：单条 wav 或 utt_id 推理，输出 score / logits / decode。

**用法**：
```bash
bash infer_wav.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/79.pt --wav /path/to/foo.wav --gpu 0
bash infer_wav.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/79.pt --wav 0000010f3cc9ff4c868117b4e4c53fb5 --gpu 0
```

**说明**：
- 优先从 `data/metadata.db` 获取 `split`，输出目录形如 `train_79/dev_79/test_79`。
- `--dataset` 可覆盖默认输出目录名。

**输出**：  
`<checkpoint_dir>/<dataset>_<checkpoint_name>/`  
包含 `score.single.*.txt`、`dump_logits/`、`decode.single.*.jsonl` 等。

## 3) Wayne Scripts（数据统计与可视化）

### `examples/hi_xiaowen/s0/wayne_scripts/stage_-1_statistics.sh`
**用途**：统计 Stage -1 数据准备后的正/负样本分布（train/dev/test）。

**用法**：
```bash
bash wayne_scripts/stage_-1_statistics.sh
```

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_-2_copy_wav.sh`
**用途**：按 utt_id 从原始数据目录复制单条 wav 到当前目录。

**用法**：
```bash
bash wayne_scripts/stage_-2_copy_wav.sh <utt_id>
```

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_-2_statistics.py`
**用途**：统计 wav 时长分布、JSON 中的多维分布（距离/角度/噪声/说话人等），可生成图表。

**用法**：
```bash
python3 wayne_scripts/stage_-2_statistics.py
```

**依赖**：
- `pywayne`（读取配置）
- `matplotlib` / `seaborn`（可视化）

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_0_dict_analysis.sh`
**用途**：分析 Stage 0 前后 dict/words 的变化，输出详细解释和统计。

**用法**：
```bash
bash wayne_scripts/stage_0_dict_analysis.sh
```

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_0_statistics.sh`
**用途**：说明 Stage 0 的作用，并统计字典/转录文本的变化情况。

**用法**：
```bash
bash wayne_scripts/stage_0_statistics.sh
```

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_1_statistics.sh`
**用途**：说明 Stage 1 特征归一化与 data.list 生成流程，输出详细说明和统计。

**用法**：
```bash
bash wayne_scripts/stage_1_statistics.sh
```

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_1.5_webui.sh`
**用途**：启动数据浏览 WebUI（streamlit）。

**用法**：
```bash
bash wayne_scripts/stage_1.5_webui.sh
```

**前置条件**：
- 已生成 `data/metadata.db`（`bash run_fsmn_ctc.sh 1.5 1.5`）
- 已安装 `streamlit`

---

### `examples/hi_xiaowen/s0/wayne_scripts/stage_1_visualize.py`
**用途**：可视化音频波形/FBANK/CMVN 等特征。

**用法**：
```bash
python wayne_scripts/stage_1_visualize.py <audio_id>
python wayne_scripts/stage_1_visualize.py <audio_id> --output-dir ./visualizations
python wayne_scripts/stage_1_visualize.py <audio_id> --no-show
```

## 4) 词表统计脚本

### `tools/analyze_asr_vocab.py`
**用途**：
1) 统计 `dict.txt` 中每个 token 在 ASR 转写中的频率  
2) 输出模型词表（`output_dim`）对应的 token  
3) 统计模型词表 token 在 ASR 转写中的频率  
4) 生成按频率降序的词表统计（含累计占比）

**用法**：
```bash
python tools/analyze_asr_vocab.py
```

**输出（默认）**：
`examples/hi_xiaowen/s0/dict/` 下生成：
- `dict_token_freq_asr.txt`
- `model_vocab_2599.txt`
- `model_vocab_freq_asr.txt`
- `model_vocab_freq_asr_sorted.txt`

**可选参数**：
```bash
python tools/analyze_asr_vocab.py \
  --dict examples/hi_xiaowen/s0/dict/dict.txt \
  --data_lists examples/hi_xiaowen/s0/data \
  --output_dir examples/hi_xiaowen/s0/dict \
  --config examples/hi_xiaowen/s0/exp/fsmn_ctc_baseline_4gpus/config.yaml \
  --vocab_file examples/hi_xiaowen/s0/speech_charctc_kws_phone-xiaoyun/train/tokens.txt \
  --output_dim 2599
```

**说明**：
- 默认 `--data_lists` 为 `examples/hi_xiaowen/s0/data`，会递归收集所有 `data.list`
- `--config` 可自动推断 `output_dim`，`--output_dim` 会覆盖该值
- `--vocab_file` 用于指定模型词表来源（如预训练模型的 `tokens.txt`）
- 词表构建规则为：从 `tokens.txt` 中取 **id <= output_dim** 的所有 token

## 实验管理注意事项（务必遵守）

- 不直接修改 baseline 训练产物；所有实验使用单独的脚本/配置/词表文件
- 如果只改词表，生成新的 `dict.txt`/`words.txt`（放到独立目录），训练脚本显式指向新词表
- 需要改训练流程时，优先新建脚本（或带后缀命名的补丁脚本），不要覆盖原始脚本

