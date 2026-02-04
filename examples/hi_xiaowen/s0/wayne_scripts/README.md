# Wayne's Scripts

各个 Stage 的统计和分析脚本。

## 📂 脚本列表

| 脚本 | 说明 |
|------|------|
| `stage_-2_statistics.py` | 数据集统计（WAV 和 JSON） |
| `stage_-2_copy_wav.sh` | 复制 WAV 文件 |
| `stage_-1_statistics.sh` | Stage -1 数据准备统计 |
| `stage_0_statistics.sh` | Stage 0 ASR 转录说明 |
| `stage_0_dict_analysis.sh` | 字典和词表分析 |
| `stage_1_statistics.sh` | Stage 1 CMVN 和数据格式化说明 |
| `stage_1_visualize.py` | 音频特征可视化工具 |
| `stage_1.5_webui.sh` | 启动 WebUI 数据浏览器 |

## 🚀 使用

```bash
# 数据集统计
python3 stage_-2_statistics.py

# Stage 说明
sh stage_0_statistics.sh
sh stage_1_statistics.sh

# 音频可视化
python3 stage_1_visualize.py <audio_id>
```

## 📊 WebUI

```bash
# 1. 构建数据库
cd ..
bash run_fsmn_ctc.sh 1.5 1.5

# 2. 启动 WebUI
cd wayne_scripts
sh stage_1.5_webui.sh
```
