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

## ⚙️ 配置文件

**配置文件：** `config.yaml`（需要手动创建）

```yaml
# 数据路径
download_dir: /home/data/datasets/kws/opensourced/nihaowenwen
wav_dir: mobvoi_hotword_dataset
json_dir: mobvoi_hotword_dataset_resources

# 输出路径
statistics_plots_dir: ./statistics_plots
visualizations_dir: ./visualizations

# 多线程配置
max_workers: null  # null = 自动检测
```

**首次使用：**
```bash
cp config.yaml.example config.yaml
vim config.yaml  # 修改 download_dir
```

## 🚀 使用

```bash
# 数据集统计
python3 stage_-2_statistics.py

# 音频可视化
python3 stage_1_visualize.py <audio_id>

# Stage 说明
sh stage_0_statistics.sh
sh stage_1_statistics.sh
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

