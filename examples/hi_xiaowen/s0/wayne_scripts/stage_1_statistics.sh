#!/bin/bash
# Stage 1 介绍脚本 - 数据预处理与特征归一化
# 作者: Wayne
# 功能: 说明 Stage 1 做了什么、输入输出是什么、在哪里

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
TOOLS_DIR="$PROJECT_DIR/../../../tools"

echo "================================================================================"
echo "                   【Stage 1】数据预处理与特征归一化"
echo "================================================================================"
echo ""
echo "脚本路径: $SCRIPT_DIR/$(basename $0)"
echo "项目路径: $PROJECT_DIR"
echo ""

echo "【1】Stage 1 概述】"
echo "================================================================================"
echo ""
echo "  Stage 1 是训练前的最后一个数据预处理阶段，主要完成三个关键任务："
echo ""
echo "  1️⃣  计算全局 CMVN 统计量 (倒谱均值方差归一化)"
echo "     - 目的: 提高模型训练的稳定性和收敛速度"
echo "     - 原理: 将音频特征归一化到统一的分布"
echo ""
echo "  2️⃣  计算所有音频的时长"
echo "     - 目的: 用于数据采样、批处理和训练策略"
echo "     - 用途: 动态批处理、时长过滤等"
echo ""
echo "  3️⃣  生成统一的数据列表文件"
echo "     - 目的: 将分散的信息整合成训练可用的格式"
echo "     - 格式: JSON Lines (每行一个 JSON 对象)"
echo ""

echo "【2】Stage 1 详细流程】"
echo "================================================================================"
echo ""

echo "步骤 1: 计算全局 CMVN 统计量"
echo "------------------------------------------------------------"
echo ""
echo "  命令:"
echo "    tools/compute_cmvn_stats.py --num_workers 16 \\"
echo "      --train_config conf/fsmn_ctc.yaml \\"
echo "      --in_scp data/train/wav.scp \\"
echo "      --out_cmvn data/train/global_cmvn"
echo ""
echo "  📥 输入文件:"
echo "    1) data/train/wav.scp"
echo "       格式: <utt_id> <wav_path>"
echo "       说明: 训练集音频路径列表"
echo ""
echo "    2) conf/fsmn_ctc.yaml"
echo "       说明: 包含特征提取配置 (FBANK 维度、采样率等)"
echo ""
echo "  📤 输出文件:"
echo "    data/train/global_cmvn"
echo "       内容: 特征的均值和方差统计量"
echo "       用途: 训练和推理时的特征归一化"
echo ""
echo "  🔬 技术原理:"
echo "    - CMVN: Cepstral Mean and Variance Normalization"
echo "    - 公式: feature_normalized = (feature - mean) / sqrt(variance)"
echo "    - 作用: 减少说话人、录音设备、环境噪声的影响"
echo "    - 时机: 在特征提取后、模型输入前应用"
echo ""
echo "  ⚙️  处理流程:"
echo "    1. 遍历所有训练音频"
echo "    2. 提取 FBANK 特征 (通常 80 维)"
echo "    3. 累积统计量 (均值和方差)"
echo "    4. 保存全局统计量到文件"
echo ""

echo "步骤 2: 计算音频时长"
echo "------------------------------------------------------------"
echo ""
echo "  命令 (对每个数据集):"
echo "    tools/wav_to_duration.sh --nj 8 \\"
echo "      data/\$x/wav.scp \\"
echo "      data/\$x/wav.dur"
echo ""
echo "  📥 输入文件 (针对 train/dev/test):"
echo "    data/train/wav.scp"
echo "    data/dev/wav.scp"
echo "    data/test/wav.scp"
echo ""
echo "  📤 输出文件:"
echo "    data/train/wav.dur"
echo "    data/dev/wav.dur"
echo "    data/test/wav.dur"
echo "    格式: <utt_id> <duration_in_seconds>"
echo ""
echo "  ⚙️  处理流程:"
echo "    1. 将 wav.scp 分割成 8 个子任务 (--nj 8)"
echo "    2. 并行读取每个音频的元信息"
echo "    3. 计算时长 (样本数 / 采样率)"
echo "    4. 合并所有结果到 wav.dur"
echo ""
echo "  💡 时长用途:"
echo "    - 动态批处理 (按时长分组，提高 GPU 利用率)"
echo "    - 数据过滤 (过滤过长或过短的音频)"
echo "    - 训练策略 (如课程学习：先训练短音频)"
echo ""

echo "步骤 3: 生成数据列表"
echo "------------------------------------------------------------"
echo ""
echo "  命令 (对每个数据集):"
echo "    tools/make_list.py \\"
echo "      data/\$x/wav.scp \\"
echo "      data/\$x/text \\"
echo "      data/\$x/wav.dur \\"
echo "      data/\$x/data.list"
echo ""
echo "  📥 输入文件 (以 train 为例):"
echo "    1) data/train/wav.scp"
echo "       格式: <utt_id> <wav_path>"
echo ""
echo "    2) data/train/text"
echo "       格式: <utt_id> <text_content>"
echo "       示例: 68c08ef7... 嗨 小 问"
echo ""
echo "    3) data/train/wav.dur"
echo "       格式: <utt_id> <duration>"
echo "       示例: 68c08ef7... 2.56"
echo ""
echo "  📤 输出文件:"
echo "    data/train/data.list"
echo "    data/dev/data.list"
echo "    data/test/data.list"
echo ""
echo "    格式: JSON Lines (每行一个 JSON 对象)"
echo ""
echo "    示例:"
echo '    {"key":"68c08ef7...", "txt":"嗨 小 问", "duration":2.56, "wav":"..."}'
echo '    {"key":"461003fa...", "txt":"if you don'\''t", "duration":3.12, "wav":"..."}'
echo ""
echo "  ⚙️  处理流程:"
echo "    1. 读取三个输入文件，建立映射表"
echo "    2. 对每个 utterance，组合信息"
echo "    3. 将文本分词 (中英文混合处理)"
echo "    4. 生成 JSON 格式的数据列表"
echo ""
echo "  🔤 文本分词说明:"
echo "    - 中文: 按字分 (\"嗨小问\" → \"嗨 小 问\")"
echo "    - 英文: 按词分 (\"hello world\" → \"hello world\")"
echo "    - 混合: 智能切分 (\"嗨hello\" → \"嗨 hello\")"
echo ""

echo "【3】输入输出文件总结】"
echo "================================================================================"
echo ""

echo "📥 输入文件 (来自 Stage -1 或 Stage -0)"
echo "------------------------------------------------------------"
echo ""
if [ -d "$DATA_DIR" ]; then
    for dataset in train dev test; do
        echo "  $dataset 数据集:"
        echo "    路径: data/$dataset/"
        echo ""
        
        if [ -f "$DATA_DIR/$dataset/wav.scp" ]; then
            wav_count=$(wc -l < "$DATA_DIR/$dataset/wav.scp")
            echo "    ✅ wav.scp       存在 (${wav_count} 条)"
        else
            echo "    ❌ wav.scp       不存在"
        fi
        
        if [ -f "$DATA_DIR/$dataset/text" ]; then
            text_count=$(wc -l < "$DATA_DIR/$dataset/text")
            echo "    ✅ text          存在 (${text_count} 条)"
        else
            echo "    ❌ text          不存在"
        fi
        
        echo ""
    done
    
    echo "  配置文件:"
    if [ -f "$PROJECT_DIR/conf/fsmn_ctc.yaml" ]; then
        echo "    ✅ conf/fsmn_ctc.yaml    存在"
    else
        echo "    ❌ conf/fsmn_ctc.yaml    不存在"
    fi
    echo ""
else
    echo "  ⚠️  数据目录不存在: $DATA_DIR"
    echo ""
fi

echo "📤 输出文件 (供 Stage 2 训练使用)"
echo "------------------------------------------------------------"
echo ""
if [ -d "$DATA_DIR" ]; then
    echo "  全局特征统计:"
    if [ -f "$DATA_DIR/train/global_cmvn" ]; then
        cmvn_size=$(ls -lh "$DATA_DIR/train/global_cmvn" 2>/dev/null | awk '{print $5}')
        echo "    ✅ data/train/global_cmvn    存在 (${cmvn_size:-未知大小})"
        echo "       内容: 特征均值和方差"
        echo "       用途: 训练和推理时的归一化"
    else
        echo "    ❌ data/train/global_cmvn    不存在 (Stage 1 未运行)"
    fi
    echo ""
    
    echo "  音频时长文件:"
    for dataset in train dev test; do
        if [ -f "$DATA_DIR/$dataset/wav.dur" ]; then
            dur_count=$(wc -l < "$DATA_DIR/$dataset/wav.dur")
            echo "    ✅ data/$dataset/wav.dur     存在 (${dur_count} 条)"
        else
            echo "    ❌ data/$dataset/wav.dur     不存在"
        fi
    done
    echo ""
    
    echo "  数据列表文件 (JSON):"
    for dataset in train dev test; do
        if [ -f "$DATA_DIR/$dataset/data.list" ]; then
            list_count=$(wc -l < "$DATA_DIR/$dataset/data.list")
            list_size=$(ls -lh "$DATA_DIR/$dataset/data.list" 2>/dev/null | awk '{print $5}')
            echo "    ✅ data/$dataset/data.list   存在 (${list_count} 条, ${list_size:-未知大小})"
        else
            echo "    ❌ data/$dataset/data.list   不存在"
        fi
    done
    echo ""
else
    echo "  ⚠️  数据目录不存在: $DATA_DIR"
    echo ""
fi

echo "【4】data.list 文件格式详解】"
echo "================================================================================"
echo ""
echo "  data.list 是训练的核心输入文件，采用 JSON Lines 格式"
echo ""
echo "  每行结构:"
echo "  {"
echo "    \"key\": \"<utt_id>\",           # 唯一标识符"
echo "    \"txt\": \"<分词后的文本>\",     # 用空格分隔的字符/词"
echo "    \"duration\": <时长秒数>,      # 浮点数"
echo "    \"wav\": \"<音频路径>\"         # 完整路径或相对路径"
echo "  }"
echo ""

if [ -f "$DATA_DIR/train/data.list" ]; then
    echo "  实际样例 (来自 data/train/data.list):"
    echo "  ----------------------------------------------------------------"
    head -3 "$DATA_DIR/train/data.list" 2>/dev/null | while IFS= read -r line; do
        echo "  $line"
    done
    echo "  ----------------------------------------------------------------"
    echo ""
fi

echo "【5】CMVN 归一化原理】"
echo "================================================================================"
echo ""
echo "  什么是 CMVN?"
echo "  ----------------------------------------------------------------"
echo "  CMVN (Cepstral Mean and Variance Normalization)"
echo "  倒谱均值方差归一化，是语音识别中常用的特征归一化技术"
echo ""
echo "  为什么需要 CMVN?"
echo "  ----------------------------------------------------------------"
echo "  1. 不同说话人的声学特征差异很大"
echo "  2. 不同录音设备有不同的频率响应"
echo "  3. 环境噪声会影响特征分布"
echo "  4. 归一化后的特征更利于模型训练"
echo ""
echo "  CMVN 如何工作?"
echo "  ----------------------------------------------------------------"
echo "  步骤 1: 在训练集上统计特征的均值和方差"
echo "    mean = E[feature]"
echo "    var = E[(feature - mean)^2]"
echo ""
echo "  步骤 2: 对每个特征向量进行归一化"
echo "    feature_norm = (feature - mean) / sqrt(var + eps)"
echo ""
echo "  步骤 3: 训练和推理时都使用相同的均值和方差"
echo "    训练: 使用 data/train/global_cmvn"
echo "    推理: 使用相同的统计量保证一致性"
echo ""
echo "  CMVN 的效果:"
echo "  ----------------------------------------------------------------"
echo "  • 降低 Word Error Rate (WER) 约 5-10%"
echo "  • 提高模型对不同说话人和环境的鲁棒性"
echo "  • 加快训练收敛速度"
echo "  • 减少过拟合风险"
echo ""

echo "【6】Stage 1 运行状态检查】"
echo "================================================================================"
echo ""

stage1_done=true

# 检查 global_cmvn
if [ -f "$DATA_DIR/train/global_cmvn" ]; then
    echo "  ✅ CMVN 统计量已生成"
else
    echo "  ❌ CMVN 统计量未生成"
    stage1_done=false
fi

# 检查 wav.dur
all_dur_exist=true
for dataset in train dev test; do
    if [ ! -f "$DATA_DIR/$dataset/wav.dur" ]; then
        all_dur_exist=false
        break
    fi
done

if $all_dur_exist; then
    echo "  ✅ 所有音频时长已计算"
else
    echo "  ❌ 音频时长未完全计算"
    stage1_done=false
fi

# 检查 data.list
all_list_exist=true
for dataset in train dev test; do
    if [ ! -f "$DATA_DIR/$dataset/data.list" ]; then
        all_list_exist=false
        break
    fi
done

if $all_list_exist; then
    echo "  ✅ 所有数据列表已生成"
else
    echo "  ❌ 数据列表未完全生成"
    stage1_done=false
fi

echo ""
if $stage1_done; then
    echo "  🎉 Stage 1 已成功完成！可以进行 Stage 2 (训练)"
else
    echo "  ⚠️  Stage 1 未完成或部分文件缺失"
    echo ""
    echo "  运行方法:"
    echo "    cd $PROJECT_DIR"
    echo "    bash run_fsmn_ctc.sh 1 1"
fi
echo ""

echo "【7】数据统计 (如果 Stage 1 已运行)】"
echo "================================================================================"
echo ""

if $stage1_done; then
    for dataset in train dev test; do
        dataset_upper=$(echo $dataset | tr '[:lower:]' '[:upper:]')
        echo "  【${dataset_upper}】数据集"
        echo "  ------------------------------------------------------------"
        
        if [ -f "$DATA_DIR/$dataset/data.list" ]; then
            total_samples=$(wc -l < "$DATA_DIR/$dataset/data.list")
            echo "    总样本数: $total_samples"
            
            # 统计总时长
            total_duration=$(python3 -c "
import json
import sys
total = 0.0
with open('$DATA_DIR/$dataset/data.list', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        total += data['duration']
print(f'{total:.2f}')
" 2>/dev/null)
            
            if [ -n "$total_duration" ]; then
                total_hours=$(python3 -c "print(f'{$total_duration/3600:.2f}')" 2>/dev/null)
                echo "    总时长: ${total_duration}s (${total_hours}h)"
                
                avg_duration=$(python3 -c "print(f'{$total_duration/$total_samples:.2f}')" 2>/dev/null)
                echo "    平均时长: ${avg_duration}s"
            fi
            
            # 显示示例
            echo ""
            echo "    数据示例:"
            head -2 "$DATA_DIR/$dataset/data.list" 2>/dev/null | python3 -c "
import json
import sys
for i, line in enumerate(sys.stdin, 1):
    data = json.loads(line.strip())
    print(f'      样本 {i}:')
    print(f'        key: {data[\"key\"][:20]}...')
    print(f'        txt: {data[\"txt\"][:50]}...')
    print(f'        duration: {data[\"duration\"]}s')
    print()
" 2>/dev/null
        fi
        echo ""
    done
    
    # CMVN 信息
    if [ -f "$DATA_DIR/train/global_cmvn" ]; then
        echo "  【CMVN 统计量】"
        echo "  ------------------------------------------------------------"
        cmvn_size=$(wc -c < "$DATA_DIR/train/global_cmvn")
        echo "    文件大小: $cmvn_size bytes"
        echo "    文件路径: data/train/global_cmvn"
        echo ""
        echo "    内容说明:"
        echo "      - 特征维度: 通常 80 维 (FBANK)"
        echo "      - 存储格式: Kaldi 格式 (均值向量 + 方差向量)"
        echo "      - 用途: 训练和推理时的特征归一化"
        echo ""
    fi
else
    echo "  ⚠️  Stage 1 未运行，无统计数据"
    echo ""
fi

echo "【8】与其他 Stage 的关系】"
echo "================================================================================"
echo ""
echo "  流程图:"
echo "  ----------------------------------------------------------------"
echo ""
echo "    Stage -1: 准备数据"
echo "       ↓ 生成 wav.scp + text"
echo ""
echo "    Stage -0: 替换为 ASR 转录 (可选)"
echo "       ↓ 更新 text 和 dict"
echo ""
echo "    Stage 1: 特征归一化与数据格式化  ← 当前阶段"
echo "       ↓ 生成 global_cmvn + data.list"
echo ""
echo "    Stage 2: 模型训练"
echo "       ↓ 使用 data.list 训练"
echo ""
echo "    Stage 3: 模型评估"
echo "       ↓ 计算 FRR/FAR"
echo ""
echo "    Stage 4: 模型导出"
echo "       ↓ 导出 ONNX 模型"
echo ""

echo "【9】常见问题 (FAQ)】"
echo "================================================================================"
echo ""
echo "  Q1: Stage 1 需要运行多久?"
echo "  ----------------------------------------------------------------"
echo "  答: 取决于数据量和硬件"
echo "    • CMVN 计算: 约 5-30 分钟 (取决于训练集大小)"
echo "    • 时长计算: 约 1-5 分钟 (并行处理)"
echo "    • 列表生成: 约 1 分钟"
echo "    • 总计: 约 10-40 分钟"
echo ""
echo "  Q2: 可以跳过 Stage 1 吗?"
echo "  ----------------------------------------------------------------"
echo "  答: 不可以"
echo "    • data.list 是训练的必需输入"
echo "    • global_cmvn 对模型性能至关重要"
echo "    • 跳过会导致训练失败或性能大幅下降"
echo ""
echo "  Q3: 修改数据后需要重新运行 Stage 1 吗?"
echo "  ----------------------------------------------------------------"
echo "  答: 是的"
echo "    • 如果修改了 wav.scp 或 text，必须重新运行"
echo "    • 如果只修改配置文件，可能不需要重新计算 CMVN"
echo "    • 建议删除旧的输出文件，重新运行"
echo ""
echo "  Q4: global_cmvn 只用训练集计算，dev/test 怎么办?"
echo "  ----------------------------------------------------------------"
echo "  答: 这是正确的做法"
echo "    • 训练集统计量应用到所有数据集"
echo "    • 这样保证训练和测试的一致性"
echo "    • dev/test 不参与统计量计算，避免信息泄露"
echo ""
echo "  Q5: data.list 中的 txt 为什么要分词?"
echo "  ----------------------------------------------------------------"
echo "  答: 适配字符级 CTC 模型"
echo "    • Stage -1: \"<HI_XIAOWEN>\" (词级标签)"
echo "    • Stage 0:  \"嗨 小 问\" (字符级标签)"
echo "    • 分词后才能与 dict.txt 中的字符对应"
echo "    • 支持中英文混合识别"
echo ""

echo "【10】运行命令】"
echo "================================================================================"
echo ""
echo "  完整命令:"
echo "  ----------------------------------------------------------------"
echo "    cd $PROJECT_DIR"
echo "    bash run_fsmn_ctc.sh 1 1"
echo ""
echo "  单独运行各步骤 (调试用):"
echo "  ----------------------------------------------------------------"
echo "    # 1. 计算 CMVN"
echo "    tools/compute_cmvn_stats.py --num_workers 16 \\"
echo "      --train_config conf/fsmn_ctc.yaml \\"
echo "      --in_scp data/train/wav.scp \\"
echo "      --out_cmvn data/train/global_cmvn"
echo ""
echo "    # 2. 计算时长"
echo "    for x in train dev test; do"
echo "      tools/wav_to_duration.sh --nj 8 \\"
echo "        data/\$x/wav.scp data/\$x/wav.dur"
echo "    done"
echo ""
echo "    # 3. 生成列表"
echo "    for x in train dev test; do"
echo "      tools/make_list.py \\"
echo "        data/\$x/wav.scp data/\$x/text \\"
echo "        data/\$x/wav.dur data/\$x/data.list"
echo "    done"
echo ""

echo "================================================================================"
echo "分析完成！"
echo "================================================================================"
echo ""
echo "建议:"
echo "  • Stage 1 是训练前的最后一步，务必确保运行成功"
echo "  • 检查生成的 data.list 文件，确保格式正确"
echo "  • CMVN 对模型性能影响很大，不要跳过"
echo "  • 如果遇到问题，检查日志文件: data/*/log/"
echo ""
echo "下一步:"
echo "  • 运行 Stage 2 开始训练: bash run_fsmn_ctc.sh 2 2"
echo "  • 查看训练配置: cat conf/fsmn_ctc.yaml"
echo "  • 监控训练日志: tail -f exp/fsmn_ctc/train.log"
echo ""
