#!/bin/bash
# Stage 0 统计脚本
# 说明 Stage 0 (实际是 stage -0) 做了什么

echo "================================================================================"
echo "Stage 0 (stage -0) 数据增强阶段统计"
echo "================================================================================"
echo ""
echo "【Stage 0 做了什么】"
echo ""
echo "  1. 克隆 ASR 转录数据仓库"
echo "     来源: https://www.modelscope.cn/datasets/thuduj12/mobvoi_kws_transcription.git"
echo "     说明: 使用 Paraformer Large ASR 模型对负样本音频进行转录"
echo ""
echo "  2. 替换文本标签为实际转录内容"
echo "     之前 (Stage -1):"
echo "       正样本: <HI_XIAOWEN> / <NIHAO_WENWEN>"
echo "       负样本: <FILLER>"
echo ""
echo "     之后 (Stage 0):"
echo "       正样本: 嗨 小 问 / 你 好 问 问"
echo "       负样本: ASR 转录的实际文本 (如: 今 天 天 气 怎 么 样)"
echo ""
echo "     操作: 将原 text 备份为 text.label，复制新的转录文本"
echo ""
echo "  3. 更新字典文件 (dict/dict.txt)"
echo "     从: 3 个关键词标签 (<FILLER>, <HI_XIAOWEN>, <NIHAO_WENWEN>)"
echo "     到: 2787 个 token (音素/字符级别)"
echo "     来源: mobvoi_kws_transcription/tokens.txt"
echo ""
echo "  4. 更新词表文件 (dict/words.txt)"
echo "     设置为: <SILENCE>, <EPS>, <BLK>"
echo ""
echo "  目的:"
echo "    - 使用实际语音内容训练，而非简单的关键词标签"
echo "    - 提高模型对真实语音环境的适应能力"
echo "    - 支持字符级别的 CTC 训练"
echo ""
echo "================================================================================"
echo "【文件变化统计】"
echo "================================================================================"
echo ""

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."
DATA_DIR="$BASE_DIR/data"
DICT_DIR="$BASE_DIR/dict"
TRANS_DIR="$BASE_DIR/mobvoi_kws_transcription"

# 检查是否已运行 stage 0
echo "检查 Stage 0 运行状态..."
echo ""

if [ -d "$TRANS_DIR" ]; then
    echo "✓ mobvoi_kws_transcription 目录存在"
    echo "  路径: $TRANS_DIR"
else
    echo "✗ mobvoi_kws_transcription 目录不存在"
    echo "  说明: Stage 0 可能尚未运行"
fi
echo ""

# 检查字典文件
if [ -f "$DICT_DIR/dict.txt" ]; then
    dict_lines=$(wc -l < "$DICT_DIR/dict.txt")
    echo "✓ dict/dict.txt"
    echo "  Token 数量: $dict_lines"
    
    if [ $dict_lines -gt 100 ]; then
        echo "  状态: 已更新为详细字典 (Stage 0 已运行)"
    else
        echo "  状态: 简单标签字典 (Stage 0 未运行)"
    fi
else
    echo "✗ dict/dict.txt 不存在"
fi
echo ""

if [ -f "$DICT_DIR/words.txt" ]; then
    words_lines=$(wc -l < "$DICT_DIR/words.txt")
    echo "✓ dict/words.txt"
    echo "  词数量: $words_lines"
    
    if grep -q "<SILENCE>" "$DICT_DIR/words.txt" 2>/dev/null; then
        echo "  状态: 包含 <SILENCE> (Stage 0 已运行)"
    else
        echo "  状态: 原始词表 (Stage 0 未运行)"
    fi
else
    echo "✗ dict/words.txt 不存在"
fi
echo ""

echo "================================================================================"
echo "【数据文本格式对比】"
echo "================================================================================"
echo ""

# 检查 text 和 text.label 文件
for dataset in train dev test; do
    text_file="$DATA_DIR/$dataset/text"
    label_file="$DATA_DIR/$dataset/text.label"
    
    echo "【$(echo $dataset | tr '[:lower:]' '[:upper:]')】数据集"
    echo "----------------------------------------"
    
    if [ -f "$text_file" ]; then
        total_lines=$(wc -l < "$text_file")
        echo "✓ text 文件存在 (行数: $total_lines)"
        
        # 检测是否是转录后的格式
        if head -1 "$text_file" | grep -q "[嗨你]"; then
            echo "  格式: ASR 转录文本 (汉字)"
            echo "  示例:"
            head -3 "$text_file" | sed 's/^/    /'
        elif head -1 "$text_file" | grep -q "<"; then
            echo "  格式: 关键词标签"
            echo "  示例:"
            head -3 "$text_file" | sed 's/^/    /'
        fi
        echo ""
        
        # 统计正负样本 (基于转录文本)
        if head -1 "$text_file" | grep -q "[嗨你]"; then
            hi_xiaowen=$(grep -c "嗨 小 问" "$text_file" 2>/dev/null || echo "0")
            nihao_wenwen=$(grep -c "你 好 问 问" "$text_file" 2>/dev/null || echo "0")
            positive=$((hi_xiaowen + nihao_wenwen))
            negative=$((total_lines - positive))
            
            echo "  正样本 (嗨 小 问): $hi_xiaowen"
            echo "  正样本 (你 好 问 问): $nihao_wenwen"
            echo "  正样本小计: $positive"
            echo "  负样本 (其他): $negative"
            echo ""
        fi
    else
        echo "✗ text 文件不存在"
        echo ""
    fi
    
    if [ -f "$label_file" ]; then
        label_lines=$(wc -l < "$label_file")
        echo "✓ text.label 备份存在 (行数: $label_lines)"
        echo "  说明: 这是 Stage -1 的原始标签文件"
        echo "  示例:"
        head -3 "$label_file" | sed 's/^/    /'
        echo ""
    fi
    
    echo ""
done

echo "================================================================================"
echo "【转录文本统计】"
echo "================================================================================"
echo ""

if [ -d "$TRANS_DIR" ]; then
    echo "mobvoi_kws_transcription 仓库内容:"
    echo ""
    
    for dataset in train dev test; do
        trans_file="$TRANS_DIR/$dataset.text"
        if [ -f "$trans_file" ]; then
            lines=$(wc -l < "$trans_file")
            printf "  %-10s: %8d 行\n" "$dataset.text" $lines
        fi
    done
    echo ""
    
    if [ -f "$TRANS_DIR/tokens.txt" ]; then
        token_lines=$(wc -l < "$TRANS_DIR/tokens.txt")
        echo "  tokens.txt: $token_lines tokens"
    fi
    
    if [ -f "$TRANS_DIR/lexicon.txt" ]; then
        lexicon_lines=$(wc -l < "$TRANS_DIR/lexicon.txt")
        echo "  lexicon.txt: $lexicon_lines 词条"
    fi
    echo ""
    
    echo "文件列表:"
    ls -lh "$TRANS_DIR" | grep -v "^d" | awk '{printf "  %-25s %8s\n", $9, $5}'
else
    echo "mobvoi_kws_transcription 目录不存在"
    echo "请先运行 stage 0 以克隆转录数据"
fi

echo ""
echo "================================================================================"
echo "【使用建议】"
echo "================================================================================"
echo ""
echo "Stage 0 是可选的增强步骤:"
echo ""
echo "  不运行 Stage 0:"
echo "    - 使用简单的关键词标签训练"
echo "    - 训练速度较快"
echo "    - 适合快速原型验证"
echo ""
echo "  运行 Stage 0:"
echo "    - 使用 ASR 转录的实际文本训练"
echo "    - 模型学习更丰富的语音特征"
echo "    - 提高在真实场景下的鲁棒性"
echo "    - 支持字符级 CTC 训练"
echo ""
echo "================================================================================"
echo "统计完成！"
echo "================================================================================"
