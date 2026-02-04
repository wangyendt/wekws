#!/usr/bin/env bash
# Stage -1 统计脚本
# 统计 train/dev/test 数据集的正负样本分布

echo "================================================================================"
echo "Stage -1 数据准备阶段统计"
echo "================================================================================"
echo ""
echo "【Stage -1 做了什么】"
echo "  1. 创建字典文件 (dict/dict.txt, dict/words.txt)"
echo "     - <FILLER> -1     : 负样本（非关键词）"
echo "     - <HI_XIAOWEN> 0  : 正样本（嗨小问）"
echo "     - <NIHAO_WENWEN> 1: 正样本（你好问问）"
echo ""
echo "  2. 从 JSON 文件准备 Kaldi 格式数据"
echo "     - 读取 mobvoi_hotword_dataset_resources/{p,n}_{train,dev,test}.json"
echo "     - 为每个数据集生成 wav.scp 和 text 文件"
echo "     - 合并正样本(p_*)和负样本(n_*)到统一目录"
echo ""
echo "  3. 输出目录结构:"
echo "     - data/train/{wav.scp, text}"
echo "     - data/dev/{wav.scp, text}"
echo "     - data/test/{wav.scp, text}"
echo ""
echo "================================================================================"
echo "【数据集统计】"
echo "================================================================================"
echo ""
echo "注意: 脚本会自动检测文件状态"
echo "  - 如果运行了 Stage 0: 使用 text.label (原始标签)"
echo "  - 如果只运行了 Stage -1: 使用 text (原始标签)"
echo ""

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

# 检查目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# 统计函数
statistics() {
    local dataset=$1
    local text_file="$DATA_DIR/$dataset/text"
    local label_file="$DATA_DIR/$dataset/text.label"
    local wav_scp_file="$DATA_DIR/$dataset/wav.scp"
    
    # 智能选择文件：优先使用 text.label（如果存在，说明运行了 stage 0）
    local target_file=""
    local file_type=""
    
    if [ -f "$label_file" ]; then
        target_file="$label_file"
        file_type="text.label (Stage -1 原始标签)"
    elif [ -f "$text_file" ]; then
        target_file="$text_file"
        file_type="text (Stage -1 标签)"
    else
        echo "警告: 文件不存在 $text_file 或 $label_file"
        return
    fi
    
    echo "----------------------------------------"
    echo "【$(echo $dataset | tr '[:lower:]' '[:upper:]')】数据集"
    echo "----------------------------------------"
    echo "使用文件: $file_type"
    echo ""
    
    # 统计总样本数（去除空格和换行）
    total=$(wc -l < "$target_file" 2>/dev/null)
    total=$(echo "$total" | tr -d ' \n\r')
    total=${total:-0}
    
    # 检测文本格式：标签格式 vs ASR 转录格式
    has_label_format=$(head -1 "$target_file" 2>/dev/null | grep -c "<" || echo "0")
    has_label_format=$(echo "$has_label_format" | tr -d ' \n\r')
    has_label_format=${has_label_format:-0}
    
    if [ "$has_label_format" -gt 0 ]; then
        # 标签格式：<HI_XIAOWEN>, <NIHAO_WENWEN>, <FILLER>
        hi_xiaowen=$(grep -c "<HI_XIAOWEN>" "$target_file" 2>/dev/null || echo "0")
        hi_xiaowen=$(echo "$hi_xiaowen" | tr -d ' \n\r')
        hi_xiaowen=${hi_xiaowen:-0}
        
        nihao_wenwen=$(grep -c "<NIHAO_WENWEN>" "$target_file" 2>/dev/null || echo "0")
        nihao_wenwen=$(echo "$nihao_wenwen" | tr -d ' \n\r')
        nihao_wenwen=${nihao_wenwen:-0}
        
        filler=$(grep -c "<FILLER>" "$target_file" 2>/dev/null || echo "0")
        filler=$(echo "$filler" | tr -d ' \n\r')
        filler=${filler:-0}
    else
        # ASR 转录格式：嗨 小 问, 你 好 问 问, 其他文本
        hi_xiaowen=$(grep -c "嗨 小 问$" "$target_file" 2>/dev/null || echo "0")
        hi_xiaowen=$(echo "$hi_xiaowen" | tr -d ' \n\r')
        hi_xiaowen=${hi_xiaowen:-0}
        
        nihao_wenwen=$(grep -c "你 好 问 问$" "$target_file" 2>/dev/null || echo "0")
        nihao_wenwen=$(echo "$nihao_wenwen" | tr -d ' \n\r')
        nihao_wenwen=${nihao_wenwen:-0}
        
        # 负样本 = 总数 - 正样本
        filler=$((total - hi_xiaowen - nihao_wenwen))
    fi
    
    # 计算正负样本
    positive=$((hi_xiaowen + nihao_wenwen))
    negative=$filler
    
    # 计算百分比
    if [ $total -gt 0 ]; then
        hi_xiaowen_pct=$(awk "BEGIN {printf \"%.2f\", $hi_xiaowen/$total*100}")
        nihao_wenwen_pct=$(awk "BEGIN {printf \"%.2f\", $nihao_wenwen/$total*100}")
        positive_pct=$(awk "BEGIN {printf \"%.2f\", $positive/$total*100}")
        negative_pct=$(awk "BEGIN {printf \"%.2f\", $negative/$total*100}")
    else
        hi_xiaowen_pct=0.00
        nihao_wenwen_pct=0.00
        positive_pct=0.00
        negative_pct=0.00
    fi
    
    # 打印统计结果
    printf "总样本数:        %8d\n" $total
    echo ""
    echo "【正样本】"
    printf "  <HI_XIAOWEN>   %8d (%6s%%)\n" $hi_xiaowen $hi_xiaowen_pct
    printf "  <NIHAO_WENWEN> %8d (%6s%%)\n" $nihao_wenwen $nihao_wenwen_pct
    printf "  小计:          %8d (%6s%%)\n" $positive $positive_pct
    echo ""
    echo "【负样本】"
    printf "  <FILLER>       %8d (%6s%%)\n" $negative $negative_pct
    echo ""
    
    # 正负样本比例
    if [ $positive -gt 0 ]; then
        ratio=$(awk "BEGIN {printf \"%.2f\", $negative/$positive}")
        echo "正负样本比例:    1 : $ratio"
    else
        echo "正负样本比例:    无正样本"
    fi
    
    # 检查 wav.scp 文件
    if [ -f "$wav_scp_file" ]; then
        wav_count=$(wc -l < "$wav_scp_file")
        if [ $wav_count -ne $total ]; then
            echo ""
            echo "警告: wav.scp 行数 ($wav_count) 与 text 行数 ($total) 不匹配！"
        fi
    fi
    
    echo ""
}

# 统计各个数据集
for dataset in train dev test; do
    statistics $dataset
done

echo "================================================================================"
echo "【汇总统计】"
echo "================================================================================"
echo ""

# 辅助函数：选择正确的文件
get_target_file() {
    local dataset=$1
    if [ -f "$DATA_DIR/$dataset/text.label" ]; then
        echo "$DATA_DIR/$dataset/text.label"
    elif [ -f "$DATA_DIR/$dataset/text" ]; then
        echo "$DATA_DIR/$dataset/text"
    else
        echo ""
    fi
}

# 汇总统计
train_file=$(get_target_file "train")
dev_file=$(get_target_file "dev")
test_file=$(get_target_file "test")

total_train=$([ -n "$train_file" ] && [ -f "$train_file" ] && wc -l < "$train_file" 2>/dev/null || echo "0")
total_train=$(echo "$total_train" | tr -d ' \n\r'); total_train=${total_train:-0}

total_dev=$([ -n "$dev_file" ] && [ -f "$dev_file" ] && wc -l < "$dev_file" 2>/dev/null || echo "0")
total_dev=$(echo "$total_dev" | tr -d ' \n\r'); total_dev=${total_dev:-0}

total_test=$([ -n "$test_file" ] && [ -f "$test_file" ] && wc -l < "$test_file" 2>/dev/null || echo "0")
total_test=$(echo "$total_test" | tr -d ' \n\r'); total_test=${total_test:-0}

total_all=$((total_train + total_dev + total_test))

# 检测文本格式
has_label_format=0
if [ -n "$train_file" ] && [ -f "$train_file" ]; then
    has_label_format=$(head -1 "$train_file" 2>/dev/null | grep -c "<" || echo "0")
    has_label_format=$(echo "$has_label_format" | tr -d ' \n\r')
    has_label_format=${has_label_format:-0}
fi

if [ "$has_label_format" -gt 0 ]; then
    # 标签格式
    pos_train=$([ -n "$train_file" ] && [ -f "$train_file" ] && (grep -c "<HI_XIAOWEN>\|<NIHAO_WENWEN>" "$train_file" 2>/dev/null || echo "0") || echo "0")
    pos_train=$(echo "$pos_train" | tr -d ' \n\r'); pos_train=${pos_train:-0}
    
    pos_dev=$([ -n "$dev_file" ] && [ -f "$dev_file" ] && (grep -c "<HI_XIAOWEN>\|<NIHAO_WENWEN>" "$dev_file" 2>/dev/null || echo "0") || echo "0")
    pos_dev=$(echo "$pos_dev" | tr -d ' \n\r'); pos_dev=${pos_dev:-0}
    
    pos_test=$([ -n "$test_file" ] && [ -f "$test_file" ] && (grep -c "<HI_XIAOWEN>\|<NIHAO_WENWEN>" "$test_file" 2>/dev/null || echo "0") || echo "0")
    pos_test=$(echo "$pos_test" | tr -d ' \n\r'); pos_test=${pos_test:-0}
    
    neg_train=$([ -n "$train_file" ] && [ -f "$train_file" ] && (grep -c "<FILLER>" "$train_file" 2>/dev/null || echo "0") || echo "0")
    neg_train=$(echo "$neg_train" | tr -d ' \n\r'); neg_train=${neg_train:-0}
    
    neg_dev=$([ -n "$dev_file" ] && [ -f "$dev_file" ] && (grep -c "<FILLER>" "$dev_file" 2>/dev/null || echo "0") || echo "0")
    neg_dev=$(echo "$neg_dev" | tr -d ' \n\r'); neg_dev=${neg_dev:-0}
    
    neg_test=$([ -n "$test_file" ] && [ -f "$test_file" ] && (grep -c "<FILLER>" "$test_file" 2>/dev/null || echo "0") || echo "0")
    neg_test=$(echo "$neg_test" | tr -d ' \n\r'); neg_test=${neg_test:-0}
else
    # ASR 转录格式
    hi_train=$([ -n "$train_file" ] && [ -f "$train_file" ] && (grep -c "嗨 小 问$" "$train_file" 2>/dev/null || echo "0") || echo "0")
    hi_train=$(echo "$hi_train" | tr -d ' \n\r'); hi_train=${hi_train:-0}
    nh_train=$([ -n "$train_file" ] && [ -f "$train_file" ] && (grep -c "你 好 问 问$" "$train_file" 2>/dev/null || echo "0") || echo "0")
    nh_train=$(echo "$nh_train" | tr -d ' \n\r'); nh_train=${nh_train:-0}
    pos_train=$((hi_train + nh_train))
    
    hi_dev=$([ -n "$dev_file" ] && [ -f "$dev_file" ] && (grep -c "嗨 小 问$" "$dev_file" 2>/dev/null || echo "0") || echo "0")
    hi_dev=$(echo "$hi_dev" | tr -d ' \n\r'); hi_dev=${hi_dev:-0}
    nh_dev=$([ -n "$dev_file" ] && [ -f "$dev_file" ] && (grep -c "你 好 问 问$" "$dev_file" 2>/dev/null || echo "0") || echo "0")
    nh_dev=$(echo "$nh_dev" | tr -d ' \n\r'); nh_dev=${nh_dev:-0}
    pos_dev=$((hi_dev + nh_dev))
    
    hi_test=$([ -n "$test_file" ] && [ -f "$test_file" ] && (grep -c "嗨 小 问$" "$test_file" 2>/dev/null || echo "0") || echo "0")
    hi_test=$(echo "$hi_test" | tr -d ' \n\r'); hi_test=${hi_test:-0}
    nh_test=$([ -n "$test_file" ] && [ -f "$test_file" ] && (grep -c "你 好 问 问$" "$test_file" 2>/dev/null || echo "0") || echo "0")
    nh_test=$(echo "$nh_test" | tr -d ' \n\r'); nh_test=${nh_test:-0}
    pos_test=$((hi_test + nh_test))
    
    neg_train=$((total_train - pos_train))
    neg_dev=$((total_dev - pos_dev))
    neg_test=$((total_test - pos_test))
fi

pos_all=$((pos_train + pos_dev + pos_test))
neg_all=$((neg_train + neg_dev + neg_test))

printf "数据集      总样本数    正样本    负样本    正样本比例\n"
printf "%-10s  %8d  %8d  %8d    %6.2f%%\n" \
    "Train" $total_train $pos_train $neg_train \
    $(awk "BEGIN {printf \"%.2f\", $pos_train/$total_train*100}" 2>/dev/null || echo "0.00")

printf "%-10s  %8d  %8d  %8d    %6.2f%%\n" \
    "Dev" $total_dev $pos_dev $neg_dev \
    $(awk "BEGIN {printf \"%.2f\", $pos_dev/$total_dev*100}" 2>/dev/null || echo "0.00")

printf "%-10s  %8d  %8d  %8d    %6.2f%%\n" \
    "Test" $total_test $pos_test $neg_test \
    $(awk "BEGIN {printf \"%.2f\", $pos_test/$total_test*100}" 2>/dev/null || echo "0.00")

echo "----------------------------------------"

printf "%-10s  %8d  %8d  %8d    %6.2f%%\n" \
    "总计" $total_all $pos_all $neg_all \
    $(awk "BEGIN {printf \"%.2f\", $pos_all/$total_all*100}" 2>/dev/null || echo "0.00")

echo ""
echo "================================================================================"
echo "统计完成！"
echo "================================================================================"
