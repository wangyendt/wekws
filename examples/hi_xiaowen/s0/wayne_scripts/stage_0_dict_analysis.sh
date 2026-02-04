#!/bin/bash
# Stage 0 字典变化分析脚本
# 详细分析 dict.txt 和 words.txt 的变化及其原因

echo "================================================================================"
echo "Stage 0 字典文件详细分析"
echo "================================================================================"
echo ""

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."
DICT_DIR="$BASE_DIR/dict"

echo "【1】字典文件变化对比"
echo "================================================================================"
echo ""

echo "Stage -1 (初始状态):"
echo "----------------------------------------"
echo "dict.txt: 3 个关键词标签"
echo "  <FILLER> -1      # 负样本（非关键词，填充词）"
echo "  <HI_XIAOWEN> 0   # 正样本（嗨小问）"
echo "  <NIHAO_WENWEN> 1 # 正样本（你好问问）"
echo ""
echo "words.txt: 3 个关键词"
echo "  <FILLER>"
echo "  <HI_XIAOWEN>"
echo "  <NIHAO_WENWEN>"
echo ""
echo "目的: 简单的关键词分类任务"
echo ""

echo "Stage 0 (增强状态):"
echo "----------------------------------------"
if [ -f "$DICT_DIR/dict.txt" ]; then
    dict_lines=$(wc -l < "$DICT_DIR/dict.txt")
    echo "dict.txt: $dict_lines 个 token (字符/音素级别)"
else
    echo "dict.txt: 文件不存在"
    dict_lines=0
fi

if [ -f "$DICT_DIR/words.txt" ]; then
    words_lines=$(wc -l < "$DICT_DIR/words.txt")
    echo "words.txt: $words_lines 个特殊标记"
    echo "  $(cat "$DICT_DIR/words.txt" | tr '\n' ' ')"
else
    echo "words.txt: 文件不存在"
fi
echo ""
echo "目的: 字符级 CTC 训练，支持任意文本识别"
echo ""

echo "【2】为什么要这样变化？"
echo "================================================================================"
echo ""
echo "原因 1: 提高模型泛化能力"
echo "  - Stage -1: 只能识别 3 个固定标签（关键词分类）"
echo "  - Stage 0:  可以识别任意字符组合（语音识别）"
echo "  - 好处:     模型学习更丰富的语音特征，不只是简单的模板匹配"
echo ""
echo "原因 2: 利用 ASR 转录数据"
echo "  - 负样本有了实际转录内容（如：'今 天 天 气 怎 么 样'）"
echo "  - 正样本也变成字符序列（'嗨 小 问' 而非 <HI_XIAOWEN>）"
echo "  - 模型可以学习字符级别的声学特征"
echo ""
echo "原因 3: 支持 CTC 训练"
echo "  - CTC (Connectionist Temporal Classification) 需要字符级别的标签"
echo "  - 可以处理不等长的输入输出对齐问题"
echo "  - 更适合端到端的语音识别任务"
echo ""

echo "【3】特殊标记详解"
echo "================================================================================"
echo ""

if [ -f "$DICT_DIR/dict.txt" ]; then
    echo "从 dict.txt 提取特殊标记:"
    echo ""
    
    # 提取特殊标记
    sil_info=$(grep "^sil " "$DICT_DIR/dict.txt" || echo "未找到")
    eps_info=$(grep "^<eps> " "$DICT_DIR/dict.txt" || echo "未找到")
    blk_info=$(grep "^<blk> " "$DICT_DIR/dict.txt" || echo "未找到")
    
    echo "1. sil (silence)"
    echo "   含义: 静音标记"
    echo "   值:   $sil_info"
    echo "   用途: 表示音频中的静音段"
    echo "   注意: 在 CTC 训练中，静音通常映射到 blank (blk)"
    echo ""
    
    echo "2. <eps> (epsilon)"
    echo "   含义: 空标记/空转移"
    echo "   值:   $eps_info"
    echo "   用途: FST (有限状态转换器) 中的 epsilon 转移"
    echo "   说明: 不消耗输入或输出，用于状态转移"
    echo ""
    
    echo "3. <blk> (blank)"
    echo "   含义: CTC blank 标记"
    echo "   值:   $blk_info"
    echo "   用途: CTC 算法中的空白标记，表示'无输出'"
    echo "   说明: 允许相同字符重复，解决输入输出对齐问题"
    echo ""
    
    echo "【为什么 sil 和 <blk> 都是 0？】"
    echo "----------------------------------------"
    echo ""
    echo "这是 CTC 训练的标准设置:"
    echo ""
    echo "  在 CTC 中:"
    echo "    - blank (blk) 通常是 ID 0"
    echo "    - 这是 CTC 算法的约定"
    echo "    - PyTorch/TensorFlow 的 CTC loss 默认 blank=0"
    echo ""
    echo "  sil 和 blk 都映射到 0 的原因:"
    echo "    - 静音 (sil) 在语音中不产生有意义的输出"
    echo "    - 将其映射到 blank，表示'这段时间无文字输出'"
    echo "    - 简化模型：静音=无输出=blank"
    echo ""
    echo "  <eps> 是 -1 的原因:"
    echo "    - 与 blank 区分开"
    echo "    - -1 在很多框架中表示'忽略/不参与计算'"
    echo "    - 用于 WFST 解码，不参与 CTC 训练"
    echo ""
fi

echo "【4】字符类型统计"
echo "================================================================================"
echo ""

if [ -f "$DICT_DIR/dict.txt" ]; then
    echo "正在分析 dict.txt 的字符组成..."
    echo ""
    
    # 创建临时 Python 脚本进行精确统计
    python3 << 'PYTHON_SCRIPT'
import re
import sys

# 读取 dict.txt
dict_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/wayne/Documents/work/code/project/ffalcon/kws/wekws/examples/hi_xiaowen/s0/dict/dict.txt"

try:
    with open(dict_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
except:
    print("无法读取文件")
    sys.exit(1)

# 统计各类字符
english_chars = []      # 英文字符
chinese_chars = []      # 中文字符
special_symbols = []    # 特殊符号
special_markers = []    # 特殊标记（<xxx>）

for line in lines:
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    
    token = parts[0]
    token_id = parts[1]
    
    # 跳过特殊标记
    if token.startswith('<') and token.endswith('>'):
        special_markers.append((token, token_id))
        continue
    
    # 判断字符类型
    if token in ['sil']:
        special_markers.append((token, token_id))
    elif re.match(r'^[a-zA-Z]+$', token):
        # 纯英文
        english_chars.append((token, token_id))
    elif re.match(r'^[\u4e00-\u9fa5]+$', token):
        # 纯中文
        chinese_chars.append((token, token_id))
    elif re.match(r"^['\-]", token) or token == '&':
        # 以标点开头的（如 'll, 're, &）
        special_symbols.append((token, token_id))
    else:
        # 其他混合或特殊字符
        # 判断主要成分
        has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', token))
        has_english = bool(re.search(r'[a-zA-Z]', token))
        
        if has_chinese:
            chinese_chars.append((token, token_id))
        elif has_english:
            english_chars.append((token, token_id))
        else:
            special_symbols.append((token, token_id))

# 输出统计
print(f"【统计结果】")
print(f"{'='*60}")
print(f"")
print(f"1. 特殊标记 (Special Markers)")
print(f"   数量: {len(special_markers)}")
print(f"   示例: {', '.join([t[0] for t in special_markers[:10]])}")
print(f"")
print(f"2. 英文字符/词 (English)")
print(f"   数量: {len(english_chars)}")
print(f"   示例: {', '.join([t[0] for t in english_chars[:15]])}")
print(f"")
print(f"3. 中文字符 (Chinese)")
print(f"   数量: {len(chinese_chars)}")
print(f"   示例: {', '.join([t[0] for t in chinese_chars[:15]])}")
print(f"")
print(f"4. 特殊符号 (Special Symbols)")
print(f"   数量: {len(special_symbols)}")
print(f"   示例: {', '.join([t[0] for t in special_symbols[:15]])}")
print(f"")
print(f"{'='*60}")
print(f"")

# 汇总
basic_chars_sum = len(special_markers) + len(english_chars) + len(chinese_chars)
total_all = len(special_markers) + len(english_chars) + len(chinese_chars) + len(special_symbols)

print(f"【汇总】")
print(f"{'-'*60}")
print(f"")
print(f"  特殊标记:        {len(special_markers):6d} 个")
print(f"  英文字符/词:     {len(english_chars):6d} 个")
print(f"  中文字符:        {len(chinese_chars):6d} 个")
print(f"  {'-'*40}")
print(f"  基本字符小计:    {basic_chars_sum:6d} 个 (特殊标记+英+中)")
print(f"")
print(f"  特殊符号:        {len(special_symbols):6d} 个")
print(f"  {'-'*40}")
print(f"  总计:            {total_all:6d} 个")
print(f"")

# 字符类型分布
if total_all > 0:
    print(f"【分布比例】")
    print(f"{'-'*60}")
    print(f"")
    print(f"  特殊标记:     {len(special_markers)/total_all*100:6.2f}%")
    print(f"  英文字符:     {len(english_chars)/total_all*100:6.2f}%")
    print(f"  中文字符:     {len(chinese_chars)/total_all*100:6.2f}%")
    print(f"  特殊符号:     {len(special_symbols)/total_all*100:6.2f}%")
    print(f"")

# 输出一些有趣的统计
print(f"【详细分析】")
print(f"{'-'*60}")
print(f"")

# 英文字符长度分布
if english_chars:
    lengths = [len(t[0]) for t in english_chars]
    avg_len = sum(lengths) / len(lengths)
    print(f"  英文词平均长度: {avg_len:.2f} 字符")
    print(f"  最长英文词: {max(english_chars, key=lambda x: len(x[0]))[0]} ({len(max(english_chars, key=lambda x: len(x[0]))[0])} 字符)")

# 中文是单字还是词
if chinese_chars:
    single_chars = [t for t in chinese_chars if len(t[0]) == 1]
    multi_chars = [t for t in chinese_chars if len(t[0]) > 1]
    print(f"  中文单字: {len(single_chars)} 个")
    print(f"  中文词组: {len(multi_chars)} 个")
    if multi_chars:
        print(f"  中文词组示例: {', '.join([t[0] for t in multi_chars[:10]])}")

print(f"")

PYTHON_SCRIPT

else
    echo "dict.txt 文件不存在，无法统计"
fi

echo ""
echo "【5】实际应用场景】"
echo "================================================================================"
echo ""
echo "这种字典设计的优势:"
echo ""
echo "  1. 开放词表 (Open Vocabulary)"
echo "     - 可以识别训练时未见过的词"
echo "     - 通过字符组合生成新词"
echo ""
echo "  2. 端到端训练"
echo "     - 直接从音频到文本"
echo "     - 不需要额外的语言模型"
echo ""
echo "  3. 多语言支持"
echo "     - 同时支持中英文"
echo "     - 可扩展到其他语言"
echo ""
echo "  4. 鲁棒性"
echo "     - 对拼写错误、口语化表达更宽容"
echo "     - 可以处理 OOV (Out-of-Vocabulary) 问题"
echo ""

echo "【6】与 Stage -1 的对比】"
echo "================================================================================"
echo ""
printf "%-20s %-30s %-30s\n" "特性" "Stage -1" "Stage 0"
printf "%-20s %-30s %-30s\n" "$(printf '%.0s-' {1..20})" "$(printf '%.0s-' {1..30})" "$(printf '%.0s-' {1..30})"
printf "%-20s %-30s %-30s\n" "词表大小" "3 个关键词" "$dict_lines 个 token"
printf "%-20s %-30s %-30s\n" "训练方式" "关键词分类" "字符级 CTC"
printf "%-20s %-30s %-30s\n" "输出类型" "类别标签" "文本序列"
printf "%-20s %-30s %-30s\n" "泛化能力" "固定关键词" "任意文本"
printf "%-20s %-30s %-30s\n" "训练难度" "简单" "复杂"
printf "%-20s %-30s %-30s\n" "识别精度" "高 (关键词)" "中 (通用识别)"
printf "%-20s %-30s %-30s\n" "应用场景" "唤醒词检测" "语音识别+唤醒"
echo ""

echo "【7】常见疑问解答 (FAQ)"
echo "================================================================================"
echo ""

echo "Q1: dict.txt 有 2787 行，但 run_fsmn_ctc.sh 中 num_keywords=2599，为什么不一致？"
echo "----------------------------------------"
echo ""
echo "  答: 这是正常的版本差异，有两种可能："
echo ""
echo "  情况 1: 字典版本不同"
echo "    - dict.txt (2787行): 来自 mobvoi_kws_transcription 的完整字典"
echo "    - num_keywords=2599: 可能是旧版本或过滤后的字典"
echo "    - 训练时会根据实际使用的 token 自动调整"
echo ""
echo "  情况 2: 实际使用的 token 数"
if [ -f "$DICT_DIR/dict.txt" ]; then
    max_id=$(awk '{print $2}' "$DICT_DIR/dict.txt" | sort -n | tail -1)
    echo "    - dict.txt 最大 token ID: $max_id"
    echo "    - 有效 token 数: $(($max_id + 2)) 个 (包含 ID -1 到 $max_id)"
fi
echo ""
echo "  验证方法:"
echo "    # 查看实际使用的词表大小"
echo "    grep -E 'output_dim|vocab_size' conf/fsmn_ctc.yaml"
echo "    grep 'num_keywords' run_fsmn_ctc.sh"
echo ""
echo "  结论: 模型会根据实际数据自动调整，不影响训练"
echo ""

echo "Q2: 为什么在 data/train/text 中搜索不到 sil、<eps>、<blk>？"
echo "----------------------------------------"
echo ""
echo "  答: 这是完全正常的！这些特殊标记在训练时自动处理，不需要出现在数据中。"
echo ""
echo "  原因分析:"
echo ""
echo "  1. sil (静音)"
echo "     - 静音由音频特征自动识别"
echo "     - 不需要在文本中标注"
echo "     - CTC 训练时自动映射到 blank"
echo ""
echo "  2. <eps> (epsilon)"
echo "     - 用于 WFST 解码的内部标记"
echo "     - 不参与 CTC 训练"
echo "     - 只在解码图中使用"
echo ""
echo "  3. <blk> (CTC blank)"
echo "     - CTC 算法自动插入"
echo "     - 用于对齐输入输出序列"
echo "     - 训练时由 CTC loss 自动处理"
echo ""
echo "  数据格式对比:"
echo ""
echo "  训练数据 (data/train/text):"
echo "    68c08ef7... 嗨 小 问         ← 只有实际文本"
echo "    461003fa... if you don't...  ← 只有实际文本"
echo ""
echo "  CTC 训练时的内部表示:"
echo "    [blk] 嗨 [blk] 小 [blk] 问 [blk]  ← blank 自动插入"
echo ""
echo "  验证方法:"
echo "    # 查看训练数据（应该找不到特殊标记）"
echo "    grep -E 'sil|eps|blk' data/train/text"
echo ""
echo "    # 查看字典定义（应该能找到）"
echo "    grep -E 'sil|eps|blk' dict/dict.txt"
echo ""
echo "  结论: 特殊标记只在 dict.txt 中定义，不出现在训练数据中"
echo ""

echo "Q3: 字典中的 token ID 分布是怎样的？"
echo "----------------------------------------"
echo ""
if [ -f "$DICT_DIR/dict.txt" ]; then
    echo "  Token ID 范围分析:"
    echo ""
    min_id=$(awk '{print $2}' "$DICT_DIR/dict.txt" | sort -n | head -1)
    max_id=$(awk '{print $2}' "$DICT_DIR/dict.txt" | sort -n | tail -1)
    
    echo "    最小 ID: $min_id (<eps>)"
    echo "    最大 ID: $max_id"
    echo "    ID 范围: $min_id ~ $max_id"
    echo ""
    echo "  特殊 ID 说明:"
    echo "    -1: <eps> (epsilon 转移)"
    echo "     0: sil + <blk> (静音和 CTC blank 共用)"
    echo "     1~$max_id: 实际字符/词"
    echo ""
fi

echo "================================================================================"
echo "分析完成！"
echo "================================================================================"
echo ""
echo "建议:"
echo "  - 如果只需要唤醒词检测，Stage -1 已经足够"
echo "  - 如果需要更强的泛化能力和抗干扰能力，使用 Stage 0"
echo "  - Stage 0 的训练时间和计算资源需求更高"
echo ""
echo "进一步探索:"
echo "  - 查看模型配置: cat conf/fsmn_ctc.yaml"
echo "  - 查看训练脚本: cat run_fsmn_ctc.sh | grep -A 5 'stage.*1'"
echo "  - 查看字典详情: head -20 dict/dict.txt && tail -20 dict/dict.txt"
echo ""
