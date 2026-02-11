#!/bin/bash
# Copyright 2024 Wayne
# 
# 用于评测关键词唤醒模型的脚本
# 用法示例:
#   bash evaluate.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/61.pt --dataset test --gpu 0
#   bash evaluate.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/61.pt --dataset dev --gpu "0,1,2,3"
#   bash evaluate.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/avg_30.pt --dataset test
#   bash evaluate.sh --checkpoint exp/.../229_int8.zip --jit_model true --dict_dir dict_top20

. ./path.sh

# 默认参数
checkpoint=""
executorch_model=""
executorch_seq_len=100
model_config=""
dataset="test"  # train, dev, test
gpu="0"
batch_size=256
num_workers=8
keywords="嗨小问,你好问问"  # 在实际调用时会转换为 Unicode 转义
dict_dir="dict"
jit_model=false
token_file="mobvoi_kws_transcription/tokens.txt"
lexicon_file="mobvoi_kws_transcription/lexicon.txt"
window_shift=50
sample_ratio=1.0
sample_seed=42

# 解析命令行参数
. tools/parse_options.sh || exit 1;

# 检查必需参数
if [ -z "$checkpoint" ] && [ -z "$executorch_model" ]; then
    echo "错误: 必须指定 --checkpoint 或 --executorch_model 参数"
    echo "用法: bash evaluate.sh [--checkpoint <checkpoint_path>] [--executorch_model <model.pte>] [--dataset train|dev|test] [--gpu <gpu_ids>]"
    echo ""
    echo "示例:"
    echo "  bash evaluate.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/61.pt --dataset test --gpu 0"
    echo "  bash evaluate.sh --checkpoint exp/fsmn_ctc_baseline_4gpus/avg_30.pt --dataset dev --gpu \"0,1,2,3\""
    echo "  bash evaluate.sh --executorch_model exp/.../229_executorch_fp32.pte --checkpoint exp/.../229.pt --dict_dir dict_top20"
    exit 1
fi

# 检查模型文件是否存在
if [ -n "$checkpoint" ] && [ ! -f "$checkpoint" ]; then
    echo "错误: checkpoint 文件不存在: $checkpoint"
    exit 1
fi
if [ -n "$executorch_model" ] && [ ! -f "$executorch_model" ]; then
    echo "错误: executorch_model 文件不存在: $executorch_model"
    exit 1
fi
# 从模型路径推导 config 和输出目录
primary_model="$checkpoint"
if [ -n "$executorch_model" ]; then
    primary_model="$executorch_model"
fi
model_dir=$(dirname "$primary_model")
model_filename=$(basename "$primary_model")
model_basename="${model_filename%.*}"

if [ -n "$model_config" ]; then
    config_file="$model_config"
else
    config_file="$model_dir/config.yaml"
fi
result_dir="$model_dir/${dataset}_${model_basename}"

# 检查 config 文件是否存在
if [ ! -f "$config_file" ]; then
    echo "错误: config 文件不存在: $config_file"
    exit 1
fi

# 设置数据文件
data_file="data/${dataset}/data.list"
if [ ! -f "$data_file" ]; then
    echo "错误: 数据文件不存在: $data_file"
    exit 1
fi
if ! [[ "$sample_ratio" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "错误: sample_ratio 必须是数字，当前: $sample_ratio"
    exit 1
fi
if ! awk -v r="$sample_ratio" 'BEGIN { exit !(r>0 && r<=1) }'; then
    echo "错误: sample_ratio 必须在 (0, 1] 范围内，当前: $sample_ratio"
    exit 1
fi

# 检查 token 和 lexicon 文件
if [ ! -f "$token_file" ]; then
    echo "错误: token 文件不存在: $token_file"
    exit 1
fi
if [ ! -f "$lexicon_file" ]; then
    echo "错误: lexicon 文件不存在: $lexicon_file"
    exit 1
fi

# 创建结果目录
mkdir -p "$result_dir"

# 可选随机抽样子集用于快速验证
tmp_eval_data=""
if ! awk -v r="$sample_ratio" 'BEGIN { exit !(r<1) }'; then
    :
else
    total_lines=$(awk 'NF>0{c++} END{print c+0}' "$data_file")
    sample_lines=$(awk -v c="$total_lines" -v r="$sample_ratio" 'BEGIN {
        n = int(c * r + 0.999999);
        if (n < 1) n = 1;
        if (n > c) n = c;
        print n
    }')
    tmp_eval_data="$result_dir/.sample_${dataset}_$(date +%s)_$$.list"
    shuf --random-source=<(yes "$sample_seed") -n "$sample_lines" "$data_file" > "$tmp_eval_data"
    data_file="$tmp_eval_data"
fi

cleanup() {
    if [ -n "$tmp_eval_data" ] && [ -f "$tmp_eval_data" ]; then
        rm -f "$tmp_eval_data"
    fi
}
trap cleanup EXIT

# 输出配置信息
echo "================================================"
echo "🎯 评测配置"
echo "================================================"
echo "模型 checkpoint: $checkpoint"
echo "ExecuTorch模型:   $executorch_model"
echo "模型 config:     $config_file"
echo "评测数据集:      $dataset ($data_file)"
echo "采样比例:        $sample_ratio"
echo "GPU:             $gpu"
echo "Batch size:      $batch_size"
echo "关键词:          $keywords"
echo "结果目录:        $result_dir"
echo "================================================"
echo ""

# 选择第一个 GPU 用于推理（多GPU时只用第一个）
first_gpu=$(echo $gpu | awk -F',' '{print $1}')

if [ "$jit_model" = "true" ]; then
    echo "JIT 模式:        已启用（TorchScript 模型，CPU 推理）"
fi
if [ -n "$executorch_model" ]; then
    echo "ExecuTorch 模式: 已启用（CPU 推理）"
fi

# Step 1: 运行推理，生成 score 文件
score_file="$result_dir/score.txt"
echo "🚀 Step 1: 运行推理，生成检测结果..."
echo "输出文件: $score_file"
echo ""

score_cmd=(python wekws/bin/score_ctc.py
    --config "$config_file"
    --test_data "$data_file"
    --gpu "$first_gpu"
    --batch_size "$batch_size"
    --dict "$dict_dir"
    --score_file "$score_file"
    --num_workers "$num_workers"
    --keywords "\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee"
    --token_file "$token_file"
    --lexicon_file "$lexicon_file")

if [ -n "$checkpoint" ]; then
    score_cmd+=(--checkpoint "$checkpoint")
fi
if [ "$jit_model" = "true" ]; then
    score_cmd+=(--jit_model)
fi
if [ -n "$executorch_model" ]; then
    score_cmd+=(--executorch_model "$executorch_model")
    score_cmd+=(--executorch_seq_len "$executorch_seq_len")
fi

echo "score_ctc Python: python"
"${score_cmd[@]}"

if [ $? -ne 0 ]; then
    echo "❌ 推理失败！"
    exit 1
fi

echo ""
echo "✅ Step 1 完成！"
echo ""

# Step 2: 计算 DET 曲线和评估指标
echo "🚀 Step 2: 计算 DET 曲线和评估指标..."
echo ""

python wekws/bin/compute_det_ctc.py \
    --keywords "\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee" \
    --test_data "$data_file" \
    --window_shift "$window_shift" \
    --step 0.001 \
    --score_file "$score_file" \
    --dict "$dict_dir" \
    --token_file "$token_file" \
    --lexicon_file "$lexicon_file"

if [ $? -ne 0 ]; then
    echo "❌ DET 计算失败！"
    exit 1
fi

echo ""
echo "================================================"
echo "✅ 评测完成！"
echo "================================================"
echo "结果保存在: $result_dir"
echo "  - score.txt:        检测结果和置信度"
echo "  - det_*.png:        DET 曲线图"
echo "  - 控制台输出:       召回率和误唤醒率统计"
echo "================================================"
