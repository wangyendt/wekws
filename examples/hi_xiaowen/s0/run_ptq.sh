#!/bin/bash
# Copyright 2026 Wayne
#
# PTQ (Post-Training Quantization) 量化脚本
#
# 支持 INT8（PyTorch 静态量化）和 INT16（模拟量化）两种模式。
# INT8 量化产出 TorchScript 模型（.zip），评估时需 --jit_model 标志。
# INT16 模拟量化产出标准 .pt checkpoint，可直接用 evaluate.sh 评估。
#
# 用法示例（选项参数必须放在位置参数之前）:
#
#   # INT8 量化 + 评估
#   bash run_ptq.sh --checkpoint exp/fsmn_ctc_distill_mini_align_20_test2/229.pt \
#       --quant_type int8 --dict_dir dict_top20 1 2
#
#   # INT16 模拟量化 + 评估
#   bash run_ptq.sh --checkpoint exp/fsmn_ctc_distill_mini_align_20_test2/229.pt \
#       --quant_type int16 --dict_dir dict_top20 1 2
#
#   # 仅量化（不评估）
#   bash run_ptq.sh --checkpoint exp/fsmn_ctc_distill_mini_align_20_test2/229.pt \
#       --quant_type int8 --dict_dir dict_top20 1 1
#
#   # 仅评估（已有量化模型）
#   bash run_ptq.sh --checkpoint exp/fsmn_ctc_distill_mini_align_20_test2/229.pt \
#       --quant_type int8 --dict_dir dict_top20 2 2
#
#   # 自定义校准参数
#   bash run_ptq.sh --checkpoint exp/.../229.pt --quant_type int8 \
#       --num_calib 500 --calib_data data/dev/data.list --dict_dir dict_top20 1 2
#
# 日志文件自动保存到: <checkpoint_dir>/logs/run_ptq_<quant_type>_<timestamp>.log

. ./path.sh

# 过滤 torchaudio 弃用警告
export PYTHONWARNINGS="ignore::UserWarning"

# 保存原始参数用于日志
original_args="$@"

stage=1
stop_stage=2

# ---- 模型参数 ----
checkpoint=""
dict_dir="dict_top20"

# ---- 量化参数 ----
quant_type="int8"       # int8 or int16
num_calib=200           # 校准样本数
calib_data="data/train/data.list"
batch_size=1
num_workers=4
seed=42

# ---- 评估参数 ----
gpu="0"
eval_batch_size=256
eval_num_workers=8
dataset="test"

. tools/parse_options.sh || exit 1;

# parse_options.sh 处理完选项后，剩余的是位置参数
if [ $# -ge 1 ]; then
  stage=$1
fi
if [ $# -ge 2 ]; then
  stop_stage=$2
fi

# 检查必需参数
if [ -z "$checkpoint" ]; then
    echo "错误: 必须指定 --checkpoint 参数"
    echo "用法: bash run_ptq.sh --checkpoint <path> --quant_type int8|int16 [options] <stage> <stop_stage>"
    echo ""
    echo "示例:"
    echo "  bash run_ptq.sh --checkpoint exp/.../229.pt --quant_type int8 --dict_dir dict_top20 1 2"
    echo "  bash run_ptq.sh --checkpoint exp/.../229.pt --quant_type int16 --dict_dir dict_top20 1 2"
    exit 1
fi

if [ ! -f "$checkpoint" ]; then
    echo "错误: checkpoint 文件不存在: $checkpoint"
    exit 1
fi

# 推导路径
checkpoint_dir=$(dirname "$checkpoint")
checkpoint_basename=$(basename "$checkpoint" .pt)
config_file="$checkpoint_dir/config.yaml"

if [ ! -f "$config_file" ]; then
    echo "错误: config 文件不存在: $config_file"
    exit 1
fi

# 量化后的模型路径
if [ "$quant_type" = "int8" ]; then
    # INT8: TorchScript 格式，.zip 后缀（去掉原 .pt 后缀的部分需要特殊处理）
    quantized_model="$checkpoint_dir/${checkpoint_basename}_int8.zip"
else
    # INT16: 标准 .pt 格式
    quantized_model="$checkpoint_dir/${checkpoint_basename}_int16_sim.pt"
fi

# 创建日志目录和日志文件
log_dir="$checkpoint_dir/logs"
mkdir -p "$log_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/run_ptq_${quant_type}_${timestamp}.log"

# 如果还没有重定向到 tee（避免递归）
if [ -z "$LOG_REDIRECT_DONE" ]; then
  echo "实验目录: $checkpoint_dir"
  echo "日志文件: $log_file"
  echo "================================================"
  export LOG_REDIRECT_DONE=1
  exec > >(tee -a "$log_file") 2>&1
  echo "================================================"
  echo "开始运行: $(date)"
  echo "   命令: bash $0 $original_args"
  echo "   Stage: $stage -> $stop_stage"
  echo "   量化类型: $quant_type"
  echo "   源 checkpoint: $checkpoint"
  echo "   量化模型输出: $quantized_model"
  echo "   日志文件: $log_file"
  echo "================================================"
fi

stage_int=$(echo "$stage" | awk '{print int($1)}')
stop_stage_int=$(echo "$stop_stage" | awk '{print int($1)}')


# ================================================================
# Stage 1: PTQ 量化
# ================================================================
if [ ${stage_int} -le 1 ] && [ ${stop_stage_int} -ge 1 ]; then
  echo ""
  echo "================================================"
  echo "Stage 1: PTQ 量化 ($quant_type)"
  echo "================================================"
  echo "源 checkpoint:  $checkpoint"
  echo "模型 config:    $config_file"
  echo "量化类型:       $quant_type"
  echo "校准数据:       $calib_data"
  echo "校准样本数:     $num_calib"
  echo "词表目录:       $dict_dir"
  echo "输出模型:       $quantized_model"
  echo "================================================"
  echo ""

  if [ ! -f "$calib_data" ]; then
    echo "错误: 校准数据文件不存在: $calib_data"
    exit 1
  fi

  python wekws/bin/ptq_quantize.py \
    --config "$config_file" \
    --checkpoint "$checkpoint" \
    --calib_data "$calib_data" \
    --num_calib $num_calib \
    --quant_type "$quant_type" \
    --output_dir "$checkpoint_dir" \
    --dict "$dict_dir" \
    --num_workers $num_workers \
    --batch_size $batch_size \
    --seed $seed

  if [ $? -ne 0 ]; then
    echo "PTQ 量化失败！"
    exit 1
  fi

  echo ""
  echo "Stage 1 完成！量化模型: $quantized_model"
fi


# ================================================================
# Stage 2: 评估量化模型
# ================================================================
if [ ${stage_int} -le 2 ] && [ ${stop_stage_int} -ge 2 ]; then
  echo ""
  echo "================================================"
  echo "Stage 2: 评估量化模型"
  echo "================================================"
  echo ""

  if [ ! -f "$quantized_model" ]; then
    echo "错误: 量化模型不存在: $quantized_model"
    echo "请先运行 Stage 1 进行量化"
    exit 1
  fi

  if [ "$quant_type" = "int8" ]; then
    # INT8: TorchScript 模型，需要 --jit_model 标志
    echo "评估 INT8 TorchScript 模型（CPU 推理）..."
    echo "  模型: $quantized_model"
    echo ""
    bash evaluate.sh \
      --checkpoint "$quantized_model" \
      --jit_model true \
      --dict_dir "$dict_dir" \
      --dataset "$dataset" \
      --gpu "$gpu" \
      --batch_size "$eval_batch_size" \
      --num_workers "$eval_num_workers"
  else
    # INT16: 标准 .pt checkpoint，直接用 evaluate.sh
    echo "评估 INT16 模拟量化模型..."
    echo "  模型: $quantized_model"
    echo ""
    bash evaluate.sh \
      --checkpoint "$quantized_model" \
      --dict_dir "$dict_dir" \
      --dataset "$dataset" \
      --gpu "$gpu" \
      --batch_size "$eval_batch_size" \
      --num_workers "$eval_num_workers"
  fi

  if [ $? -ne 0 ]; then
    echo "评估失败！"
    exit 1
  fi

  echo ""
  echo "Stage 2 评估完成！"
fi


# 脚本结束日志
if [ -n "$LOG_REDIRECT_DONE" ]; then
  echo ""
  echo "================================================"
  echo "运行完成: $(date)"
  echo "   量化类型: $quant_type"
  echo "   源 checkpoint: $checkpoint"
  echo "   量化模型: $quantized_model"
  echo "   日志文件: $log_file"
  echo "================================================"
fi
