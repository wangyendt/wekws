#!/bin/bash
# Copyright 2026 Wayne
#
# ExecuTorch PT2E 导出脚本
#
# 用法示例:
#   # 导出 FP32 ExecuTorch 模型
#   bash run_executorch_pt2e.sh --checkpoint exp/.../229.pt --quant_type fp32 --dict_dir dict_top20 1 1
#
#   # 导出 INT8 ExecuTorch 模型（PT2E 校准）
#   bash run_executorch_pt2e.sh --checkpoint exp/.../229.pt --quant_type int8 --dict_dir dict_top20 1 1
#
#   # 导出后直接评估（Stage 2）
#   bash run_executorch_pt2e.sh --checkpoint exp/.../229.pt --quant_type fp32 --dict_dir dict_top20 1 2

. ./path.sh

stage=1
stop_stage=2

checkpoint=""
dict_dir="dict_top20"
quant_type="fp32"      # fp32 or int8
num_calib=200
calib_data="data/train/data.list"
batch_size=1
num_workers=4
seed=42
export_seq_len=100
dataset="test"
gpu="0"
eval_batch_size=256
eval_num_workers=8
output_model=""

. tools/parse_options.sh || exit 1

if [ $# -ge 1 ]; then
  stage=$1
fi
if [ $# -ge 2 ]; then
  stop_stage=$2
fi

if [ -z "$checkpoint" ]; then
  echo "错误: 必须指定 --checkpoint 参数"
  exit 1
fi
if [ ! -f "$checkpoint" ]; then
  echo "错误: checkpoint 文件不存在: $checkpoint"
  exit 1
fi

checkpoint_dir=$(dirname "$checkpoint")
checkpoint_basename=$(basename "$checkpoint" .pt)
config_file="$checkpoint_dir/config.yaml"

if [ ! -f "$config_file" ]; then
  echo "错误: config 文件不存在: $config_file"
  exit 1
fi

if [ -z "$output_model" ]; then
  output_model="$checkpoint_dir/${checkpoint_basename}_executorch_${quant_type}.pte"
fi

stage_int=$(echo "$stage" | awk '{print int($1)}')
stop_stage_int=$(echo "$stop_stage" | awk '{print int($1)}')

if [ ${stage_int} -le 1 ] && [ ${stop_stage_int} -ge 1 ]; then
  echo "================================================"
  echo "Stage 1: 导出 ExecuTorch 模型 ($quant_type)"
  echo "================================================"
  echo "checkpoint: $checkpoint"
  echo "config:     $config_file"
  echo "输出模型:    $output_model"
  echo "Python运行器: python"
  echo "================================================"

  python wekws/bin/export_executorch_pt2e.py \
    --config "$config_file" \
    --checkpoint "$checkpoint" \
    --output_model "$output_model" \
    --quant_type "$quant_type" \
    --dict "$dict_dir" \
    --calib_data "$calib_data" \
    --num_calib "$num_calib" \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --seed "$seed" \
    --export_seq_len "$export_seq_len"

  if [ $? -ne 0 ]; then
    echo "错误: ExecuTorch 导出失败"
    exit 1
  fi
fi

if [ ${stage_int} -le 2 ] && [ ${stop_stage_int} -ge 2 ]; then
  echo "================================================"
  echo "Stage 2: 评估 ExecuTorch 模型"
  echo "================================================"
  if [ ! -f "$output_model" ]; then
    echo "错误: 模型不存在: $output_model"
    exit 1
  fi

  eval_cmd=(bash evaluate.sh
    --checkpoint "$checkpoint"
    --executorch_model "$output_model"
    --dict_dir "$dict_dir"
    --dataset "$dataset"
    --gpu "$gpu"
    --batch_size "$eval_batch_size"
    --num_workers "$eval_num_workers")
  "${eval_cmd[@]}"

  if [ $? -ne 0 ]; then
    echo "错误: ExecuTorch 评估失败"
    exit 1
  fi
fi

echo "================================================"
echo "完成！"
echo "ExecuTorch 模型: $output_model"
echo "================================================"
