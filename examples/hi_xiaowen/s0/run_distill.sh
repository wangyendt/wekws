#!/bin/bash
# Copyright 2026 Wayne
#
# Feature Alignment 蒸馏训练脚本：用教师模型的 block 特征指导学生 backbone 学习，
# 并复用教师的 HEAD (out_linear1 + out_linear2)。
#
# 两阶段训练：
#   Phase 1 (align_epochs):    纯 MSE feature alignment，HEAD 冻结
#   Phase 2 (finetune_epochs): MSE + CTC 混合，HEAD 解冻（极低学习率）
#
# 用法示例（选项参数必须放在位置参数之前）:
#
#   # 使用 top20 教师蒸馏（默认）
#   bash run_distill.sh 2 3
#
#   # 使用 top440 教师蒸馏
#   bash run_distill.sh \
#     --teacher_checkpoint exp/fsmn_ctc_top440_weight_surgery/79.pt \
#     --num_keywords 440 \
#     --dict_dir dict_top440 \
#     --target_exp_dir exp/fsmn_ctc_distill_mini_align_440 \
#     2 3
#
#   # 使用 baseline 2599 教师蒸馏
#   bash run_distill.sh \
#     --teacher_checkpoint exp/fsmn_ctc_baseline_4gpus/79.pt \
#     --num_keywords 2599 \
#     --dict_dir dict \
#     --target_exp_dir exp/fsmn_ctc_distill_mini_align_2599 \
#     2 3
#
#   # 自定义训练参数
#   bash run_distill.sh --gpus "0,1" --align_epochs 80 --finetune_epochs 30 2 3
#
#   # 从某个学生 checkpoint 继续训练（并可重置学习率）
#   bash run_distill.sh \
#     --checkpoint exp/fsmn_ctc_distill_mini_align_20/79.pt \
#     --resume_lr 0.001 \
#     --align_epochs 200 --finetune_epochs 50 \
#     --finetune_lr 0.0001 \
#     --target_exp_dir exp/fsmn_ctc_distill_mini_align_20_more \
#     2 2
#
# 日志文件会自动保存到: <target_exp_dir>/logs/run_distill_stage_<stage>_<stop_stage>_<timestamp>.log

. ./path.sh

# 过滤 torchaudio 弃用警告
export PYTHONWARNINGS="ignore::UserWarning"

# 保存原始参数用于日志
original_args="$@"

stage=2
stop_stage=3

# ---- 教师模型 ----
teacher_checkpoint=exp/fsmn_ctc_top20_weight_surgery/79.pt
teacher_config=  # 为空则从 teacher_checkpoint 目录自动推导

# ---- 学生模型 ----
student_config=conf/fsmn_ctc_student_mini.yaml
num_keywords=20
dict_dir="dict_top20"

# ---- 断点继续训练 ----
# student checkpoint to resume (e.g. exp/fsmn_ctc_distill_mini_align_20/79.pt)
checkpoint=
# override lr when resuming from --checkpoint (e.g. 0.001)
resume_lr=

# ---- 实验目录 ----
target_exp_dir=exp/fsmn_ctc_distill_mini_align

# ---- 训练参数 ----
gpus="0,1,2,3"
norm_mean=true
norm_var=true
seed=666

# ---- 学习率调度（train_distill.py）----
# 默认: scheduler 只在 finetune 阶段生效（scheduler_start_epoch=-1 -> align_epochs）
lr_scheduler=plateau
scheduler_start_epoch=-1
plateau_factor=0.5
plateau_patience=3
plateau_threshold=0.01
plateau_min_lr=1e-6
plateau_cooldown=0

# ---- Feature Alignment 蒸馏参数 ----
align_epochs=100
finetune_epochs=20
finetune_lr=
head_lr_ratio=0.01
finetune_mse_weight_start=0.5
finetune_mse_weight_end=0.1
layer_mapping="0:1,1:2,2:3"

# ---- 评测参数 ----
average_model=true
num_average=30
window_shift=50
token_file="mobvoi_kws_transcription/tokens.txt"
lexicon_file="mobvoi_kws_transcription/lexicon.txt"

. tools/parse_options.sh || exit 1;

# parse_options.sh 处理完选项后，剩余的是位置参数
if [ $# -ge 1 ]; then
  stage=$1
fi
if [ $# -ge 2 ]; then
  stop_stage=$2
fi

dir=$target_exp_dir
if $average_model; then
  score_checkpoint=$dir/avg_${num_average}.pt
else
  score_checkpoint=$dir/final.pt
fi

# 创建日志目录和日志文件
log_dir=$dir/logs
mkdir -p $log_dir

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=$log_dir/run_distill_stage_${stage}_${stop_stage}_${timestamp}.log

# 如果还没有重定向到 tee（避免递归）
if [ -z "$LOG_REDIRECT_DONE" ]; then
  echo "实验目录: $dir"
  echo "日志文件: $log_file"
  echo "================================================"
  export LOG_REDIRECT_DONE=1
  exec > >(tee -a "$log_file") 2>&1
  echo "================================================"
  echo "开始运行: $(date)"
  echo "   命令: bash $0 $original_args"
  echo "   Stage: $stage -> $stop_stage"
  echo "   实验目录: $dir"
  echo "   日志文件: $log_file"
  echo "================================================"
fi

stage_int=$(echo "$stage" | awk '{print int($1)}')
stop_stage_int=$(echo "$stop_stage" | awk '{print int($1)}')

# ================================================================
# Stage 2: Feature Alignment 蒸馏训练
# ================================================================
if [ ${stage_int} -le 2 ] && [ ${stop_stage_int} -ge 2 ]; then
  echo ""
  echo "================================================"
  echo "Stage 2: Feature Alignment 蒸馏训练"
  echo "================================================"
  echo "教师模型:           $teacher_checkpoint"
  echo "教师配置:           ${teacher_config:-auto}"
  echo "学生配置:           $student_config"
  echo "词表目录:           $dict_dir"
  echo "输出关键词数:       $num_keywords"
  echo "Phase 1 对齐 epoch: $align_epochs"
  echo "Phase 2 微调 epoch: $finetune_epochs"
  echo "Finetune lr:        ${finetune_lr:-keep}"
  echo "HEAD lr ratio:      $head_lr_ratio"
  echo "微调 MSE weight:    $finetune_mse_weight_start -> $finetune_mse_weight_end"
  echo "Layer mapping:      $layer_mapping"
  echo "Resume checkpoint:  ${checkpoint:-none}"
  echo "Resume lr override: ${resume_lr:-none}"
  echo "LR scheduler:       $lr_scheduler"
  echo "Scheduler start:    $scheduler_start_epoch"
  echo "Plateau factor:     $plateau_factor"
  echo "Plateau patience:   $plateau_patience"
  echo "Plateau threshold:  $plateau_threshold"
  echo "Plateau min lr:     $plateau_min_lr"
  echo "Plateau cooldown:   $plateau_cooldown"
  echo "GPU:                $gpus"
  echo "================================================"
  echo ""

  # 检查教师模型文件
  if [ ! -f "$teacher_checkpoint" ]; then
    echo "错误: 教师模型文件不存在: $teacher_checkpoint"
    exit 1
  fi

  # 检查学生配置
  if [ ! -f "$student_config" ]; then
    echo "错误: 学生配置文件不存在: $student_config"
    exit 1
  fi

  # 检查 CMVN 文件
  if [ ! -f data/global_cmvn.kaldi ]; then
    echo "CMVN 文件不存在，尝试从预训练模型复制..."
    if [ -f speech_charctc_kws_phone-xiaoyun/train/feature_transform.txt.80dim-l2r2 ]; then
      cp speech_charctc_kws_phone-xiaoyun/train/feature_transform.txt.80dim-l2r2 data/global_cmvn.kaldi
    else
      echo "错误: 无法找到 CMVN 文件"
      exit 1
    fi
  fi

  echo "开始 Feature Alignment 蒸馏训练 ..."
  mkdir -p $dir

  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/global_cmvn.kaldi"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"

  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  teacher_config_opt=
  if [ -n "$teacher_config" ]; then
    teacher_config_opt="--teacher_config $teacher_config"
  fi

  checkpoint_opt=
  if [ -n "$checkpoint" ]; then
    checkpoint_opt="--checkpoint $checkpoint"
  fi
  resume_lr_opt=
  if [ -n "$resume_lr" ]; then
    resume_lr_opt="--resume_lr $resume_lr"
  fi
  finetune_lr_opt=
  if [ -n "$finetune_lr" ]; then
    finetune_lr_opt="--finetune_lr $finetune_lr"
  fi

  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wekws/bin/train_distill.py --gpus $gpus \
      --config $student_config \
      --train_data data/train/data.list \
      --cv_data data/dev/data.list \
      --model_dir $dir \
      --num_workers 8 \
      --num_keywords $num_keywords \
      --dict $dict_dir \
      --min_duration 50 \
      --seed $seed \
      $checkpoint_opt \
      $resume_lr_opt \
      $finetune_lr_opt \
      --lr_scheduler $lr_scheduler \
      --scheduler_start_epoch $scheduler_start_epoch \
      --plateau_factor $plateau_factor \
      --plateau_patience $plateau_patience \
      --plateau_threshold $plateau_threshold \
      --plateau_min_lr $plateau_min_lr \
      --plateau_cooldown $plateau_cooldown \
      --teacher_checkpoint $teacher_checkpoint \
      $teacher_config_opt \
      --align_epochs $align_epochs \
      --finetune_epochs $finetune_epochs \
      --head_lr_ratio $head_lr_ratio \
      --finetune_mse_weight_start $finetune_mse_weight_start \
      --finetune_mse_weight_end $finetune_mse_weight_end \
      --layer_mapping "$layer_mapping" \
      $cmvn_opts

  if [ $? -ne 0 ]; then
    echo "蒸馏训练失败！"
    exit 1
  fi
  echo ""
  echo "Stage 2 蒸馏训练完成！"
fi


# ================================================================
# Stage 3: 模型平均 + 评测
# ================================================================
if [ ${stage_int} -le 3 ] && [ ${stop_stage_int} -ge 3 ]; then
  echo ""
  echo "================================================"
  echo "Stage 3: 模型平均 + 评测"
  echo "================================================"
  echo ""

  if $average_model; then
    echo "模型平均: 最后 ${num_average} 个 epoch (val_best)..."
    python wekws/bin/average_model.py \
      --dst_model $score_checkpoint \
      --src_path $dir \
      --num ${num_average} \
      --val_best

    if [ $? -ne 0 ]; then
      echo "模型平均失败！"
      exit 1
    fi
  fi

  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir

  echo "推理评测中..."
  python wekws/bin/score_ctc.py \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --gpu 0 \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --dict $dict_dir \
    --score_file $result_dir/score.txt \
    --num_workers 8 \
    --keywords "嗨小问,你好问问" \
    --token_file $token_file \
    --lexicon_file $lexicon_file

  if [ $? -ne 0 ]; then
    echo "推理失败！"
    exit 1
  fi

  echo "计算 DET 曲线..."
  python wekws/bin/compute_det_ctc.py \
    --keywords "嗨小问,你好问问" \
    --test_data data/test/data.list \
    --window_shift $window_shift \
    --step 0.001 \
    --score_file $result_dir/score.txt \
    --dict $dict_dir \
    --token_file $token_file \
    --lexicon_file $lexicon_file

  if [ $? -ne 0 ]; then
    echo "DET 计算失败！"
    exit 1
  fi

  echo ""
  echo "Stage 3 评测完成！结果保存在: $result_dir"
fi


# 脚本结束日志
if [ -n "$LOG_REDIRECT_DONE" ]; then
  echo ""
  echo "================================================"
  echo "运行完成: $(date)"
  echo "   实验目录: $dir"
  echo "   日志文件: $log_file"
  echo "================================================"
fi
