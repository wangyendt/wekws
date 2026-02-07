#!/bin/bash
# Copyright 2026 Wayne
#
# 知识蒸馏训练脚本：用教师模型（FSMN top20）蒸馏训练学生模型（FSMN-mini）
#
# 用法示例（选项参数必须放在位置参数之前）:
#   bash run_distill.sh 2 2                                             # Stage 2: 蒸馏训练
#   bash run_distill.sh 3 3                                             # Stage 3: 模型平均 + 评测
#   bash run_distill.sh 2 3                                             # Stage 2+3: 蒸馏训练 + 评测
#   bash run_distill.sh --gpus "0,1" --kd_temperature 4.0 2 3         # 自定义参数
#   bash run_distill.sh --teacher_checkpoint exp/xxx/79.pt 2 3        # 指定教师模型
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

# ---- 实验目录 ----
target_exp_dir=exp/fsmn_ctc_distill_mini

# ---- 训练参数 ----
gpus="0,1,2,3"
norm_mean=true
norm_var=true
seed=666

# ---- 蒸馏参数 ----
kd_temperature=2.0
kd_lambda_init=0.7
kd_lambda_final=0.5
kd_lambda_switch_epoch=20
finetune_epochs=10
init_from_teacher=false

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
  echo "📝 实验目录: $dir"
  echo "📝 日志文件: $log_file"
  echo "================================================"
  export LOG_REDIRECT_DONE=1
  exec > >(tee -a "$log_file") 2>&1
  echo "================================================"
  echo "🚀 开始运行: $(date)"
  echo "   命令: bash $0 $original_args"
  echo "   Stage: $stage -> $stop_stage"
  echo "   实验目录: $dir"
  echo "   日志文件: $log_file"
  echo "================================================"
fi

stage_int=$(echo "$stage" | awk '{print int($1)}')
stop_stage_int=$(echo "$stop_stage" | awk '{print int($1)}')

# ================================================================
# Stage 2: 蒸馏训练
# ================================================================
if [ ${stage_int} -le 2 ] && [ ${stop_stage_int} -ge 2 ]; then
  echo ""
  echo "================================================"
  echo "🎓 Stage 2: 知识蒸馏训练"
  echo "================================================"
  echo "教师模型:       $teacher_checkpoint"
  echo "教师配置:       ${teacher_config:-auto}"
  echo "学生配置:       $student_config"
  echo "词表目录:       $dict_dir"
  echo "输出关键词数:   $num_keywords"
  echo "蒸馏温度 T:     $kd_temperature"
  echo "Lambda 初始:    $kd_lambda_init"
  echo "Lambda 后期:    $kd_lambda_final"
  echo "Lambda 切换:    epoch $kd_lambda_switch_epoch"
  echo "纯CTC收尾:     最后 $finetune_epochs epoch"
  echo "教师权重初始化: $init_from_teacher"
  echo "GPU:            $gpus"
  echo "================================================"
  echo ""

  # 检查教师模型文件
  if [ ! -f "$teacher_checkpoint" ]; then
    echo "❌ 错误: 教师模型文件不存在: $teacher_checkpoint"
    exit 1
  fi

  # 检查学生配置
  if [ ! -f "$student_config" ]; then
    echo "❌ 错误: 学生配置文件不存在: $student_config"
    exit 1
  fi

  # 检查 CMVN 文件
  if [ ! -f data/global_cmvn.kaldi ]; then
    echo "⚠️  CMVN 文件不存在，尝试从预训练模型复制..."
    if [ -f speech_charctc_kws_phone-xiaoyun/train/feature_transform.txt.80dim-l2r2 ]; then
      cp speech_charctc_kws_phone-xiaoyun/train/feature_transform.txt.80dim-l2r2 data/global_cmvn.kaldi
    else
      echo "❌ 错误: 无法找到 CMVN 文件"
      exit 1
    fi
  fi

  echo "开始蒸馏训练 ..."
  mkdir -p $dir

  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/global_cmvn.kaldi"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"

  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  teacher_config_opt=
  if [ -n "$teacher_config" ]; then
    teacher_config_opt="--teacher_config $teacher_config"
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
      --teacher_checkpoint $teacher_checkpoint \
      $teacher_config_opt \
      --kd_temperature $kd_temperature \
      --kd_lambda_init $kd_lambda_init \
      --kd_lambda_final $kd_lambda_final \
      --kd_lambda_switch_epoch $kd_lambda_switch_epoch \
      --finetune_epochs $finetune_epochs \
      --init_from_teacher $init_from_teacher \
      $cmvn_opts

  if [ $? -ne 0 ]; then
    echo "❌ 蒸馏训练失败！"
    exit 1
  fi
  echo ""
  echo "✅ Stage 2 蒸馏训练完成！"
fi


# ================================================================
# Stage 3: 模型平均 + 评测
# ================================================================
if [ ${stage_int} -le 3 ] && [ ${stop_stage_int} -ge 3 ]; then
  echo ""
  echo "================================================"
  echo "📊 Stage 3: 模型平均 + 评测"
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
      echo "❌ 模型平均失败！"
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
    --keywords "\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee" \
    --token_file $token_file \
    --lexicon_file $lexicon_file

  if [ $? -ne 0 ]; then
    echo "❌ 推理失败！"
    exit 1
  fi

  echo "计算 DET 曲线..."
  python wekws/bin/compute_det_ctc.py \
    --keywords "\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee" \
    --test_data data/test/data.list \
    --window_shift $window_shift \
    --step 0.001 \
    --score_file $result_dir/score.txt \
    --dict $dict_dir \
    --token_file $token_file \
    --lexicon_file $lexicon_file

  if [ $? -ne 0 ]; then
    echo "❌ DET 计算失败！"
    exit 1
  fi

  echo ""
  echo "✅ Stage 3 评测完成！结果保存在: $result_dir"
fi


# 脚本结束日志
if [ -n "$LOG_REDIRECT_DONE" ]; then
  echo ""
  echo "================================================"
  echo "✅ 运行完成: $(date)"
  echo "   实验目录: $dir"
  echo "   日志文件: $log_file"
  echo "================================================"
fi
