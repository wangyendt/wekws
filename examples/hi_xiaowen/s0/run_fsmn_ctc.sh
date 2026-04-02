#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#           2023  Jing Du(thuduj12@163.com)
#
# 用法示例（注意：选项参数必须放在位置参数之前）:
#   bash run_fsmn_ctc.sh 2 2                                        # 使用默认实验目录 exp/fsmn_ctc
#   bash run_fsmn_ctc.sh --target_exp_dir exp/my_exp 2 2          # 指定实验目录（正确）
#   bash run_fsmn_ctc.sh 2 2 --target_exp_dir exp/my_exp          # 错误！选项会被忽略
#
# 日志文件会自动保存到: <target_exp_dir>/logs/run_stage_<stage>_<stop_stage>_<timestamp>.log
# 日志同时输出到终端和文件（使用 tee 命令）

. ./path.sh

# 过滤 torchaudio 弃用警告
export PYTHONWARNINGS="ignore::UserWarning"

# 保存原始参数用于日志
original_args="$@"

stage=-1
stop_stage=-1
num_keywords=2599

config=conf/fsmn_ctc.yaml
norm_mean=true
norm_var=true
gpus="0"
dict_dir="dict"
checkpoint_dict="dict"
checkpoint_strict=true
dict_auto_build=false
dict_sorted_file="examples/hi_xiaowen/s0/dict/model_vocab_freq_asr_sorted.txt"
data_dir=data

checkpoint=
target_exp_dir=exp/fsmn_ctc
average_model=true
num_average=30

# 尝试从配置文件读取 download_dir
if [ -f wayne_scripts/config.yaml ]; then
    config_download_dir=$(python tools/read_config.py download_dir 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$config_download_dir" ]; then
        download_dir="$config_download_dir"
        echo "✅ 从 config.yaml 读取 download_dir: $download_dir"
    else
        download_dir=/home/data/datasets/kws/opensourced/nihaowenwen
        echo "⚠️  配置文件读取失败，使用默认 download_dir: $download_dir"
    fi
else
    # download_dir=/mnt/52_disk/back/DuJing/data/nihaowenwen # your data dir
    # download_dir=/Users/wayne/Documents/work/code/project/ffalcon/kws/wekws/examples/hi_xiaowen/s0/data
    download_dir=/home/data/datasets/kws/opensourced/nihaowenwen
fi

. tools/parse_options.sh || exit 1;

# parse_options.sh 处理完选项后，剩余的是位置参数
if [ $# -ge 1 ]; then
  stage=$1
fi
if [ $# -ge 2 ]; then
  stop_stage=$2
fi

window_shift=50

# 设置实验目录
dir=$target_exp_dir
if $average_model ;then
  score_checkpoint=$dir/avg_${num_average}.pt
else
  score_checkpoint=$dir/final.pt
fi

# 创建日志目录和日志文件
log_dir=$dir/logs
mkdir -p $log_dir

# 生成日志文件名（带时间戳）
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=$log_dir/run_stage_${stage}_${stop_stage}_${timestamp}.log

# 如果还没有重定向到 tee（避免递归）
if [ -z "$LOG_REDIRECT_DONE" ]; then
  echo "📝 实验目录: $dir"
  echo "📝 日志文件: $log_file"
  echo "================================================"
  export LOG_REDIRECT_DONE=1
  # 重新执行脚本，输出同时到终端和日志文件
  exec > >(tee -a "$log_file") 2>&1
  # 记录脚本开始时间和参数
  echo "================================================"
  echo "🚀 开始运行: $(date)"
  echo "   命令: bash $0 $original_args"
  echo "   Stage: $stage -> $stop_stage"
  echo "   实验目录: $dir"
  echo "   日志文件: $log_file"
  echo "================================================"
fi

# 将浮点数 stage 转换为整数进行比较（如果是整数则直接使用）
if [ "$stage" = "1.5" ] || [ "$stop_stage" = "1.5" ]; then
  # Stage 1.5 是特殊的中间 stage，单独处理
  stage_int=999
  stop_stage_int=999
else
  stage_int=$(echo "$stage" | awk '{print int($1)}')
  stop_stage_int=$(echo "$stop_stage" | awk '{print int($1)}')
fi

if $dict_auto_build; then
  echo "Auto build dict from sorted vocab..."
  mkdir -p "$dict_dir"
  python tools/gen_reduced_dict.py \
    --sorted_file "$dict_sorted_file" \
    --num_keywords "$num_keywords" \
    --output_dir "$dict_dir"
fi

if [ ${stage_int} -le -2 ] && [ ${stop_stage_int} -ge -2 ]; then
  echo "Download and extracte all datasets"
  local/mobvoi_data_download.sh --dl_dir $download_dir
fi


if [ ${stage_int} -le -1 ] && [ ${stop_stage_int} -ge -1 ]; then
  echo "Preparing datasets..."
  mkdir -p dict
  echo "<FILLER> -1" > dict/dict.txt
  echo "<HI_XIAOWEN> 0" >> dict/dict.txt
  echo "<NIHAO_WENWEN> 1" >> dict/dict.txt
  awk '{print $1}' dict/dict.txt > dict/words.txt

  for folder in train dev test; do
    mkdir -p ${data_dir}/$folder
    for prefix in p n; do
      mkdir -p ${data_dir}/${prefix}_$folder
      json_path=$download_dir/mobvoi_hotword_dataset_resources/${prefix}_$folder.json
      local/prepare_data.py $download_dir/mobvoi_hotword_dataset $json_path \
        ${dict_dir}/dict.txt ${data_dir}/${prefix}_$folder
    done
    cat ${data_dir}/p_$folder/wav.scp ${data_dir}/n_$folder/wav.scp > ${data_dir}/$folder/wav.scp
    cat ${data_dir}/p_$folder/text ${data_dir}/n_$folder/text > ${data_dir}/$folder/text
    rm -rf ${data_dir}/p_$folder ${data_dir}/n_$folder
  done
fi

if [ ${stage_int} -le -0 ] && [ ${stop_stage_int} -ge -0 ]; then
# Here we Use Paraformer Large(https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
# to transcribe the negative wavs, and upload the transcription to modelscope.
  git clone https://www.modelscope.cn/datasets/thuduj12/mobvoi_kws_transcription.git
  for folder in train dev test; do
    if [ -f ${data_dir}/$folder/text ];then
      mv ${data_dir}/$folder/text ${data_dir}/$folder/text.label
    fi
    cp mobvoi_kws_transcription/$folder.text ${data_dir}/$folder/text
  done

  # and we also copy the tokens and lexicon that used in
  # https://modelscope.cn/models/damo/speech_charctc_kws_phone-xiaoyun/summary
  awk '{print $1, $2-1}' mobvoi_kws_transcription/tokens.txt > ${dict_dir}/dict.txt
  sed -i 's/& 1/<filler> 1/' ${dict_dir}/dict.txt
  echo '<SILENCE>' > ${dict_dir}/words.txt
  echo '<EPS>' >> ${dict_dir}/words.txt
  echo '<BLK>' >> ${dict_dir}/words.txt
fi

if [ ${stage_int} -le 1 ] && [ ${stop_stage_int} -ge 1 ]; then
  echo "Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp ${data_dir}/train/wav.scp \
    --out_cmvn ${data_dir}/train/global_cmvn

  for x in train dev test; do
    tools/wav_to_duration.sh --nj 8 ${data_dir}/$x/wav.scp ${data_dir}/$x/wav.dur

    # Here we use tokens.txt and lexicon.txt to convert txt into index
    tools/make_list.py ${data_dir}/$x/wav.scp ${data_dir}/$x/text \
      ${data_dir}/$x/wav.dur ${data_dir}/$x/data.list
  done
fi

# Stage 1.5 (Optional): Build metadata database for WebUI
# This stage creates a SQLite database for fast searching and filtering
# Run separately: bash run_fsmn_ctc.sh 1.5 1.5
if [ ${stage} == "1.5" ]; then
  echo "Building metadata database for WebUI..."
  python3 tools/generate_metadata_db.py --output-db ${data_dir}/metadata.db --force
  echo ""
  echo "To start the WebUI, run:"
  echo "  cd wayne_scripts && sh stage_1.5_webui.sh"
  echo ""
  exit 0
fi

if [ ${stage_int} -le 2 ] && [ ${stop_stage_int} -ge 2 ]; then

  echo "Prepare checkpoint and CMVN"
  if [ ! -d speech_charctc_kws_phone-xiaoyun ] ;then
      git lfs install
      git clone https://www.modelscope.cn/damo/speech_charctc_kws_phone-xiaoyun.git
  fi
  if [ -z "$checkpoint" ]; then
    echo "Use the base model from modelscope"
    checkpoint=speech_charctc_kws_phone-xiaoyun/train/base.pt
  else
    echo "Use custom checkpoint: $checkpoint"
  fi
  cp speech_charctc_kws_phone-xiaoyun/train/feature_transform.txt.80dim-l2r2 ${data_dir}/global_cmvn.kaldi

  echo "Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file ${data_dir}/global_cmvn.kaldi"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  # 使用当前 conda 环境的 python，而非系统 python
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wekws/bin/train.py --gpus $gpus \
      --config $config \
      --train_data ${data_dir}/train/data.list \
      --cv_data ${data_dir}/dev/data.list \
      --model_dir $dir \
      --num_workers 8 \
      --num_keywords $num_keywords \
      --dict $dict_dir \
      --checkpoint_dict $checkpoint_dict \
      --checkpoint_strict $checkpoint_strict \
      --min_duration 50 \
      --seed 666 \
      $cmvn_opts \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage_int} -le 3 ] && [ ${stop_stage_int} -ge 3 ]; then
  echo "Do model average, Compute FRR/FAR ..."
  if $average_model; then
    python wekws/bin/average_model.py \
      --dst_model $score_checkpoint \
      --src_path $dir  \
      --num ${num_average} \
      --val_best
  fi
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  stream=false  # we detect keyword online with ctc_prefix_beam_search
  score_prefix=""
  if $stream ; then
    score_prefix=stream_
  fi
  python wekws/bin/${score_prefix}score_ctc.py \
    --config $dir/config.yaml \
    --test_data ${data_dir}/test/data.list \
    --gpu 0  \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --dict $dict_dir \
    --score_file $result_dir/score.txt  \
    --num_workers 8  \
    --keywords "\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee" \
    --token_file data/tokens.txt \
    --lexicon_file data/lexicon.txt

  python wekws/bin/compute_det_ctc.py \
      --keywords "\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee" \
      --test_data ${data_dir}/test/data.list \
      --window_shift $window_shift \
      --step 0.001  \
      --score_file $result_dir/score.txt \
      --dict $dict_dir \
      --token_file data/tokens.txt \
      --lexicon_file data/lexicon.txt
fi


if [ ${stage_int} -le 4 ] && [ ${stop_stage_int} -ge 4 ]; then
  jit_model=$(basename $score_checkpoint | sed -e 's:.pt$:.zip:g')
  onnx_model=$(basename $score_checkpoint | sed -e 's:.pt$:.onnx:g')
  # For now, FSMN can not export to JITScript
#  python wekws/bin/export_jit.py \
#    --config $dir/config.yaml \
#    --checkpoint $score_checkpoint \
#    --jit_model $dir/$jit_model
  python wekws/bin/export_onnx.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --onnx_model $dir/$onnx_model
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
