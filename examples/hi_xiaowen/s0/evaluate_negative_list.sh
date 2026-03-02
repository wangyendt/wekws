#!/bin/bash
#
# 负样本列表评测脚本
# 目标：给定模型与负样本音频列表，批量推理并统计误唤醒率
#
# 用法示例：
#   bash evaluate_negative_list.sh \
#     --checkpoint exp/fsmn_ctc_distill_mini_align_20_test2/229.pt \
#     --audio_list /home/xushang/kws_ws/wekws/examples/hi_xiaowen/s0/datasets/mining_0207/audio_segments.list \
#     --dict_dir dict_top20 --gpu 0
#
# 说明：
# - 本脚本不会改动现有 train/dev/test 的 data.list；
# - 会在 result_dir 下生成独立的标准化 data.list 与评测结果；
# - 推理复用 wekws/bin/score_ctc.py，特征处理流程与现有评测一致。

set -euo pipefail

. ./path.sh

checkpoint=""
executorch_model=""
executorch_seq_len=100
model_config=""
audio_list=""
gpu="0"
batch_size=256
num_workers=8
dict_dir="dict_top20"
keywords="\\u55e8\\u5c0f\\u95ee,\\u4f60\\u597d\\u95ee\\u95ee"
token_file="mobvoi_kws_transcription/tokens.txt"
lexicon_file="mobvoi_kws_transcription/lexicon.txt"
result_dir=""
max_samples=0
skip_missing_wav=true

. tools/parse_options.sh || exit 1

if [ -z "${audio_list}" ]; then
  echo "错误: 必须指定 --audio_list"
  exit 1
fi
if [ ! -f "${audio_list}" ]; then
  echo "错误: audio_list 不存在: ${audio_list}"
  exit 1
fi

if [ -z "${checkpoint}" ] && [ -z "${executorch_model}" ]; then
  echo "错误: 必须指定 --checkpoint 或 --executorch_model"
  exit 1
fi
if [ -n "${checkpoint}" ] && [ ! -f "${checkpoint}" ]; then
  echo "错误: checkpoint 不存在: ${checkpoint}"
  exit 1
fi
if [ -n "${executorch_model}" ] && [ ! -f "${executorch_model}" ]; then
  echo "错误: executorch_model 不存在: ${executorch_model}"
  exit 1
fi
if [ ! -f "${token_file}" ]; then
  echo "错误: token_file 不存在: ${token_file}"
  exit 1
fi
if [ ! -f "${lexicon_file}" ]; then
  echo "错误: lexicon_file 不存在: ${lexicon_file}"
  exit 1
fi

primary_model="${checkpoint}"
if [ -n "${executorch_model}" ]; then
  primary_model="${executorch_model}"
fi
model_dir="$(dirname "${primary_model}")"
model_filename="$(basename "${primary_model}")"
model_basename="${model_filename%.*}"

if [ -n "${model_config}" ]; then
  config_file="${model_config}"
else
  config_file="${model_dir}/config.yaml"
fi
if [ ! -f "${config_file}" ]; then
  echo "错误: config 不存在: ${config_file}"
  exit 1
fi

audio_list_abs="$(python - "${audio_list}" <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"
audio_list_name="$(basename "${audio_list_abs}")"
audio_list_stem="${audio_list_name%.*}"

if [ -z "${result_dir}" ]; then
  result_dir="${model_dir}/neg_eval_${audio_list_stem}_${model_basename}"
fi
mkdir -p "${result_dir}"

prepared_list="${result_dir}/negative_eval.data.list"
prepare_stats_file="${result_dir}/prepare_stats.json"
score_file="${result_dir}/score.txt"
summary_file="${result_dir}/summary.json"
detected_file="${result_dir}/detected.list"

echo "================================================"
echo "🎯 负样本评测配置"
echo "================================================"
echo "模型 checkpoint:   ${checkpoint}"
echo "ExecuTorch 模型:   ${executorch_model}"
echo "模型 config:       ${config_file}"
echo "输入列表:          ${audio_list_abs}"
echo "标准化列表:        ${prepared_list}"
echo "结果目录:          ${result_dir}"
echo "GPU:               ${gpu}"
echo "Batch size:        ${batch_size}"
echo "Num workers:       ${num_workers}"
echo "关键词:            ${keywords}"
echo "max_samples:       ${max_samples}"
echo "skip_missing_wav:  ${skip_missing_wav}"
echo "================================================"
echo ""

python - "${audio_list_abs}" "${prepared_list}" "${prepare_stats_file}" "${max_samples}" "${skip_missing_wav}" <<'PY'
import json
import os
import sys
from pathlib import Path

src_path = Path(sys.argv[1]).resolve()
dst_path = Path(sys.argv[2]).resolve()
stats_path = Path(sys.argv[3]).resolve()
max_samples = int(sys.argv[4])
skip_missing_wav = str(sys.argv[5]).lower() == 'true'

def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

total_lines = 0
kept = 0
bad_lines = 0
missing_wav = 0
duration_sum = 0.0

dst_path.parent.mkdir(parents=True, exist_ok=True)

with src_path.open('r', encoding='utf8') as fin, \
        dst_path.open('w', encoding='utf8') as fout:
    for idx, raw in enumerate(fin, 1):
        line = raw.strip()
        if not line:
            continue
        total_lines += 1

        obj = None
        if line.startswith('{'):
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue
        else:
            arr = line.split()
            if len(arr) >= 2:
                obj = {'key': arr[0], 'wav': arr[1], 'txt': '<FILLER>'}
            elif len(arr) == 1:
                wav = arr[0]
                obj = {'key': Path(wav).stem, 'wav': wav, 'txt': '<FILLER>'}
            else:
                bad_lines += 1
                continue

        wav = str(obj.get('wav', '')).strip()
        if not wav:
            bad_lines += 1
            continue
        if not os.path.isabs(wav):
            wav = str((src_path.parent / wav).resolve())

        if not os.path.exists(wav):
            missing_wav += 1
            if skip_missing_wav:
                continue
            raise FileNotFoundError(f'missing wav: {wav}')

        key = str(obj.get('key', '')).strip()
        if not key:
            key = Path(wav).stem

        txt = str(obj.get('txt', '<FILLER>') or '<FILLER>')
        duration = to_float(obj.get('duration', 0.0), 0.0)
        if duration < 0:
            duration = 0.0

        out_obj = {
            'key': key,
            'wav': wav,
            'txt': txt,
            'duration': duration,
            'label_type': 'negative',
            'split': 'neg_eval'
        }
        fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
        kept += 1
        duration_sum += duration

        if max_samples > 0 and kept >= max_samples:
            break

stats = {
    'src_path': str(src_path),
    'dst_path': str(dst_path),
    'total_lines': total_lines,
    'kept': kept,
    'bad_lines': bad_lines,
    'missing_wav': missing_wav,
    'duration_sum_sec': duration_sum,
    'duration_hours': duration_sum / 3600.0 if duration_sum > 0 else 0.0,
}
with stats_path.open('w', encoding='utf8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(json.dumps(stats, ensure_ascii=False))
PY

prepared_kept="$(python - "${prepare_stats_file}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], 'r', encoding='utf8'))['kept'])
PY
)"
if [ "${prepared_kept}" -le 0 ]; then
  echo "错误: 标准化后没有可用样本，退出。"
  exit 1
fi

first_gpu="$(echo "${gpu}" | awk -F',' '{print $1}')"
score_cmd=(python wekws/bin/score_ctc.py
  --config "${config_file}"
  --test_data "${prepared_list}"
  --gpu "${first_gpu}"
  --batch_size "${batch_size}"
  --dict "${dict_dir}"
  --score_file "${score_file}"
  --num_workers "${num_workers}"
  --keywords "${keywords}"
  --token_file "${token_file}"
  --lexicon_file "${lexicon_file}")

if [ -n "${checkpoint}" ]; then
  score_cmd+=(--checkpoint "${checkpoint}")
fi
if [ -n "${executorch_model}" ]; then
  score_cmd+=(--executorch_model "${executorch_model}")
  score_cmd+=(--executorch_seq_len "${executorch_seq_len}")
fi

echo ""
echo "🚀 运行批量推理..."
echo "命令: ${score_cmd[*]}"
"${score_cmd[@]}"

if [ $? -ne 0 ]; then
  echo "❌ 推理失败"
  exit 1
fi

python - "${prepared_list}" "${score_file}" "${summary_file}" "${detected_file}" <<'PY'
import json
import sys
from collections import Counter

data_list = sys.argv[1]
score_file = sys.argv[2]
summary_file = sys.argv[3]
detected_file = sys.argv[4]

total_utts = 0
duration_sum = 0.0
with open(data_list, 'r', encoding='utf8') as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total_utts += 1
        obj = json.loads(line)
        try:
            duration_sum += float(obj.get('duration', 0.0) or 0.0)
        except Exception:
            pass

detected = 0
keyword_counter = Counter()
detected_rows = []
with open(score_file, 'r', encoding='utf8') as fin:
    for line in fin:
        arr = line.strip().split()
        if len(arr) >= 2 and arr[1] == 'detected':
            detected += 1
            key = arr[0]
            keyword = arr[2] if len(arr) >= 3 else 'unknown'
            score = arr[3] if len(arr) >= 4 else ''
            keyword_counter[keyword] += 1
            detected_rows.append((key, keyword, score))

with open(detected_file, 'w', encoding='utf8') as fout:
    for key, keyword, score in detected_rows:
        fout.write(f'{key}\t{keyword}\t{score}\n')

fa_rate = (detected / total_utts) if total_utts > 0 else 0.0
hours = duration_sum / 3600.0 if duration_sum > 0 else 0.0
fa_per_hour = (detected / hours) if hours > 0 else 0.0

summary = {
    'total_utts': total_utts,
    'detected_utts': detected,
    'false_alarm_rate': fa_rate,
    'false_alarm_rate_percent': fa_rate * 100.0,
    'duration_sum_sec': duration_sum,
    'duration_hours': hours,
    'fa_per_hour': fa_per_hour,
    'per_keyword_detected': dict(keyword_counter),
    'score_file': score_file,
    'detected_file': detected_file,
}
with open(summary_file, 'w', encoding='utf8') as fout:
    json.dump(summary, fout, ensure_ascii=False, indent=2)

print("================================================")
print("✅ 负样本评测完成")
print("================================================")
print(f"总样本数:        {total_utts}")
print(f"误唤醒样本数:    {detected}")
print(f"误唤醒率:        {fa_rate*100:.4f}%")
print(f"总时长(小时):    {hours:.4f}")
print(f"误唤醒(次/小时): {fa_per_hour:.6f}")
print(f"按关键词统计:    {dict(keyword_counter)}")
print("================================================")
print(f"summary:         {summary_file}")
print(f"score:           {score_file}")
print(f"detected list:   {detected_file}")
PY
