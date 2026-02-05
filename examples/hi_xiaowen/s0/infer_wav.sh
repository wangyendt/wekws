#!/bin/bash
#
# Single-wav inference helper.
# It reuses wekws/bin/score_ctc.py so features/model behavior match evaluate.sh.
#
# Example:
#   bash infer_wav.sh \
#     --checkpoint exp/fsmn_ctc_baseline_4gpus/79.pt \
#     --wav /path/to/foo.wav \
#     --gpu 0
#
# Or pass an utt_id (will be resolved via data/metadata.db):
#   bash infer_wav.sh \
#     --checkpoint exp/fsmn_ctc_baseline_4gpus/79.pt \
#     --wav 0000010f3cc9ff4c868117b4e4c53fb5 \
#     --gpu 0
#

set -euo pipefail

: "${PYTHONPATH:=}"
. ./path.sh


checkpoint=""
dataset=""
wav=""
txt=""
gpu="0"
batch_size=1
# score_ctc.py uses prefetch_factor; it requires num_workers > 0
num_workers=1

token_file="mobvoi_kws_transcription/tokens.txt"
lexicon_file="mobvoi_kws_transcription/lexicon.txt"

# NOTE: score_ctc.py expects unicode-escaped keywords and will decode with 'unicode_escape'.
keywords="\\u55e8\\u5c0f\\u95ee,\\u4f60\\u597d\\u95ee\\u95ee"

. tools/parse_options.sh || exit 1;

if [ -z "${checkpoint}" ]; then
  echo "ERROR: --checkpoint is required"
  exit 1
fi
if [ ! -f "${checkpoint}" ]; then
  echo "ERROR: checkpoint not found: ${checkpoint}"
  exit 1
fi
if [ -z "${wav}" ]; then
  echo "ERROR: --wav is required"
  exit 1
fi

checkpoint_dir=$(dirname "${checkpoint}")
checkpoint_basename=$(basename "${checkpoint}" .pt)
config_file="${checkpoint_dir}/config.yaml"

if [ ! -f "${config_file}" ]; then
  echo "ERROR: config not found: ${config_file}"
  exit 1
fi
if [ ! -f "${token_file}" ]; then
  echo "ERROR: token file not found: ${token_file}"
  exit 1
fi
if [ ! -f "${lexicon_file}" ]; then
  echo "ERROR: lexicon file not found: ${lexicon_file}"
  exit 1
fi

# Look up metadata from SQLite DB (gold label/text/split).
if [ ! -f "data/metadata.db" ]; then
  echo "ERROR: data/metadata.db not found. Build it first:"
  echo "  bash run_fsmn_ctc.sh 1.5 1.5"
  exit 1
fi

resolved_wav=""
if [ -f "${wav}" ]; then
  resolved_wav=$(python - "${wav}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)
fi

meta_json=$(python - "data/metadata.db" "${wav}" "${resolved_wav}" <<'PY' || echo ""
import json
import os
import sqlite3
import sys

db_path = sys.argv[1]
wav_arg = sys.argv[2]
wav_resolved = sys.argv[3]  # may be empty

def is_hex32(s: str) -> bool:
  if len(s) != 32:
    return False
  for ch in s:
    if ch not in "0123456789abcdefABCDEF":
      return False
  return True

conn = sqlite3.connect(db_path)
cur = conn.cursor()

def query_by_wav_path(path: str):
  cur.execute(
    "SELECT utt_id, split, label_type, wav_path, duration, text_content "
    "FROM audio_metadata WHERE wav_path = ?",
    (path,),
  )
  return cur.fetchall()

def query_by_utt_id(utt_id: str):
  cur.execute(
    "SELECT utt_id, split, label_type, wav_path, duration, text_content "
    "FROM audio_metadata WHERE utt_id = ?",
    (utt_id,),
  )
  return cur.fetchall()

def query_by_wav_suffix(suffix: str):
  # Use suffix match only as a fallback: it may be ambiguous.
  cur.execute(
    "SELECT utt_id, split, label_type, wav_path, duration, text_content "
    "FROM audio_metadata WHERE wav_path LIKE ?",
    (f"%/{suffix}",),
  )
  return cur.fetchall()

rows = []
if wav_resolved:
  wav_abs = os.path.abspath(wav_resolved)
  wav_real = os.path.realpath(wav_resolved)
  rows = query_by_wav_path(wav_abs)
  if not rows and wav_real != wav_abs:
    rows = query_by_wav_path(wav_real)
else:
  # Treat wav_arg as utt_id (preferred) or basename/suffix.
  utt_id = wav_arg
  if utt_id.endswith(".wav"):
    utt_id = utt_id[:-4]
  if is_hex32(utt_id):
    rows = query_by_utt_id(utt_id)
    if not rows:
      # Fallback: match "<utt_id>.wav"
      rows = query_by_wav_suffix(utt_id + ".wav")
  else:
    # Last resort: suffix match as provided.
    rows = query_by_wav_suffix(wav_arg)

conn.close()

if not rows:
  raise SystemExit(2)
if len(rows) > 1:
  # Avoid picking the wrong gold label when the path is ambiguous.
  print(json.dumps({"error": "multiple_rows", "rows": rows}, ensure_ascii=False))
  raise SystemExit(3)

utt_id, split, label_type, wav_path, duration, text_content = rows[0]
print(json.dumps({
  "utt_id": utt_id,
  "split": split,
  "label_type": label_type,
  "wav_path": wav_path,
  "duration": duration,
  "text_content": text_content,
}, ensure_ascii=False))
PY
)

if [ -z "${meta_json}" ]; then
  echo "ERROR: wav not found in data/metadata.db: ${wav}"
  echo "Hint: pass a full wav path, or pass an utt_id that exists in the DB."
  exit 1
fi

meta_err=$(python - "${meta_json}" <<'PY'
import json, sys
try:
  obj=json.loads(sys.argv[1])
except Exception:
  raise SystemExit(0)
print(obj.get("error",""))
PY
)

if [ "${meta_err}" = "multiple_rows" ]; then
  echo "ERROR: multiple rows matched this wav path in metadata.db, refusing to guess."
  python - "${meta_json}" <<'PY'
import json, sys
obj=json.loads(sys.argv[1])
for r in obj["rows"]:
  print(r)
PY
  exit 1
fi

key=$(python - "${meta_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["utt_id"])
PY
)

wav_abs=$(python - "${meta_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["wav_path"])
PY
)

txt=$(python - "${meta_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("text_content",""))
PY
)

duration=$(python - "${meta_json}" <<'PY'
import json, sys
v=json.loads(sys.argv[1]).get("duration", 0.0)
print(0.0 if v is None else v)
PY
)

label_type=$(python - "${meta_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("label_type",""))
PY
)

split_from_db=$(python - "${meta_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("split",""))
PY
)

if [ -z "${dataset}" ]; then
  dataset="${split_from_db:-single}"
fi

result_dir="${checkpoint_dir}/${dataset}_${checkpoint_basename}"
mkdir -p "${result_dir}"

data_list="${result_dir}/single.${key}.data.list"
score_file="${result_dir}/score.single.${key}.txt"
dump_logits_dir="${result_dir}/dump_logits"
dump_probs_dir="${result_dir}/dump_probs"
decode_file="${result_dir}/decode.single.${key}.jsonl"

python - "${key}" "${txt}" "${duration}" "${wav_abs}" "${label_type}" "${split_from_db}" > "${data_list}" <<'PY'
import json, sys
key, txt, duration, wav, label_type, split = \
  sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
obj = {
  "key": key,
  "txt": txt,
  "duration": duration,
  "wav": wav,
  "label_type": label_type,
  "split": split,
}
print(json.dumps(obj, ensure_ascii=False))
PY

echo "Result dir: ${result_dir}"
echo "Data list:  ${data_list}"
echo "Score:      ${score_file}"
echo "Logits dir: ${dump_logits_dir}"
echo "Probs dir:  ${dump_probs_dir}"
echo "Decode:     ${decode_file}"
echo ""

python wekws/bin/score_ctc.py \
  --config "${config_file}" \
  --test_data "${data_list}" \
  --gpu "${gpu}" \
  --batch_size "${batch_size}" \
  --checkpoint "${checkpoint}" \
  --score_file "${score_file}" \
  --num_workers "${num_workers}" \
  --keywords "${keywords}" \
  --token_file "${token_file}" \
  --lexicon_file "${lexicon_file}" \
  --dump_logits_dir "${dump_logits_dir}" \
  --dump_probs_dir "${dump_probs_dir}" \
  --dump_decode_file "${decode_file}" \
  --decode_greedy \
  --decode_beam \
  --decode_prob_threshold 0.0
