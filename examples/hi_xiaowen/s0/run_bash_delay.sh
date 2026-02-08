#!/usr/bin/env bash
set -euo pipefail

help_message=$(
  cat <<'EOF'
Usage:
  bash ./run_bash_delay.sh --delay 6 --cmd "bash ./run_distill.sh ... "

Options:
  --delay <hours>         Delay in hours before running command (default: 8)
  --delay_hours <hours>   Alias of --delay
  --cmd "<command>"       Command to execute (required)
  --log_file <path>       Optional log file path (default: no log file)
  --dry_run <true|false>  Print schedule and exit (default: false)
EOF
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

delay=8
delay_hours=""
cmd=""
log_file=""
dry_run=false

. tools/parse_options.sh || exit 1;

if [ -n "${delay_hours}" ]; then
  delay="${delay_hours}"
fi

if [ -z "${cmd}" ]; then
  echo "[ERROR] --cmd is required."
  echo "${help_message}"
  exit 1
fi

if ! [[ "${delay}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] --delay must be a non-negative number (hours), got: ${delay}"
  exit 1
fi

delay_seconds=$(awk -v d="${delay}" 'BEGIN { printf "%.0f", d * 3600 }')
if [ "${delay_seconds}" -lt 0 ]; then
  echo "[ERROR] Computed delay seconds is negative: ${delay_seconds}"
  exit 1
fi

if [ -z "${log_file}" ]; then
  log_file=""
fi

start_epoch="$(date '+%s')"
run_epoch=$((start_epoch + delay_seconds))
start_time="$(date '+%Y-%m-%d %H:%M:%S')"
run_time="$(date -d "@${run_epoch}" '+%Y-%m-%d %H:%M:%S')"

echo "[INFO] Current time: ${start_time}"
echo "[INFO] Delay: ${delay} hours (${delay_seconds} seconds)"
echo "[INFO] Will run at: ${run_time}"
if [ -n "${log_file}" ]; then
  echo "[INFO] Log file: ${SCRIPT_DIR}/${log_file}"
else
  echo "[INFO] Log file: disabled"
fi
echo "[INFO] Command: ${cmd}"

if [ "${dry_run}" = "true" ]; then
  echo "[INFO] dry_run=true, exiting without execution."
  exit 0
fi

if [ "${delay_seconds}" -gt 0 ]; then
  echo "[INFO] Sleeping for ${delay_seconds} seconds..."
  sleep "${delay_seconds}"
fi

echo "[INFO] Launching command at $(date '+%Y-%m-%d %H:%M:%S')"
set +e
if [ -n "${log_file}" ]; then
  bash -lc "${cmd}" 2>&1 | tee -a "${log_file}"
  cmd_exit_code=${PIPESTATUS[0]}
else
  bash -lc "${cmd}"
  cmd_exit_code=$?
fi
set -e

echo "[INFO] Command finished with exit code: ${cmd_exit_code}"
exit "${cmd_exit_code}"
