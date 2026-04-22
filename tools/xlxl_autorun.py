#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StatRow:
    threshold: float
    fa_per_hour: float
    frr: float


@dataclass
class ExperimentSpec:
    name: str
    exp_dir: str
    data_dir: str
    replay_factor: int
    finetune_epochs: int
    finetune_lr: float
    head_lr_ratio: float
    finetune_mse_weight_start: float
    finetune_mse_weight_end: float
    launch_mode: str  # "external_wait" | "run"
    external_session: Optional[str] = None
    notes: str = ""


REPO_ROOT = Path(__file__).resolve().parents[1]
S0_DIR = REPO_ROOT / "examples" / "hi_xiaowen" / "s0"
DOCS_ROOT = REPO_ROOT / "docs" / "xiaolei-xiaolei"
STATE_ROOT = S0_DIR / "codex_artifacts" / "xlxl_autorun"
LOG_ROOT = STATE_ROOT / "logs"
EXPERIMENT_DOC = DOCS_ROOT / "experiment" / "20260422_199K自治实验记录.md"
PLAN_DOC = DOCS_ROOT / "plan" / "20260422_199K自治实验队列计划.md"
STATE_JSON = STATE_ROOT / "state.json"
HISTORY_JSONL = STATE_ROOT / "history.jsonl"

BEST_STUDENT = {"小雷小雷": 96.70, "小雷快拍": 99.61}
BASELINE_HARDCASE_ACTIVATIONS = {
    "XIAOLEI_XIAOLEI_0031_1m_f_04_fast_01_rep_10": 2,
    "XIAOLEI_XIAOLEI_0021_1m_f_01_fast_04_rep_14": 2,
}
HARDCASES = [
    "/home/xushang/kws_ws/extra_datasets/kws_data_0327/XIAOLEI_XIAOLEI/XIAOLEI_XIAOLEI_0031_1m_f_04_fast_01_rep_10.wav",
    "/home/xushang/kws_ws/extra_datasets/kws_data_0327/XIAOLEI_XIAOLEI/XIAOLEI_XIAOLEI_0021_1m_f_01_fast_04_rep_14.wav",
]


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "hardpos_replay_v2": ExperimentSpec(
        name="hardpos_replay_v2",
        exp_dir="exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_from119_v2",
        data_dir="data_xlxl_0327_ctc_v1_clean_hardpos_replay_v2",
        replay_factor=10,
        finetune_epochs=30,
        finetune_lr=5e-5,
        head_lr_ratio=0.1,
        finetune_mse_weight_start=0.3,
        finetune_mse_weight_end=0.1,
        launch_mode="external_wait",
        external_session="xlxl_hardpos_v2",
        notes="更强 replay + 更短 schedule，先验证是否能把 1/3 hardcase 拉回到至少 2/3。",
    ),
    "hardpos_replay_v3": ExperimentSpec(
        name="hardpos_replay_v3",
        exp_dir="exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_from119_v3",
        data_dir="data_xlxl_0327_ctc_v1_clean_hardpos_replay_v3",
        replay_factor=20,
        finetune_epochs=15,
        finetune_lr=3e-5,
        head_lr_ratio=0.1,
        finetune_mse_weight_start=0.3,
        finetune_mse_weight_end=0.1,
        launch_mode="run",
        notes="如果 v2 仍然 1/3，则进一步放大 replay，继续缩短 schedule 压制漂移。",
    ),
    "hardpos_replay_v4": ExperimentSpec(
        name="hardpos_replay_v4",
        exp_dir="exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_from119_v4",
        data_dir="data_xlxl_0327_ctc_v1_clean_hardpos_replay_v4",
        replay_factor=20,
        finetune_epochs=10,
        finetune_lr=2e-5,
        head_lr_ratio=0.05,
        finetune_mse_weight_start=0.3,
        finetune_mse_weight_end=0.1,
        launch_mode="run",
        notes="若 v3 仍无法恢复 hardcase，则再缩短 schedule、进一步压低 lr 与 head 漂移。",
    ),
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


def append_history(event: Dict) -> None:
    ensure_dirs()
    payload = {"ts": now_str(), **event}
    with HISTORY_JSONL.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_state(state: Dict) -> None:
    ensure_dirs()
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_shell(cmd: str, *, log_path: Path, cwd: Path = S0_DIR) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n[{now_str()}] CMD: {cmd}\n")
        fout.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            executable="/bin/bash",
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\nsee log: {log_path}")
    return proc


def run_json_cmd(cmd: str, *, log_path: Path, cwd: Path = S0_DIR) -> Dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n[{now_str()}] CMD: {cmd}\n")
        fout.write(proc.stdout)
        if proc.stderr:
            fout.write("\n[stderr]\n")
            fout.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\nsee log: {log_path}")
    return json.loads(proc.stdout)


def tmux_has_session(session_name: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    return proc.returncode == 0


def wait_external_session(session_name: str, *, poll_sec: int = 120) -> None:
    while tmux_has_session(session_name):
        append_history({"type": "wait_session", "session": session_name})
        time.sleep(poll_sec)


def parse_stats_file(path: Path) -> List[StatRow]:
    rows: List[StatRow] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append(StatRow(threshold=float(parts[0]), fa_per_hour=float(parts[1]), frr=float(parts[2])))
    return rows


def pick_best_row(rows: List[StatRow], target_fa: float = 1.0) -> Optional[StatRow]:
    if not rows:
        return None
    candidates = [r for r in rows if r.fa_per_hour <= target_fa]
    if candidates:
        return min(candidates, key=lambda r: (r.frr, r.fa_per_hour, r.threshold))
    return min(rows, key=lambda r: (r.frr, r.fa_per_hour, r.threshold))


def keyword_from_stats_filename(filename: str) -> str:
    return filename[len("stats.") : -len(".txt")].replace("_", " ")


def collect_metrics(exp_dir: str, test_id: str) -> Dict[str, Dict]:
    test_name = test_id if test_id.startswith("test_") else f"test_{test_id}"
    test_dir = S0_DIR / exp_dir / test_name
    if not test_dir.exists():
        raise FileNotFoundError(f"test dir not found: {test_dir}")
    results: Dict[str, Dict] = {}
    for path in sorted(test_dir.glob("stats.*.txt")):
        keyword = keyword_from_stats_filename(path.name)
        best = pick_best_row(parse_stats_file(path))
        if best is None:
            continue
        results[keyword] = {
            "threshold": best.threshold,
            "accuracy": round((1.0 - best.frr) * 100.0, 2),
            "frr": round(best.frr * 100.0, 2),
            "fa_per_hour": round(best.fa_per_hour, 2),
        }
    return results


def build_dataset(spec: ExperimentSpec, *, log_path: Path) -> Dict:
    cmd = (
        "python tools/build_xiaolei_hard_positive_replay_data.py "
        "--source-data-dir data_xlxl_0327_ctc_v1_clean "
        "--teacher-score exp/fsmn_ctc_xlxl_top20_weight_surgery/train_159/score.txt "
        "--student-score exp/fsmn_ctc_xlxl_distill_199k/train_119/score.txt "
        f"--keyword 小雷小雷 --replay-factor {spec.replay_factor} "
        f"--output-dir {spec.data_dir}"
    )
    return run_json_cmd(cmd, log_path=log_path)


def launch_experiment(spec: ExperimentSpec, *, log_path: Path) -> None:
    cmd = (
        "bash run_distill.sh "
        f"--data_dir {spec.data_dir} "
        "--teacher_checkpoint exp/fsmn_ctc_xlxl_top20_weight_surgery/159.pt "
        "--student_config conf/fsmn_ctc_student_mini.yaml "
        "--dict_dir dict_top20_xlxl "
        "--num_keywords 20 "
        '--keywords "小雷小雷,小雷快拍" '
        f"--target_exp_dir {spec.exp_dir} "
        "--gpus 0,1,2,3 "
        "--checkpoint exp/fsmn_ctc_xlxl_distill_199k/119.pt "
        "--align_epochs 120 "
        f"--finetune_epochs {spec.finetune_epochs} "
        f"--finetune_lr {spec.finetune_lr:.8f} "
        f"--head_lr_ratio {spec.head_lr_ratio} "
        f"--finetune_mse_weight_start {spec.finetune_mse_weight_start} "
        f"--finetune_mse_weight_end {spec.finetune_mse_weight_end} "
        "2 3"
    )
    run_shell(cmd, log_path=log_path)


def run_streaming_eval(spec: ExperimentSpec, *, log_path: Path) -> Dict[str, Dict]:
    cmd = (
        "python evaluate_infer_wav.py "
        f"--model_dir {spec.exp_dir} "
        "--checkpoint_name avg_30.pt "
        "--dict_dir dict_top20_xlxl "
        '--keywords "小雷小雷,小雷快拍" '
        "--test_data data_xlxl_0327_ctc_v1_clean/test/data.list "
        "--gpus 0 --streaming --chunk_ms 300 "
        "--result_test_id test_infer_stream_avg30_chunk300"
    )
    run_shell(cmd, log_path=log_path)
    return collect_metrics(spec.exp_dir, "test_infer_stream_avg30_chunk300")


def run_hardcases(spec: ExperimentSpec, *, log_path: Path) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for wav in HARDCASES:
        cmd = (
            "python tools/analyze_repeated_hardcase_streaming.py "
            f"--wav {wav} "
            f"--student-exp-dir {spec.exp_dir} "
            "--student-test-id avg_30 "
            "--repeat 3 --chunk-ms 300 --gpu 0"
        )
        data = run_json_cmd(cmd, log_path=log_path)
        wav_key = Path(wav).stem
        results[wav_key] = {
            "student_num_activations": data["models"]["student"]["streaming"]["num_activations"],
            "student_activations_per_segment": data["models"]["student"]["streaming"]["activations_per_segment"],
            "teacher_num_activations": data["models"]["teacher"]["streaming"]["num_activations"],
            "teacher_activations_per_segment": data["models"]["teacher"]["streaming"]["activations_per_segment"],
            "plots": data.get("plots", {}),
        }
    return results


def experiment_score(result: Dict) -> tuple:
    static_metrics = result["static_metrics"]
    xlxl = static_metrics.get("小 雷 小 雷", static_metrics.get("小雷小雷", {})).get("accuracy", -1.0)
    kp = static_metrics.get("小 雷 快 拍", static_metrics.get("小雷快拍", {})).get("accuracy", -1.0)
    hard_min = min(v["student_num_activations"] for v in result["hardcases"].values())
    hard_sum = sum(v["student_num_activations"] for v in result["hardcases"].values())
    return (xlxl, kp, hard_min, hard_sum)


def decide_next(result: Dict, prior_results: List[Dict]) -> Optional[ExperimentSpec]:
    static_metrics = result["static_metrics"]
    xlxl = static_metrics.get("小 雷 小 雷", static_metrics.get("小雷小雷", {})).get("accuracy", -1.0)
    kp = static_metrics.get("小 雷 快 拍", static_metrics.get("小雷快拍", {})).get("accuracy", -1.0)
    hard_min = min(v["student_num_activations"] for v in result["hardcases"].values())

    if xlxl >= BEST_STUDENT["小雷小雷"] and kp >= BEST_STUDENT["小雷快拍"] and hard_min >= 2:
        return None

    if result["name"] == "hardpos_replay_v2":
        return EXPERIMENTS["hardpos_replay_v3"]

    if result["name"] == "hardpos_replay_v3":
        return EXPERIMENTS["hardpos_replay_v4"]

    return None


def ensure_experiment_doc() -> None:
    if EXPERIMENT_DOC.exists():
        return
    EXPERIMENT_DOC.write_text(
        "# 20260422 199K 自治实验记录\n\n"
        "这份文档记录 199K 学生在 2026-04-22 当天的自治实验队列执行过程。\n\n"
        "当前策略：先沿 hard positive replay 做有限队列搜索，重点看三道 gate：\n\n"
        "- 原始 test split\n"
        "- 全量 streaming\n"
        "- 两条 repeated hard case (`0031` / `0021`)\n\n",
        encoding="utf-8",
    )


def ensure_plan_doc() -> None:
    if PLAN_DOC.exists():
        return
    PLAN_DOC.write_text(
        "# 20260422 199K 自治实验队列计划\n\n"
        "## 1. 目标\n\n"
        "今天开始把 199K 学生的实验流程收口成一个本地自治队列：\n\n"
        "- 自动等待当前实验结束\n"
        "- 自动补 static / streaming / repeated hard case 评测\n"
        "- 自动写结果记录\n"
        "- 自动决定下一条实验\n\n"
        "## 2. 队列\n\n"
        "1. `hardpos_replay_v2`\n"
        "   - `replay_factor=10`\n"
        "   - `finetune_epochs=30`\n"
        "   - `finetune_lr=5e-5`\n"
        "2. `hardpos_replay_v3`\n"
        "   - `replay_factor=20`\n"
        "   - `finetune_epochs=15`\n"
        "   - `finetune_lr=3e-5`\n"
        "3. `hardpos_replay_v4`\n"
        "   - `replay_factor=20`\n"
        "   - `finetune_epochs=10`\n"
        "   - `finetune_lr=2e-5`\n"
        "   - `head_lr_ratio=0.05`\n\n"
        "## 3. Gate\n\n"
        "- 原始 `test split` 优先看 `小雷小雷`\n"
        "- 全量 streaming 不能比静态更差\n"
        "- 两条 repeated hard case 至少要恢复到旧学生 `119.pt` 的 `2/3`\n\n"
        "## 4. 停止条件\n\n"
        "- 如果静态精度追平 `119.pt` 且 repeated hard case 恢复到 `2/3`，本轮队列停止\n"
        "- 如果 `v4` 仍然无法恢复 hardcase，则下一步改做样本加权，不再继续堆 replay factor\n",
        encoding="utf-8",
    )


def append_experiment_markdown(result: Dict, decision: Optional[ExperimentSpec]) -> None:
    ensure_experiment_doc()
    static_metrics = result["static_metrics"]
    streaming_metrics = result["streaming_metrics"]
    xlxl = static_metrics.get("小 雷 小 雷", static_metrics.get("小雷小雷", {}))
    kp = static_metrics.get("小 雷 快 拍", static_metrics.get("小雷快拍", {}))

    lines = [
        f"## {now_str()} {result['name']}",
        "",
        "### 配置",
        f"- `exp_dir`: `{result['exp_dir']}`",
        f"- `replay_factor`: `{result['dataset_summary'].get('replay_factor', 'N/A')}`",
        f"- `hard_positive_count`: `{result['dataset_summary'].get('hard_positive_count', 'N/A')}`",
        f"- `added_rows`: `{result['dataset_summary'].get('added_rows', 'N/A')}`",
        f"- `finetune_epochs`: `{result['spec']['finetune_epochs']}`",
        f"- `finetune_lr`: `{result['spec']['finetune_lr']}`",
        "",
        "### 原始 test split",
        f"- `小雷小雷`: `{xlxl.get('accuracy', 'N/A')}%`, `fa/h={xlxl.get('fa_per_hour', 'N/A')}`",
        f"- `小雷快拍`: `{kp.get('accuracy', 'N/A')}%`, `fa/h={kp.get('fa_per_hour', 'N/A')}`",
        "",
        "### 全量 streaming",
        f"- `小雷小雷`: `{streaming_metrics.get('小 雷 小 雷', streaming_metrics.get('小雷小雷', {})).get('accuracy', 'N/A')}%`",
        f"- `小雷快拍`: `{streaming_metrics.get('小 雷 快 拍', streaming_metrics.get('小雷快拍', {})).get('accuracy', 'N/A')}%`",
        "",
        "### repeated hard case",
    ]
    for wav_key, hc in result["hardcases"].items():
        lines.append(
            f"- `{wav_key}`: 学生 `{hc['student_num_activations']}/3`, "
            f"segments={hc['student_activations_per_segment']}; "
            f"老师 `{hc['teacher_num_activations']}/3`"
        )

    if decision is None:
        lines += [
            "",
            "### 决策",
            "- 本轮队列在这里停止。",
        ]
    else:
        lines += [
            "",
            "### 决策",
            f"- 下一轮转到 `{decision.name}`",
            f"- 原因：{decision.notes}",
        ]

    with EXPERIMENT_DOC.open("a", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n\n")


def evaluate_experiment(spec: ExperimentSpec) -> Dict:
    log_path = LOG_ROOT / f"{spec.name}.log"
    static_metrics = collect_metrics(spec.exp_dir, "avg_30")
    streaming_metrics = run_streaming_eval(spec, log_path=log_path)
    hardcases = run_hardcases(spec, log_path=log_path)
    dataset_summary_path = S0_DIR / spec.data_dir / "hard_positive_replay_summary.json"
    dataset_summary = {}
    if dataset_summary_path.exists():
        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    return {
        "name": spec.name,
        "exp_dir": spec.exp_dir,
        "spec": asdict(spec),
        "dataset_summary": dataset_summary,
        "static_metrics": static_metrics,
        "streaming_metrics": streaming_metrics,
        "hardcases": hardcases,
    }


def main() -> int:
    ensure_dirs()
    ensure_plan_doc()
    ensure_experiment_doc()

    ordered = [EXPERIMENTS["hardpos_replay_v2"], EXPERIMENTS["hardpos_replay_v3"], EXPERIMENTS["hardpos_replay_v4"]]
    results: List[Dict] = []
    current = ordered[0]

    while current is not None:
        log_path = LOG_ROOT / f"{current.name}.log"
        append_history({"type": "start_experiment", "experiment": current.name})
        save_state({"ts": now_str(), "current_experiment": current.name, "phase": "running", "results": results})

        if current.launch_mode == "external_wait":
            if current.external_session and tmux_has_session(current.external_session):
                wait_external_session(current.external_session)
            else:
                append_history({"type": "external_session_missing", "experiment": current.name, "session": current.external_session})
        else:
            dataset_summary = build_dataset(current, log_path=log_path)
            append_history({"type": "dataset_ready", "experiment": current.name, "dataset_summary": dataset_summary})
            launch_experiment(current, log_path=log_path)

        result = evaluate_experiment(current)
        results.append(result)
        next_exp = decide_next(result, results[:-1])
        append_experiment_markdown(result, next_exp)
        append_history({"type": "finish_experiment", "experiment": current.name, "result": result, "next": next_exp.name if next_exp else None})
        save_state(
            {
                "ts": now_str(),
                "current_experiment": next_exp.name if next_exp else None,
                "phase": "completed" if next_exp is None else "queueing_next",
                "results": results,
            }
        )
        current = next_exp

    return 0


if __name__ == "__main__":
    sys.exit(main())
