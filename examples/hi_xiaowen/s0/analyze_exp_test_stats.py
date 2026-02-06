#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class StatRow:
    threshold: float
    fa_per_hour: float
    frr: float


def parse_stats_file(path: str) -> List[StatRow]:
    rows: List[StatRow] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            # 跳过注释行（以 # 开头）
            if line.strip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            threshold, fa_per_hour, frr = parts[:3]
            rows.append(
                StatRow(
                    threshold=float(threshold),
                    fa_per_hour=float(fa_per_hour),
                    frr=float(frr),
                )
            )
    return rows


def pick_best_row(rows: List[StatRow], target_fa: float) -> Tuple[Optional[StatRow], str]:
    if not rows:
        return None, "no_data"
    candidates = [r for r in rows if r.fa_per_hour <= target_fa]
    if candidates:
        best = min(candidates, key=lambda r: (r.frr, r.fa_per_hour, r.threshold))
        return best, f"fa<=target({target_fa})"
    best = min(rows, key=lambda r: (r.frr, r.fa_per_hour, r.threshold))
    return best, "min_frr_overall"


def keyword_from_stats_filename(filename: str) -> Optional[str]:
    match = re.match(r"stats\.(.+)\.txt$", filename)
    if not match:
        return None
    return match.group(1).replace("_", " ")


def find_test_dirs(exp_dir: str, test_name: str) -> List[str]:
    results: List[str] = []
    for root, dirs, _ in os.walk(exp_dir):
        if os.path.basename(root) == test_name:
            results.append(root)
    return sorted(results)


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))
    sep = "+".join("-" * (w + 2) for w in col_widths)
    sep = f"+{sep}+"
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    header_line = f"| {header_line} |"
    body_lines = []
    for row in rows:
        line = " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers)))
        body_lines.append(f"| {line} |")
    return "\n".join([sep, header_line, sep] + body_lines + [sep])


def try_generate_stats(
    test_dir: str,
    score_file: str,
    label_file: str,
    keywords: str,
    dict_dir: str,
    token_file: str,
    lexicon_file: str,
    window_shift: int,
    step: float,
) -> bool:
    cmd = [
        "python",
        "wekws/bin/compute_det_ctc.py",
        "--keywords",
        keywords,
        "--test_data",
        label_file,
        "--window_shift",
        str(window_shift),
        "--step",
        str(step),
        "--score_file",
        score_file,
        "--dict",
        dict_dir,
        "--token_file",
        token_file,
        "--lexicon_file",
        lexicon_file,
        "--stats_dir",
        test_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统计 exp 下 test_N 的唤醒词准确率/误检率（从 stats.*.txt 或 score.txt）。"
    )
    parser.add_argument(
        "--exp-dir",
        default="exp",
        help="实验目录，默认 exp",
    )
    parser.add_argument(
        "--test-id",
        type=int,
        default=2,
        help="测试集编号，例如 test_2 中的 2",
    )
    parser.add_argument(
        "--target-fa-per-hour",
        type=float,
        default=1.0,
        help="选择阈值时优先满足的误检/小时上限（例如 1/12 表示 12 小时一次）",
    )
    parser.add_argument(
        "--gen-stats",
        action="store_true",
        help="当缺少 stats.*.txt 时，尝试用 score.txt 生成",
    )
    parser.add_argument(
        "--label-file",
        default="data/test/data.list",
        help="数据标签文件（用于根据 score.txt 生成 stats）",
    )
    parser.add_argument(
        "--keywords",
        default="嗨小问,你好问问",
        help="关键词列表，逗号分隔（用于生成 stats）",
    )
    parser.add_argument(
        "--dict-dir",
        default="dict",
        help="dict 目录（用于生成 stats）",
    )
    parser.add_argument(
        "--token-file",
        default="mobvoi_kws_transcription/tokens.txt",
        help="token 文件（用于生成 stats）",
    )
    parser.add_argument(
        "--lexicon-file",
        default="mobvoi_kws_transcription/lexicon.txt",
        help="lexicon 文件（用于生成 stats）",
    )
    parser.add_argument(
        "--window-shift",
        type=int,
        default=50,
        help="window_shift（用于生成 stats）",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.001,
        help="threshold step（用于生成 stats）",
    )
    args = parser.parse_args()

    test_name = f"test_{args.test_id}"
    exp_dir = args.exp_dir
    test_dirs = find_test_dirs(exp_dir, test_name)

    if not test_dirs:
        print(f"[WARN] 未找到任何目录: {exp_dir}/**/{test_name}")
        return 1

    headers = [
        "exp_dir",
        "test_dir",
        "keyword",
        "threshold",
        "accuracy",
        "frr",
        "fa_per_hour",
        "picked_by",
    ]
    table_rows: List[List[str]] = []
    total_rows = 0
    missing_stats = 0

    for test_dir in test_dirs:
        stats_files = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.startswith("stats.") and f.endswith(".txt")
        ]

        if not stats_files and args.gen_stats:
            score_file = os.path.join(test_dir, "score.txt")
            if os.path.isfile(score_file) and os.path.isfile(args.label_file):
                ok = try_generate_stats(
                    test_dir=test_dir,
                    score_file=score_file,
                    label_file=args.label_file,
                    keywords=args.keywords,
                    dict_dir=args.dict_dir,
                    token_file=args.token_file,
                    lexicon_file=args.lexicon_file,
                    window_shift=args.window_shift,
                    step=args.step,
                )
                if ok:
                    stats_files = [
                        os.path.join(test_dir, f)
                        for f in os.listdir(test_dir)
                        if f.startswith("stats.") and f.endswith(".txt")
                    ]

        if not stats_files:
            missing_stats += 1
            continue

        for stats_file in sorted(stats_files):
            keyword = keyword_from_stats_filename(os.path.basename(stats_file))
            if not keyword:
                continue
            rows = parse_stats_file(stats_file)
            best, picked_by = pick_best_row(rows, args.target_fa_per_hour)
            if not best:
                continue
            accuracy = 1.0 - best.frr
            table_rows.append(
                [
                    exp_dir,
                    test_dir,
                    keyword,
                    f"{best.threshold:.3f}",
                    fmt_percent(accuracy),
                    fmt_percent(best.frr),
                    fmt_number(best.fa_per_hour, 2),
                    picked_by,
                ]
            )
            total_rows += 1

    table = render_table(headers, table_rows)
    if table:
        print(table)
    print(f"[INFO] 统计完成: {total_rows} 行; 缺少 stats 的 test_dir: {missing_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
