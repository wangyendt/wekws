#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def normalize_text(text: str) -> str:
    return "".join(str(text).split())


def load_dataset_stats(test_data: str, keyword: str):
    keyword_norm = normalize_text(keyword)
    pos_total = 0
    neg_total = 0
    neg_hours = 0.0
    for line in Path(test_data).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        txt = normalize_text(obj["txt"])
        if txt == keyword_norm:
            pos_total += 1
        else:
            neg_total += 1
            neg_hours += float(obj["duration"]) / 3600.0
    if pos_total <= 0:
        raise ValueError(f"No positive sample found for keyword={keyword}")
    if neg_total <= 0 or neg_hours <= 0:
        raise ValueError("No negative sample or negative duration found")
    return pos_total, neg_total, neg_hours


def load_stats_rows(stats_file: str, pos_total: int, neg_hours: float):
    rows = []
    for line in Path(stats_file).open(encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        threshold, fa_per_hour, frr = map(float, line.split())
        false_reject = int(round(frr * pos_total))
        recall_count = pos_total - false_reject
        false_alarm_count = int(round(fa_per_hour * neg_hours))
        rows.append({
            "threshold": threshold,
            "fa_per_hour": fa_per_hour,
            "frr": frr,
            "recall_count": recall_count,
            "false_reject_count": false_reject,
            "false_alarm_count": false_alarm_count,
        })
    if not rows:
        raise ValueError(f"No valid row in stats_file={stats_file}")
    rows.sort(key=lambda x: x["threshold"])
    return rows


def summarize_thresholds(rows, thresholds, pos_total, neg_total):
    result = []
    for target in thresholds:
        row = min(rows, key=lambda r: abs(r["threshold"] - target))
        result.append({
            "query_threshold": target,
            "threshold": row["threshold"],
            "recall_count": row["recall_count"],
            "pos_total": pos_total,
            "false_alarm_count": row["false_alarm_count"],
            "neg_total": neg_total,
            "fa_per_hour": row["fa_per_hour"],
            "frr": row["frr"],
        })
    return result


def summarize_fa_budgets(rows, budgets, pos_total, neg_total):
    result = []
    for budget in budgets:
        candidates = [r for r in rows if r["false_alarm_count"] <= budget]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda r: (
                r["recall_count"],
                r["false_alarm_count"],
                -r["threshold"],
            ),
        )
        result.append({
            "fa_budget": budget,
            "threshold": best["threshold"],
            "recall_count": best["recall_count"],
            "pos_total": pos_total,
            "false_alarm_count": best["false_alarm_count"],
            "neg_total": neg_total,
            "fa_per_hour": best["fa_per_hour"],
            "frr": best["frr"],
        })
    return result


def fmt_ratio(n, d):
    return f"{n}/{d}"


def fmt_pct(n, d):
    if d <= 0:
        return "N/A"
    return f"{(100.0 * n / d):.2f}%"


def print_markdown(title, dataset_stats, threshold_rows, budget_rows):
    pos_total, neg_total, neg_hours = dataset_stats
    print(f"## {title}")
    print()
    print(f"- positives: {pos_total}")
    print(f"- negatives: {neg_total}")
    print(f"- negative_hours: {neg_hours:.4f}")
    print()
    if budget_rows:
        print("### FA Budget Table")
        print()
        print("| fa_budget | threshold | recall_count/total | recall | false_alarm_count/total | FRR | FA/h |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for row in budget_rows:
            print(
                f"| {row['fa_budget']} | {row['threshold']:.3f} | {fmt_ratio(row['recall_count'], row['pos_total'])} | {fmt_pct(row['recall_count'], row['pos_total'])} | {fmt_ratio(row['false_alarm_count'], row['neg_total'])} | {row['frr'] * 100:.2f}% | {row['fa_per_hour']:.2f} |"
            )
        print()
    if threshold_rows:
        print("### Fixed Threshold Table")
        print()
        print("| query_threshold | threshold | recall_count/total | recall | false_alarm_count/total | FRR | FA/h |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for row in threshold_rows:
            print(
                f"| {row['query_threshold']:.2f} | {row['threshold']:.3f} | {fmt_ratio(row['recall_count'], row['pos_total'])} | {fmt_pct(row['recall_count'], row['pos_total'])} | {fmt_ratio(row['false_alarm_count'], row['neg_total'])} | {row['frr'] * 100:.2f}% | {row['fa_per_hour']:.2f} |"
            )
        print()


def parse_float_list(text: str):
    if not text:
        return []
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str):
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize external DET stats by threshold sweep or FA budgets"
    )
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--stats_file", required=True)
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--thresholds",
        default="",
        help="comma-separated thresholds, e.g. 0,0.01,0.02",
    )
    parser.add_argument(
        "--fa_counts",
        default="1,2,5,10,20,50",
        help="comma-separated FA count budgets",
    )
    args = parser.parse_args()

    dataset_stats = load_dataset_stats(args.test_data, args.keyword)
    pos_total, neg_total, neg_hours = dataset_stats
    rows = load_stats_rows(args.stats_file, pos_total, neg_hours)

    thresholds = parse_float_list(args.thresholds)
    fa_counts = parse_int_list(args.fa_counts)

    threshold_rows = summarize_thresholds(rows, thresholds, pos_total, neg_total)
    budget_rows = summarize_fa_budgets(rows, fa_counts, pos_total, neg_total)
    title = args.title or Path(args.stats_file).stem
    print_markdown(title, dataset_stats, threshold_rows, budget_rows)


if __name__ == "__main__":
    main()
