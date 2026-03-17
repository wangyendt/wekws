#!/usr/bin/env python3

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import infer_wav as iw
import torch
from torch.utils.data import DataLoader
from wenet.text.char_tokenizer import CharTokenizer

from wekws.dataset.init_dataset import init_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KEYWORDS = "嗨小问,你好问问"


@dataclass
class EvalSummaryRow:
    keyword: str
    threshold: float
    accuracy: float
    frr: float
    fa_per_hour: float
    picked_by: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于 infer_wav.py 的逻辑做全量离线评估，并生成 score.txt / stats.*.txt。"
    )
    parser.add_argument("--test_data", default="data/test/data.list", help="待评估 data.list")
    parser.add_argument("--model", default="s3", help="模型别名，例如 s3 / s1 / v2 / top20")
    parser.add_argument("--checkpoint", default="", help="显式指定 checkpoint/.zip")
    parser.add_argument("--model_dir", default="", help="显式指定实验目录")
    parser.add_argument("--checkpoint_name", default="", help="配合 --model_dir 使用")
    parser.add_argument("--config", default="", help="显式指定 config.yaml")
    parser.add_argument("--dict_dir", default="", help="显式指定 dict 目录")
    parser.add_argument("--stats_dir", default="", help="显式指定已有 stats 目录，仅用于阈值回退")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="关键词列表，逗号分隔")
    parser.add_argument("--threshold_map", default="", help="手动阈值覆盖，例如 嗨小问=0.272,你好问问=0.016")
    parser.add_argument("--target_fa_per_hour", type=float, default=1.0, help="挑选阈值时优先满足的 FA/h 上限")
    parser.add_argument("--pick_mode", choices=["legacy", "recall", "robust"], default="legacy")
    parser.add_argument("--frr_eps", type=float, default=0.001, help="robust 模式下 FRR 容差")
    parser.add_argument("--gpus", default="0", help='GPU 列表，例如 "0" 或 "0,1,2,3"；传 "-1" 表示 CPU')
    parser.add_argument("--result_dir", default="", help="输出目录；默认写到模型目录下 test_infer_<ckpt>")
    parser.add_argument("--result_test_id", default="", help="输出目录名，建议形如 test_infer_399")
    parser.add_argument("--token_file", default="mobvoi_kws_transcription/tokens.txt", help="token 文件")
    parser.add_argument("--lexicon_file", default="mobvoi_kws_transcription/lexicon.txt", help="lexicon 文件")
    parser.add_argument("--window_shift", type=int, default=50, help="compute_det_ctc.py 的 window_shift")
    parser.add_argument("--step", type=float, default=0.001, help="compute_det_ctc.py 的阈值步长")
    parser.add_argument("--max_utts", type=int, default=0, help="只评估前 N 条，用于快速验证；0 表示全量")
    parser.add_argument("--batch_size", type=int, default=256, help="每个 worker 的推理 batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="每个 worker 的 dataloader worker 数")
    parser.add_argument("--prefetch", type=int, default=100, help="每个 worker 的 dataloader prefetch_factor")
    parser.add_argument("--pin_memory", action="store_true", help="是否启用 dataloader pin_memory")
    parser.add_argument("--progress_every", type=int, default=1000, help="每处理多少条打印一次进度")
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进空格数")
    return parser.parse_args()


def parse_gpu_list(raw_gpus: str) -> List[int]:
    parts = [item.strip() for item in raw_gpus.split(",") if item.strip()]
    if not parts:
        return [0]
    return [int(item) for item in parts]


def load_eval_items(test_data: Path, max_utts: int) -> List[Dict]:
    items: List[Dict] = []
    with open(test_data, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_utts > 0 and len(items) >= max_utts:
                break
    if not items:
        raise ValueError(f"评估列表为空: {test_data}")
    return items


def build_infer_namespace(args) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        config=args.config,
        dict_dir=args.dict_dir,
        stats_dir=args.stats_dir,
        keywords=args.keywords,
        threshold_map=args.threshold_map,
        target_fa_per_hour=args.target_fa_per_hour,
        pick_mode=args.pick_mode,
        frr_eps=args.frr_eps,
        gpu=-1,
        disable_threshold=False,
        indent=args.indent,
    )


def sanitize_test_id(name: str) -> str:
    name = name.strip()
    if not name:
        return name
    if name.startswith("test_"):
        return name
    return f"test_{name}"


def make_default_result_dir(args, model_info: Dict[str, Optional[Path]]) -> Path:
    checkpoint_name = model_info["checkpoint"].stem
    test_id = sanitize_test_id(args.result_test_id) if args.result_test_id else f"test_infer_{checkpoint_name}"
    if args.max_utts > 0:
        test_id = f"{test_id}_max{args.max_utts}"
    return model_info["checkpoint"].parent / test_id


def keyword_to_stats_filename(keyword: str) -> str:
    return f"stats.{iw.space_mixed_label(keyword).replace(' ', '_')}.txt"


def score_line_from_decode(key: str, decode_result: Dict[str, object]) -> str:
    keyword = decode_result["candidate_keyword"]
    score = decode_result["candidate_score"]
    if keyword is None or score is None:
        return f"{key} rejected\n"
    return f"{key} detected {keyword} {score:.3f}\n"


def shard_items(items: List[Dict], num_shards: int, shard_id: int) -> List[Dict]:
    return [item for index, item in enumerate(items) if index % num_shards == shard_id]


def ensure_clean_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_labels_by_key(data_list_path: Path) -> Dict[str, Dict]:
    labels: Dict[str, Dict] = {}
    with open(data_list_path, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if key:
                labels[key] = obj
    return labels


def build_test_conf(configs: Dict, batch_size: int) -> Dict:
    test_conf = copy.deepcopy(configs["dataset_conf"])
    test_conf["filter_conf"]["max_length"] = 102400
    test_conf["filter_conf"]["min_length"] = 0
    test_conf["filter_conf"]["token_max_length"] = 10240
    test_conf["filter_conf"]["token_min_length"] = 1
    test_conf["filter_conf"]["min_output_input_ratio"] = 1e-6
    test_conf["filter_conf"]["max_output_input_ratio"] = 1
    test_conf["speed_perturb"] = False
    test_conf["spec_aug"] = False
    test_conf["shuffle"] = False
    feats_type = test_conf.get("feats_type", "fbank")
    test_conf[f"{feats_type}_conf"]["dither"] = 0.0
    test_conf["batch_conf"]["batch_size"] = batch_size
    return test_conf


def build_dataloader(shard_list_path: Path, model_info: Dict[str, Optional[Path]], configs: Dict, args):
    test_conf = build_test_conf(configs, args.batch_size)
    tokenizer = CharTokenizer(
        str(model_info["dict_dir"] / "dict.txt"),
        str(model_info["dict_dir"] / "words.txt"),
        unk="<filler>",
        split_with_space=True,
    )
    dataset = init_dataset(
        data_list_file=str(shard_list_path),
        conf=test_conf,
        tokenizer=tokenizer,
        split="test",
    )
    dataloader_kwargs = {
        "batch_size": None,
        "pin_memory": args.pin_memory,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch
    return DataLoader(dataset, **dataloader_kwargs)


def run_batch_forward(model, feats: torch.Tensor, device: torch.device, is_jit: bool):
    feats = feats.to(device)
    with torch.no_grad():
        if is_jit:
            empty_cache = torch.zeros(0, 0, 0, dtype=torch.float32, device=device)
            logits_raw, _ = model(feats, empty_cache)
        else:
            logits_raw, _ = model(feats)
    return logits_raw.softmax(2).detach().cpu()


def build_result_from_meta(
    meta: Dict,
    model_info: Dict[str, Optional[Path]],
    threshold_map: Dict[str, Optional[float]],
    decode_result: Dict[str, object],
    time_resolution_sec: float,
):
    result = iw.format_result(
        wav_path=Path(meta["wav"]),
        model_info=model_info,
        threshold_map=threshold_map,
        decode_result=decode_result,
        time_resolution_sec=time_resolution_sec,
        disable_threshold=False,
    )
    result["key"] = meta.get("key")
    result["txt"] = meta.get("txt")
    result["duration"] = meta.get("duration")
    return result


def worker_eval(
    worker_id: int,
    gpu_id: int,
    shard_list_path_str: str,
    num_items: int,
    args_dict: Dict[str, object],
    temp_dir_str: str,
):
    args = SimpleNamespace(**args_dict)
    temp_dir = Path(temp_dir_str)
    shard_list_path = Path(shard_list_path_str)
    infer_args = build_infer_namespace(args)
    infer_args.gpu = gpu_id

    keywords = iw.parse_keywords_arg(args.keywords)
    model_info = iw.resolve_model_paths(infer_args)
    configs = iw.load_config(model_info["config"])
    model, device, is_jit = iw.load_model(model_info["checkpoint"], configs, gpu_id)
    threshold_map = iw.load_threshold_map(infer_args, model_info, keywords)
    time_resolution_sec = iw.get_time_resolution_sec(configs)
    keywords_token, keywords_idxset = iw.build_keyword_token_info(keywords, model_info["dict_dir"])
    labels_by_key = load_labels_by_key(shard_list_path)
    dataloader = build_dataloader(shard_list_path, model_info, configs, args)

    score_part = temp_dir / f"score.part{worker_id}.txt"
    jsonl_part = temp_dir / f"results.part{worker_id}.jsonl"

    with open(score_part, "w", encoding="utf8") as score_fout, open(jsonl_part, "w", encoding="utf8") as jsonl_fout:
        processed = 0
        for batch_dict in dataloader:
            keys = batch_dict["keys"]
            feats = batch_dict["feats"]
            lengths = batch_dict["feats_lengths"].to(device)
            probs_batch = run_batch_forward(model, feats, device, is_jit)

            for item_index, key in enumerate(keys):
                utt_len = int(lengths[item_index].item())
                probs_i = probs_batch[item_index][:utt_len]
                decode_result = iw.decode_keyword_hit_with_token_info(
                    probs=probs_i,
                    keywords=keywords,
                    keywords_token=keywords_token,
                    keywords_idxset=keywords_idxset,
                )
                score_fout.write(score_line_from_decode(key, decode_result))
                meta = labels_by_key[key]
                result = build_result_from_meta(
                    meta=meta,
                    model_info=model_info,
                    threshold_map=threshold_map,
                    decode_result=decode_result,
                    time_resolution_sec=time_resolution_sec,
                )
                jsonl_fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1

                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(
                        f"[worker {worker_id}] processed {processed}/{num_items} on gpu={gpu_id}",
                        flush=True,
                    )

    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "score_part": str(score_part),
        "jsonl_part": str(jsonl_part),
        "num_items": num_items,
    }


def merge_text_files(part_paths: List[Path], merged_path: Path):
    with open(merged_path, "w", encoding="utf8") as fout:
        for path in part_paths:
            with open(path, "r", encoding="utf8") as fin:
                shutil.copyfileobj(fin, fout)


def run_compute_det(args, model_info: Dict[str, Optional[Path]], result_dir: Path, score_file: Path):
    cmd = [
        sys.executable,
        "wekws/bin/compute_det_ctc.py",
        "--keywords",
        args.keywords,
        "--test_data",
        str(Path(args.test_data).resolve()),
        "--window_shift",
        str(args.window_shift),
        "--step",
        str(args.step),
        "--score_file",
        str(score_file),
        "--dict",
        str(model_info["dict_dir"]),
        "--token_file",
        str((SCRIPT_DIR / args.token_file).resolve()),
        "--lexicon_file",
        str((SCRIPT_DIR / args.lexicon_file).resolve()),
        "--stats_dir",
        str(result_dir),
    ]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SCRIPT_DIR}:{old_pythonpath}" if old_pythonpath else str(SCRIPT_DIR)
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env, check=True)


def summarize_stats(args, result_dir: Path) -> List[EvalSummaryRow]:
    rows: List[EvalSummaryRow] = []
    keywords = iw.parse_keywords_arg(args.keywords)
    for keyword in keywords:
        stats_file = result_dir / keyword_to_stats_filename(keyword)
        if not stats_file.exists():
            continue
        stat_rows = iw.parse_stats_file(stats_file)
        best = iw.pick_best_row(stat_rows, args.target_fa_per_hour, args.pick_mode, args.frr_eps)
        if best is None:
            continue
        picked_by = args.pick_mode
        if args.pick_mode == "legacy":
            if any(row.fa_per_hour <= args.target_fa_per_hour for row in stat_rows):
                picked_by = f"legacy:fa<=target({args.target_fa_per_hour})"
            else:
                picked_by = "legacy:min_frr_overall"
        elif args.pick_mode == "recall":
            if any(row.fa_per_hour <= args.target_fa_per_hour for row in stat_rows):
                picked_by = f"recall:min_frr_then_fa_close_to_target({args.target_fa_per_hour})"
            else:
                picked_by = "recall:min_frr_overall(no_fa_candidate)"
        elif args.pick_mode == "robust":
            if any(row.fa_per_hour <= args.target_fa_per_hour for row in stat_rows):
                picked_by = f"robust:frr<=min+{args.frr_eps}_then_max_threshold"
            else:
                picked_by = "robust:min_frr_overall(no_fa_candidate)"

        rows.append(
            EvalSummaryRow(
                keyword=keyword,
                threshold=best.threshold,
                accuracy=1.0 - best.frr,
                frr=best.frr,
                fa_per_hour=best.fa_per_hour,
                picked_by=picked_by,
            )
        )
    return rows


def write_shard_lists(items: List[Dict], num_shards: int, temp_dir: Path) -> List[Path]:
    shard_paths: List[Path] = []
    shard_rows: List[List[Dict]] = [[] for _ in range(num_shards)]
    for index, item in enumerate(items):
        shard_rows[index % num_shards].append(item)

    for shard_id, rows in enumerate(shard_rows):
        shard_path = temp_dir / f"data.part{shard_id}.list"
        with open(shard_path, "w", encoding="utf8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        shard_paths.append(shard_path)
    return shard_paths


def write_summary_json(
    args,
    model_info: Dict[str, Optional[Path]],
    result_dir: Path,
    score_file: Path,
    results_jsonl: Path,
    summary_rows: List[EvalSummaryRow],
    items: List[Dict],
):
    payload = {
        "test_data": str(Path(args.test_data).resolve()),
        "num_utts": len(items),
        "model_alias": model_info["alias"],
        "checkpoint": str(model_info["checkpoint"]),
        "config": str(model_info["config"]),
        "dict_dir": str(model_info["dict_dir"]),
        "result_dir": str(result_dir),
        "score_file": str(score_file),
        "results_jsonl": str(results_jsonl),
        "keywords": [
            {
                "keyword": row.keyword,
                "threshold": row.threshold,
                "accuracy": row.accuracy,
                "frr": row.frr,
                "fa_per_hour": row.fa_per_hour,
                "picked_by": row.picked_by,
            }
            for row in summary_rows
        ],
    }
    with open(result_dir / "summary.json", "w", encoding="utf8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=args.indent)


def print_summary(result_dir: Path, summary_rows: List[EvalSummaryRow]):
    print("================================================")
    print("infer_wav 全量评估完成")
    print("================================================")
    print(f"结果目录: {result_dir}")
    for row in summary_rows:
        print(
            f"{row.keyword}\tthreshold={row.threshold:.3f}\t"
            f"accuracy={row.accuracy * 100:.2f}%\t"
            f"frr={row.frr * 100:.2f}%\t"
            f"fa/h={row.fa_per_hour:.2f}\t"
            f"picked_by={row.picked_by}"
        )
    print("================================================")


def print_analyze_hint(result_dir: Path, model_exp_dir: Path):
    try:
        relative = result_dir.resolve().relative_to(model_exp_dir.resolve())
    except ValueError:
        print("结果目录不在模型实验目录下，若要复用 analyze_exp_test_stats.py，请把结果目录放到该实验目录下面。")
        return

    if len(relative.parts) != 1:
        print("结果目录位于实验目录的多级子目录中，analyze_exp_test_stats.py 默认不一定能直接找到，请优先使用一级子目录。")
        return

    test_name = relative.parts[0]
    print("可用下面这条命令再走一次 analyze_exp_test_stats.py 对比：")
    print(
        f'python analyze_exp_test_stats.py --exp-dir "{model_exp_dir}" --test-id "{test_name}"'
    )


def main():
    args = parse_args()
    test_data = Path(args.test_data)
    if not test_data.is_absolute():
        test_data = (SCRIPT_DIR / test_data).resolve()
    if not test_data.exists():
        raise FileNotFoundError(f"找不到 test_data: {test_data}")
    args.test_data = str(test_data)

    infer_args = build_infer_namespace(args)
    model_info = iw.resolve_model_paths(infer_args)

    result_dir = Path(args.result_dir).resolve() if args.result_dir else make_default_result_dir(args, model_info)
    ensure_clean_dir(result_dir)
    temp_dir = result_dir / ".parts"
    ensure_clean_dir(temp_dir)

    items = load_eval_items(test_data, args.max_utts)
    gpu_list = parse_gpu_list(args.gpus)
    shard_paths = write_shard_lists(items, len(gpu_list), temp_dir)
    if len(gpu_list) == 1:
        worker_plan = [(0, gpu_list[0], shard_paths[0], len(items))]
    else:
        worker_plan = []
        for worker_id, gpu_id in enumerate(gpu_list):
            shard = shard_items(items, len(gpu_list), worker_id)
            if shard:
                worker_plan.append((worker_id, gpu_id, shard_paths[worker_id], len(shard)))

    print(f"总样本数: {len(items)}")
    print(f"工作进程数: {len(worker_plan)}")
    print(f"GPU 列表: {gpu_list}")
    print(f"结果目录: {result_dir}")
    print("")

    args_dict = vars(args).copy()
    ctx = get_context("spawn")
    futures = []
    worker_results = []
    with ProcessPoolExecutor(max_workers=len(worker_plan), mp_context=ctx) as executor:
        for worker_id, gpu_id, shard_path, shard_size in worker_plan:
            futures.append(
                executor.submit(
                    worker_eval,
                    worker_id,
                    gpu_id,
                    str(shard_path),
                    shard_size,
                    args_dict,
                    str(temp_dir),
                )
            )
        for future in futures:
            worker_results.append(future.result())

    worker_results.sort(key=lambda item: item["worker_id"])
    score_parts = [Path(item["score_part"]) for item in worker_results]
    jsonl_parts = [Path(item["jsonl_part"]) for item in worker_results]

    score_file = result_dir / "score.txt"
    results_jsonl = result_dir / "results.jsonl"
    merge_text_files(score_parts, score_file)
    merge_text_files(jsonl_parts, results_jsonl)

    run_compute_det(args, model_info, result_dir, score_file)
    summary_rows = summarize_stats(args, result_dir)
    write_summary_json(args, model_info, result_dir, score_file, results_jsonl, summary_rows, items)
    print_summary(result_dir, summary_rows)

    model_exp_dir = model_info["checkpoint"].parent
    print_analyze_hint(result_dir, model_exp_dir)


if __name__ == "__main__":
    main()
