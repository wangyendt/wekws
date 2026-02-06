# exp 实验简述

来源：各实验 `logs/run_stage_2_2_*.log` 的启动参数；FAR/FRR 来自  
`python examples/hi_xiaowen/s0/analyze_exp_test_stats.py --test-id 2`（`fa<=1.0`，单位：次/小时）。

## 实验差异（按日志与词表）

- `fsmn_ctc_baseline_4gpus`：dict `./dict`，num_keywords=2599，默认参数（未显式设置 `checkpoint_strict`/`dict_auto_build`）。
  - test_2：嗨小问 FAR 0.93/h，FRR 3.04%；你好问问 FAR 0.41/h，FRR 3.74%。
- `fsmn_ctc_top2598`：dict `dict_top2598`，num_keywords=2598，`dict_auto_build=true`，`checkpoint_strict=false`。
  - test_2：嗨小问 FAR 0.93/h，FRR 98.88%；你好问问 FAR 0.00/h，FRR 100.00%。
- `fsmn_ctc_top2599`：dict `dict_top2599`，num_keywords=2599，`dict_auto_build=true`，`checkpoint_strict=true`。
  - test_2：嗨小问 FAR 0.99/h，FRR 3.90%；你好问问 FAR 0.58/h，FRR 3.29%。
- `fsmn_ctc_top2599_strict_true_clip`：dict `dict_top2599_clip`，num_keywords=2599，`checkpoint_strict=true`。
  - test_2：嗨小问 FAR 0.95/h，FRR 3.34%；你好问问 FAR 0.32/h，FRR 4.28%。
- `fsmn_ctc_top2599_strict_true_clip_remove_sil`：dict `dict_top2599_clip_remove_sil`，num_keywords=2599，`checkpoint_strict=true`。
  - test_2：嗨小问 FAR 0.98/h，FRR 3.23%；你好问问 FAR 0.16/h，FRR 4.26%。
- `fsmn_ctc_top440`：dict `dict_top440`，num_keywords=442，`checkpoint_strict=false`。
  - test_2：嗨小问 FAR 0.99/h，FRR 99.00%；你好问问 FAR 0.86/h，FRR 99.33%。
