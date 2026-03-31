# 20260331 dev/test 非 clean 样本清单

## 输出文件

- 总表 JSONL: `/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/codex_artifacts/xlxl_dev_test_nonclean_all.jsonl`
- 总表 CSV: `/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/codex_artifacts/xlxl_dev_test_nonclean_all.csv`
- 汇总 JSON: `/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/codex_artifacts/xlxl_dev_test_nonclean_all_summary.json`

## 范围

统计对象为 `dev/test` 两个 split 中所有 `screen_status != clean_exact/clean_negative` 的条目，覆盖：

- 正样本异常
- 负样本异常
- 音频读取失败

## 字段说明

- `split`: `dev` 或 `test`
- `bucket`: `ambiguous` 或 `reject`
- `status`: 机器筛洗状态
- `reason`: 对该状态的中文解释
- `key`: 样本 key
- `wav`: 音频路径
- `duration`: 时长（秒）
- `label_text`: 原始标注文本
- `asr_text`: ASR 转写文本
- `timestamp`: ASR 时间戳
- `asr_error`: 读取失败时的错误信息

## 统计

- 总数: `399`
- `dev`: `203`
- `test`: `196`
- `ambiguous`: `205`
- `reject`: `194`

按状态：

- `reject_positive_miss`: `176`
- `ambiguous_extra_context`: `103`
- `ambiguous_partial_keyword`: `62`
- `ambiguous_negative_partial_keyword`: `16`
- `ambiguous_other_target`: `14`
- `reject_load_error`: `11`
- `ambiguous_negative_target_overlap`: `8`
- `reject_negative_exact_target`: `7`
- `ambiguous_target_overlap`: `2`

## 使用建议

- `reject_positive_miss`: 优先从当前 `dev/test` 剔除
- `ambiguous_*`: 进入人工复核列表
- `reject_negative_exact_target`: 从负样本中移除并人工确认
- `reject_load_error`: 检查权限或文件损坏，暂时剔除
