# dev/test 正样本与 ASR 不一致清单

- 总数: `357`
- 按 split: `{'dev': 182, 'test': 175}`
- 按状态: `{'reject_positive_miss': 176, 'ambiguous_other_target': 14, 'ambiguous_extra_context': 103, 'ambiguous_partial_keyword': 62, 'ambiguous_target_overlap': 2}`
- JSONL: [`xlxl_dev_test_positive_mismatch.jsonl`](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/codex_artifacts/xlxl_dev_test_positive_mismatch.jsonl)
- CSV: [`xlxl_dev_test_positive_mismatch.csv`](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/codex_artifacts/xlxl_dev_test_positive_mismatch.csv)

## 按目标词拆分

- `小 雷 小 雷`: `{'reject_positive_miss': 169, 'ambiguous_other_target': 14, 'ambiguous_extra_context': 77, 'ambiguous_partial_keyword': 53, 'ambiguous_target_overlap': 2}`
- `小 雷 快 拍`: `{'ambiguous_partial_keyword': 9, 'ambiguous_extra_context': 26, 'reject_positive_miss': 7}`
