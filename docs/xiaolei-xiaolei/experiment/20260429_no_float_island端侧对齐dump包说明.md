# 20260429 no-float-island 端侧对齐 dump 包说明

## 目的

给端侧同事提供一份可逐步对齐的数据包，覆盖从输入 mp3 到 fbank、TFLite 每帧输入输出、每个 op 的中间 tensor、logits/probs、CTC beam search 和最终唤醒 flag。

本次使用模型：

```text
examples/hi_xiaowen/s0/exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/avg_10_stream_native_fullint8_calib200_no_float_island.tflite
```

## 样本

选用 2 秒左右的小雷小雷正样本，并转成 mp3 作为端侧输入：

```text
source wav: /home/xushang/kws_ws/extra_datasets/kws_data_0327/XIAOLEI_XIAOLEI/XIAOLEI_XIAOLEI_389_female_07_low_0034.wav
package mp3: examples/hi_xiaowen/s0/codex_artifacts/xlxl_no_float_island_alignment/audio/input.mp3
```

音频信息：

| 字段 | 值 |
|---|---:|
| sample_rate | 16000 |
| samples | 33408 |
| duration | 2.088s |
| streaming chunk | 300ms |
| raw fbank shape | `[207, 80]` |
| model input shape | `[69, 400]` |

## 结果

两个后处理口径都记录在包里：

| 口径 | keyword | score | start_frame | end_frame | wake_time |
|---|---|---:|---:|---:|---:|
| 首次触发 | 小雷小雷 | 0.9929889659 | 81 | 171 | 1.71s |
| 评测 best decode | 小雷小雷 | 0.9972793064 | 81 | 177 | 1.77s |

说明：

- 单条 `infer_wav_stream.py` 默认遇到首次触发就返回，所以是 `1.71s`。
- 全量 `evaluate_infer_wav.py --streaming` 使用整段 best decode，所以是 `1.77s`。
- dump 包同时保留 `python_first_activation`、`c_first_activation`、`python_decoder_result`、`c_decoder_result`，避免对齐时混淆。

## 在线首次触发全量评测

为了确认端侧实时口径的整体效果，新增 `evaluate_infer_wav.py --streaming_first_trigger`，含义是一条音频在流式过程中首次满足 decoder 触发条件就停止；未触发则记 rejected。

评测命令：

```bash
python evaluate_infer_wav.py --checkpoint exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/avg_10_stream_native_fullint8_calib200_no_float_island.tflite --config exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/config.yaml --dict_dir dict_top20_xlxl --stats_dir exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/test_avg_10 --keywords 小雷小雷,小雷快拍 --test_data data_xlxl_0327_ctc_v1_clean/test/data.list --gpus=-1,-1,-1,-1 --streaming --streaming_first_trigger --chunk_ms 300 --use_c_decoder --result_dir exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/test_infer_stream_avg10_native_fullint8_no_float_island_calib200_chunk300_first_trigger_c --target_fa_per_hour 1.0 --pick_mode legacy --progress_every 1000
```

结果：

| keyword | accuracy | FRR | FA/h | threshold |
|---|---:|---:|---:|---:|
| 小雷小雷 | 98.28% | 1.72% | 0.51 | 0.000 |
| 小雷快拍 | 99.35% | 0.65% | 0.00 | 0.000 |

对比 best decode 口径：

| keyword | best decode | first trigger |
|---|---:|---:|
| 小雷小雷 | 98.28%, FA/h 0.17 | 98.28%, FA/h 0.51 |
| 小雷快拍 | 99.61%, FA/h 0.00 | 99.35%, FA/h 0.00 |

结论：在线首次触发口径下，小雷小雷召回不掉，但误唤醒从 `0.17/h` 升到 `0.51/h`；小雷快拍召回略降。

## 生成命令

```bash
python torch2lite/dump_tflite_alignment_package.py --audio codex_artifacts/xlxl_no_float_island_alignment/audio/xiaolei_positive_short.mp3 --model exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/avg_10_stream_native_fullint8_calib200_no_float_island.tflite --config exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/config.yaml --dict_dir dict_top20_xlxl --stats_dir exp/fsmn_ctc_xlxl_distill_199k_hardpos_replay_head_only_from119_v1/test_avg_10 --keywords 小雷小雷,小雷快拍 --output_dir codex_artifacts/xlxl_no_float_island_alignment --chunk_ms 300 --target_fa_per_hour 1.0 --pick_mode legacy
```

## 包路径

```text
examples/hi_xiaowen/s0/codex_artifacts/xlxl_no_float_island_alignment/
```

包内 `README.md` 是给同事看的入口说明。

核心文件：

| 文件/目录 | 含义 |
|---|---|
| `audio/input.mp3` | 端侧对齐用输入音频 |
| `audio/decoded_pcm_int16.*` | mp3 解码并重采样到 16k 后的 int16 PCM |
| `features/streaming_fbank_80dim_float32.*` | 流式 fbank 输出 |
| `features/model_input_400dim_float32.*` | 实际送入 TFLite 的 400 维特征 |
| `steps/step_XXXX/inputs/` | 每帧模型输入 feature/cache |
| `steps/step_XXXX/outputs/` | 每帧 logits/probs/out_cache |
| `steps/step_XXXX/tensors/` | 每帧每个 TFLite op 的输出 tensor |
| `steps/step_XXXX/tensor_manifest.json` | 每个 tensor 的 shape、dtype、量化参数、路径 |
| `network/*_by_frame.*` | 按帧汇总的 logits/probs/cache |
| `decoder/decoder_trace.jsonl` | 逐帧 beam search trace |
| `decoder/final_result.json` | 最终结果摘要 |

所有 `.bin` 均为 numpy C-order 原始内存，shape/dtype 在同名 `.json` 中。
