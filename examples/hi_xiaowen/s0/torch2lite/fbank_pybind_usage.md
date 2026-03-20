# torch2lite fbank / pybind 使用说明

## 1. `fft_size` / `n_fft` 在 C 接口和 pybind 里怎么对应

现在 `fbank.cc` 和 pybind 都支持 `fft_size`。

- 在 [fbank.h](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank.h#L68) 里，`FbankConfig.fft_size` 是可选字段。
- 在 [fbank.cc](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank.cc#L38) 里，`resolve_fft_size(...)` 的逻辑是：
  - 如果 `config->fft_size > 0`，就用显式指定的 `fft_size`
  - 否则如果 `round_to_power_of_two = 1`，就用 `frame_length_samples` 向上取 2 的幂
  - 否则直接用 `frame_length_samples`
- 在 [fbank_pybind.cc](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank_pybind.cc#L39) 里，pybind 现在也暴露了 `fft_size` 参数；默认值仍然是 `0`，也就是继续走自动解析路径。

所以现在 pybind 有两种用法：

- `fft_size=0`
  - 自动解析 FFT size
  - 这是当前最接近 `torchaudio.compliance.kaldi.fbank(...)` / Kaldi 默认行为的路径
- `fft_size>0`
  - 显式指定 FFT size
  - 更接近一些工程里手动传 `n_fft` 的习惯

以当前默认参数为例：

- `sample_rate = 16000`
- `frame_length = 25 ms`
- `frame_length_samples = 400`
- `round_to_power_of_two = True`
- `fft_size = 0`

所以最终实际使用的是：

- `fft_size = 512`

你可以通过 pybind 对象的 `fft_size` 属性看到最终解析后的值。

需要注意的是，当前 pybind 是按 `FBANK_USE_PRECOMPUTED_TABLES` 编译的，所以如果你显式传 `fft_size`，它仍然必须和固定表配置匹配，否则初始化会失败。

## 2. pybind 是怎么 build 的

Python 入口在 [fbank_pybind.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank_pybind.py)。

它不是 CMake 预编译，而是首次 import 时通过 `torch.utils.cpp_extension.load(...)` 即时编译：

- 源文件：
  - [fbank_pybind.cc](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank_pybind.cc)
  - [fbank.cc](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank.cc)
- 编译宏：
  - `-DFBANK_USE_PRECOMPUTED_TABLES`
- 构建目录：
  - [`.fbank_pybind_build`](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/.fbank_pybind_build)

也就是说，下面这些脚本第一次运行时都会自动触发编译：

- [compare_fbank.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/compare_fbank.py)
- [evaluate_fbank_pybind.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/evaluate_fbank_pybind.py)

## 3. compare / evaluate 脚本怎么用

### 3.1 `compare_fbank.py`

文件：

- [compare_fbank.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/compare_fbank.py)

作用：

- 对比 `torchaudio.compliance.kaldi.fbank` 和 `torch2lite` pybind fbank
- 同时比较：
  - 单帧接口
  - 整段 batch 接口
  - `float` 输入
  - `int16` 输入

常用命令：

```bash
cd /home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite
PYTHONPATH=/home/wayne/work/code/project/ffalcon/wekws_baseline python compare_fbank.py
```

可调参数：

- `--sample_rate`
- `--num_samples`
- `--num_mel_bins`
- `--frame_length`
- `--frame_shift`
- `--random_seed`
- `--single_frame_trials`
- `--indent`

说明：

- 当前脚本没有单独暴露 `--fft_size` 参数
- 默认测的是 `fft_size=0` 这条路径，也就是按 `frame_length + round_to_power_of_two` 自动解析，当前默认配置下实际为 `512`

示例：

```bash
PYTHONPATH=/home/wayne/work/code/project/ffalcon/wekws_baseline \
python compare_fbank.py \
  --sample_rate 16000 \
  --num_samples 32000 \
  --frame_length 25 \
  --frame_shift 10 \
  --single_frame_trials 64
```

输出内容：

- `config`
- `sampled_frame_indices`
- `single_frame.kaldi_vs_pybind_float`
- `single_frame.kaldi_vs_pybind_int16`
- `batch.kaldi_vs_pybind_float`
- `batch.kaldi_vs_pybind_int16`

误差指标包括：

- `max_abs`
- `mean_abs`
- `median_abs`
- `p95_abs`
- `p99_abs`
- `rmse`

### 3.2 `evaluate_fbank_pybind.py`

文件：

- [evaluate_fbank_pybind.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/evaluate_fbank_pybind.py)

作用：

- 不再和 Kaldi 对比
- 专门检查 pybind 内部不同接口之间是否自洽
- 对比：
  - `single frame`
  - `batch extract`
  - `streaming accept`
  - `float` 和 `int16`

常用命令：

```bash
cd /home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite
PYTHONPATH=/home/wayne/work/code/project/ffalcon/wekws_baseline python evaluate_fbank_pybind.py
```

示例：

```bash
PYTHONPATH=/home/wayne/work/code/project/ffalcon/wekws_baseline \
python evaluate_fbank_pybind.py \
  --num_samples 48000 \
  --single_frame_trials 64 \
  --stream_min_update_frames 1 \
  --stream_max_update_frames 5
```

输出重点：

- `float.single_vs_batch`
- `float.single_vs_stream`
- `float.batch_vs_stream`
- `int16.single_vs_batch`
- `int16.single_vs_stream`
- `int16.batch_vs_stream`
- `stream_updates`

说明：

- 当前脚本同样没有单独暴露 `--fft_size` 参数
- 默认也是测 `fft_size=0 -> 自动解析` 这条路径

这个脚本适合检查流式接口有没有边界错位、缓存错位、或 int16/float 分支不一致。

## 3.3 当前实测结果

下面结果是在 2026-03-20 本地执行得到的，脚本默认配置为：

- `sample_rate=16000`
- `num_samples=32000`
- `num_mel_bins=80`
- `frame_length=25ms`
- `frame_shift=10ms`
- `fft_size=512`
- `random_seed=0`

### Kaldi vs pybind

来源脚本：

- [compare_fbank.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/compare_fbank.py)

`float` 和 `int16` 两组结果在这次测试里完全一样。

单帧对比：

- `kaldi_vs_pybind_float`: `max_abs=9.4414e-05`, `mean_abs=4.1805e-06`, `p99_abs=2.0981e-05`, `rmse=6.5405e-06`
- `kaldi_vs_pybind_int16`: `max_abs=9.4414e-05`, `mean_abs=4.1805e-06`, `p99_abs=2.0981e-05`, `rmse=6.5405e-06`

整段 batch 对比：

- `kaldi_vs_pybind_float`: `max_abs=3.8433e-04`, `mean_abs=4.1660e-06`, `p99_abs=2.0981e-05`, `rmse=7.0098e-06`
- `kaldi_vs_pybind_int16`: `max_abs=3.8433e-04`, `mean_abs=4.1660e-06`, `p99_abs=2.0981e-05`, `rmse=7.0098e-06`

这说明当前 pybind 实现和 `torchaudio.compliance.kaldi.fbank` 已经非常接近，可以作为替换候选。

### pybind 自身接口一致性

来源脚本：

- [evaluate_fbank_pybind.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/evaluate_fbank_pybind.py)

这个脚本的目的不是和 Kaldi 比，而是验证 pybind 内部 6 组方法是否彼此一致：

`float` 组：

- `process_frame_float` vs `extract_float`
- `process_frame_float` vs `StreamingFbankExtractor.accept_float`
- `extract_float` vs `StreamingFbankExtractor.accept_float`

`int16` 组：

- `process_frame_int16` vs `extract_int16`
- `process_frame_int16` vs `StreamingFbankExtractor.accept_int16`
- `extract_int16` vs `StreamingFbankExtractor.accept_int16`

这次实测结果：

`float` 组：

- `single_vs_batch`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`
- `single_vs_stream`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`
- `batch_vs_stream`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`

`int16` 组：

- `single_vs_batch`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`
- `single_vs_stream`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`
- `batch_vs_stream`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`

也就是说，在当前随机样本和默认参数下：

- pybind 的单帧接口、整段接口、流式接口彼此完全一致
- float 分支和 int16 分支在各自组内也完全一致

## 4. Python pybind 接口

Python 包装文件：

- [fbank_pybind.py](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank_pybind.py)

底层绑定文件：

- [fbank_pybind.cc](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank_pybind.cc)

### 4.1 `FbankExtractor`

构造参数：

- `num_mel_bins=80`
- `frame_length=25.0`
- `frame_shift=10.0`
- `dither=0.0`
- `energy_floor=0.0`
- `sample_frequency=16000.0`
- `low_freq=20.0`
- `high_freq=0.0`
- `preemphasis_coefficient=0.97`
- `remove_dc_offset=True`
- `round_to_power_of_two=True`
- `snip_edges=True`
- `fft_size=0`

属性：

- `frame_length`
  - 单位：sample
- `frame_shift`
  - 单位：sample
- `fft_size`
  - 最终生效的 FFT size，可能来自自动解析，也可能来自显式传参
- `num_mel_bins`
- `work_buffer_bytes`

方法：

- `reset()`
  - 重置 extractor 状态
- `num_frames(num_samples)`
  - 根据输入样本数估算可产生多少帧
- `process_frame_float(waveform)`
  - 输入一帧 float waveform，长度必须等于 `frame_length`
  - 输出 shape 是 `(1, num_mel_bins)`
- `process_frame_int16(waveform)`
  - 输入一帧 int16 waveform，长度必须等于 `frame_length`
  - 输出 shape 是 `(1, num_mel_bins)`
- `extract_float(waveform, num_samples=-1, max_frames=-1)`
  - 对整段 float waveform 提取 fbank
  - 输出 shape 是 `(num_frames, num_mel_bins)`
- `extract_int16(waveform, num_samples=-1, max_frames=-1)`
  - 对整段 int16 waveform 提取 fbank
  - 输出 shape 是 `(num_frames, num_mel_bins)`

### 4.2 `StreamingFbankExtractor`

构造参数和 `FbankExtractor` 一样。

属性：

- `pending_samples`
  - 当前流式缓存里还没凑够下一帧的样本数
- `frame_length`
- `frame_shift`
- `fft_size`
- `num_mel_bins`

方法：

- `reset()`
  - 清空流式状态
- `accept_float(waveform, num_samples=-1, max_frames=-1)`
  - 向流式前端喂一段 float waveform
  - 返回本次新产生的若干帧特征，shape `(N, num_mel_bins)`
- `accept_int16(waveform, num_samples=-1, max_frames=-1)`
  - 向流式前端喂一段 int16 waveform
  - 返回本次新产生的若干帧特征，shape `(N, num_mel_bins)`

### 4.3 Python 使用示例

#### 整段提特征

```python
import torch
from examples.hi_xiaowen.s0.torch2lite import fbank_pybind

waveform = torch.randn(16000, dtype=torch.float32)
extractor = fbank_pybind.FbankExtractor(
    num_mel_bins=80,
    frame_length=25.0,
    frame_shift=10.0,
    sample_frequency=16000.0,
    fft_size=0,
)
feats = extractor.extract_float(waveform)
print(feats.shape, extractor.fft_size)
```

#### 流式提特征

```python
import torch
from examples.hi_xiaowen.s0.torch2lite import fbank_pybind

stream = fbank_pybind.StreamingFbankExtractor(
    num_mel_bins=80,
    frame_length=25.0,
    frame_shift=10.0,
    sample_frequency=16000.0,
    fft_size=0,
)

chunk1 = torch.randn(800, dtype=torch.float32)
chunk2 = torch.randn(1200, dtype=torch.float32)

feat1 = stream.accept_float(chunk1)
feat2 = stream.accept_float(chunk2)
print(feat1.shape, feat2.shape, stream.pending_samples)
```

## 5. C 接口功能总表

头文件：

- [fbank.h](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank.h)

实现文件：

- [fbank.cc](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank.cc)

### 5.1 数据结构

- `SparseMelFilter`
  - 稀疏 mel 滤波器
  - 字段：
    - `start_bin`
    - `end_bin`
    - `weights[FBANK_MAX_FILTER_WIDTH]`

- `FbankConfig`
  - 前端配置
  - 关键字段：
    - `sample_rate`
    - `frame_length_ms`
    - `frame_shift_ms`
    - `num_mel_bins`
    - `fft_size`
    - `dither`
    - `preemph_coeff`
    - `use_energy`
    - `low_freq`
    - `high_freq`
    - `remove_dc_offset`
    - `round_to_power_of_two`
    - `snip_edges`

- `FbankExtractor`
  - 非流式 extractor 状态
  - 内含：
    - config
    - mel filters
    - window
    - frame / fft / power spectrum 工作缓冲
    - `frame_length`
    - `frame_shift`

- `FbankStreamingState`
  - 流式 extractor 状态
  - 内含：
    - 一个 `FbankExtractor`
    - 工作缓冲区
    - float / int16 输入缓存
    - 当前缓存样本数

### 5.2 初始化与资源管理

- `void fbank_init_default(FbankExtractor *extractor)`
  - 用默认配置初始化
  - 默认就是当前项目常用的 `16k / 25ms / 10ms / 80 mel / dither=0 / round_to_power_of_two=1`

- `int32_t fbank_get_work_buffer_bytes(const FbankConfig *config)`
  - 计算给定配置所需工作内存大小

- `int32_t fbank_init_with_buffer(FbankExtractor *extractor, const FbankConfig *config, void *work_buffer)`
  - 使用外部 work buffer 初始化 extractor
  - MCU 场景推荐这个接口

- `int32_t fbank_init(FbankExtractor *extractor, const FbankConfig *config)`
  - 使用内部 static buffer 初始化 extractor

- `void fbank_reset(FbankExtractor *extractor)`
  - 重置 extractor 状态

### 5.3 非流式特征提取接口

- `int32_t fbank_num_frames(const FbankExtractor *extractor, int32_t num_samples)`
  - 计算整段输入可产生多少帧

- `int32_t fbank_extract_float(...)`
  - 对整段 float PCM 提取 fbank

- `int32_t fbank_extract_int16(...)`
  - 对整段 int16 PCM 提取 fbank

- `int32_t fbank_process_frame(...)`
  - 对单帧 float 输入提取一帧 fbank

- `int32_t fbank_process_frame_int16(...)`
  - 对单帧 int16 输入提取一帧 fbank

### 5.4 流式接口

- `int32_t fbank_stream_init(FbankStreamingState *state, const FbankConfig *config)`
  - 初始化流式状态

- `void fbank_stream_reset(FbankStreamingState *state)`
  - 重置流式状态

- `void fbank_stream_free(FbankStreamingState *state)`
  - 释放流式状态内部分配的资源

- `int32_t fbank_stream_pending_samples(const FbankStreamingState *state)`
  - 返回当前流式缓存里剩余的样本数

- `int32_t fbank_stream_accept_float(...)`
  - 喂一段 float PCM
  - 返回本次新产生的帧数

- `int32_t fbank_stream_accept_int16(...)`
  - 喂一段 int16 PCM
  - 返回本次新产生的帧数

### 5.5 工具函数

- `float hz_to_mel(float hz)`
  - Hz 转 Mel

- `float mel_to_hz(float mel)`
  - Mel 转 Hz

- `void fft_radix2(float *real, float *imag, int32_t n)`
  - 基础 radix-2 FFT

## 6. `fbank.cc` 里几个重要的内部行为

虽然这些不是头文件公开 API，但理解它们有助于排查数值对齐问题。

- `resolve_fft_size(...)`
  - 决定最终 FFT size
- `init_mel_banks(...)`
  - 运行时生成 Kaldi 风格 mel filter bank
- `validate_precomputed_table_config(...)`
  - 检查当前配置是否和固定表 [fbank_tables_fixed_config.h](/home/wayne/work/code/project/ffalcon/wekws_baseline/examples/hi_xiaowen/s0/torch2lite/fbank_tables_fixed_config.h) 完全一致
- `init_povey_window(...)`
  - 生成 Kaldi 默认的 Povey window

## 7. 当前项目里推荐怎么用

如果目标是和 `infer_wav.py` 里的 `kaldi.fbank(...)` 对齐，建议参数固定为：

```python
num_mel_bins=80
frame_length=25.0
frame_shift=10.0
dither=0.0
energy_floor=0.0
sample_frequency=16000.0
low_freq=20.0
high_freq=0.0
preemphasis_coefficient=0.97
remove_dc_offset=True
round_to_power_of_two=True
snip_edges=True
fft_size=0
```

在这组参数下，`fft_size` 会自动解析为 `512`，这是当前 pybind 与 `kaldi.fbank` 对齐的默认路径。

如果你想显式指定 FFT size，也可以改成例如 `fft_size=512`；但当前 pybind 是按 `FBANK_USE_PRECOMPUTED_TABLES` 编译的，所以显式配置仍需要和固定表匹配，否则初始化会失败。
