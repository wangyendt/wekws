from __future__ import annotations

from pathlib import Path

import numpy as np


def load_tflite_interpreter(model_path: str | Path):
    model_path = str(model_path)
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter(model_path=model_path)
        except ImportError as exc:
            raise RuntimeError(
                "Neither tensorflow nor tflite_runtime is available in the current environment."
            ) from exc


def _get_single_quant_param(detail: dict, key: str, fallback_value):
    quant_params = detail.get("quantization_parameters", {})
    values = quant_params.get(key)
    if values is not None:
        values = np.asarray(values)
        if values.size == 1:
            return values.reshape(-1)[0].item()
    quant = detail.get("quantization")
    if quant is not None and len(quant) == 2:
        return quant[0 if key == "scales" else 1]
    return fallback_value


def get_tensor_quant_params(detail: dict) -> tuple[float | None, int | None]:
    scale = _get_single_quant_param(detail, "scales", 0.0)
    zero_point = _get_single_quant_param(detail, "zero_points", 0)
    scale = float(scale)
    zero_point = int(zero_point)
    if scale == 0.0:
        return None, None
    return scale, zero_point


def quantize_to_detail(data, detail: dict) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    array = np.asarray(data)
    if np.issubdtype(dtype, np.floating):
        return array.astype(dtype, copy=False)

    scale, zero_point = get_tensor_quant_params(detail)
    if scale is None:
        return array.astype(dtype, copy=False)

    info = np.iinfo(dtype)
    quantized = np.rint(array.astype(np.float32) / scale + zero_point)
    quantized = np.clip(quantized, info.min, info.max)
    return quantized.astype(dtype)


def dequantize_from_detail(data, detail: dict) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    array = np.asarray(data)
    if np.issubdtype(dtype, np.floating):
        return array.astype(np.float32, copy=False)

    scale, zero_point = get_tensor_quant_params(detail)
    if scale is None:
        return array.astype(np.float32, copy=False)
    return (array.astype(np.float32) - zero_point) * scale
