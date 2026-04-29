#!/usr/bin/env python3
"""Rewrite the onnx2tf FSMN temporal-conv float island back to int8.

This script targets the current xiaolei onnx2tf full-integer model where the
third FSMN cache branch contains:

  int8 cache -> DEQUANTIZE -> float STRIDED_SLICE/TRANSPOSE/CONV_2D/ADD -> QUANTIZE

The rewrite keeps the same graph topology, quantizes the two float temporal
Conv2D weights to int8, adds int32 bias tensors, and changes the intermediate
activation tensors to int8. It is meant as deployment graph surgery, so always
run accuracy and target allocator checks after generating a candidate.
"""

import argparse
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


SOURCE_SCALE_TENSORS = (85, 88, 89, 91, 92, 95, 96, 97)
LEFT_OUTPUT_TENSORS = (90, 93)
RIGHT_OUTPUT_TENSORS = (98,)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source onnx2tf full-int8 .tflite")
    parser.add_argument("--output", required=True, help="rewritten .tflite")
    parser.add_argument("--source-tensor", type=int, default=84)
    parser.add_argument("--dequantize-op", type=int, default=42)
    parser.add_argument("--left-conv-op", type=int, default=47)
    parser.add_argument("--right-conv-op", type=int, default=55)
    parser.add_argument("--left-output-quant", type=int, default=94)
    parser.add_argument("--right-output-quant", type=int, default=99)
    return parser.parse_args()


def buffer_array(buffer, dtype, shape):
    raw = np.asarray(buffer.data, dtype=np.uint8).tobytes()
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def set_buffer_array(buffer, array):
    buffer.data = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint8)


def clone_quant(src):
    if src is None:
        raise RuntimeError("missing quantization parameters")
    quant = schema.QuantizationParametersT()
    quant.min = None if src.min is None else list(src.min)
    quant.max = None if src.max is None else list(src.max)
    quant.scale = None if src.scale is None else list(src.scale)
    quant.zeroPoint = None if src.zeroPoint is None else list(src.zeroPoint)
    quant.detailsType = src.detailsType
    quant.details = src.details
    quant.quantizedDimension = src.quantizedDimension
    return quant


def make_quant(scales, zero_points, quantized_dimension=0):
    quant = schema.QuantizationParametersT()
    quant.scale = [float(x) for x in scales]
    quant.zeroPoint = [int(x) for x in zero_points]
    quant.quantizedDimension = int(quantized_dimension)
    return quant


def single_scale(tensor):
    quant = tensor.quantization
    if (
        quant is None
        or quant.scale is None
        or quant.zeroPoint is None
        or len(quant.scale) != 1
    ):
        raise RuntimeError(f"tensor {tensor.name!r} must have per-tensor quantization")
    return float(quant.scale[0])


def builtin_code(model, op):
    return model.operatorCodes[op.opcodeIndex].builtinCode


def require_builtin(model, op, expected, op_index):
    actual = builtin_code(model, op)
    if actual != expected:
        raise RuntimeError(f"op {op_index} builtin={actual}, expected={expected}")


def find_or_add_opcode(model, builtin, version):
    for index, opcode in enumerate(model.operatorCodes):
        if opcode.builtinCode == builtin:
            return index
    opcode = schema.OperatorCodeT()
    opcode.builtinCode = builtin
    opcode.deprecatedBuiltinCode = builtin
    opcode.version = version
    model.operatorCodes.append(opcode)
    return len(model.operatorCodes) - 1


def quantize_per_output_channel(weight):
    flat = weight.reshape(weight.shape[0], -1)
    max_abs = np.max(np.abs(flat), axis=1)
    scales = np.maximum(max_abs / 127.0, 1.0e-8).astype(np.float32)
    qweight = np.round(weight / scales.reshape((-1, 1, 1, 1)))
    qweight = np.clip(qweight, -127, 127).astype(np.int8)
    return qweight, scales


def quantize_weight(model, tensor, input_scale):
    if tensor.type != schema.TensorType.FLOAT32:
        raise RuntimeError(f"weight tensor {tensor.name!r} is not FLOAT32")
    weight = buffer_array(model.buffers[tensor.buffer], np.float32, [int(x) for x in tensor.shape])
    qweight, scales = quantize_per_output_channel(weight)
    set_buffer_array(model.buffers[tensor.buffer], qweight)
    tensor.type = schema.TensorType.INT8
    tensor.quantization = make_quant(scales, [0] * len(scales), 0)
    return np.asarray(scales, dtype=np.float32)


def append_int32_bias(model, subgraph, name, input_scale, weight_scales):
    bias = np.zeros((len(weight_scales),), dtype=np.int32)

    buffer = schema.BufferT()
    set_buffer_array(buffer, bias)
    model.buffers.append(buffer)

    tensor = schema.TensorT()
    tensor.shape = [int(len(weight_scales))]
    tensor.type = schema.TensorType.INT32
    tensor.buffer = len(model.buffers) - 1
    tensor.name = name
    tensor.quantization = make_quant(weight_scales * np.float32(input_scale), [0] * len(weight_scales), 0)
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1


def replace_bias_input(op, bias_tensor_index):
    inputs = list(op.inputs)
    if len(inputs) < 3:
        raise RuntimeError("Conv2D op has no bias input")
    inputs[2] = bias_tensor_index
    op.inputs = inputs


def set_tensor_int8(tensor, quant):
    tensor.type = schema.TensorType.INT8
    tensor.quantization = clone_quant(quant)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(input_path.read_bytes(), 0))
    if len(model.subgraphs) != 1:
        raise RuntimeError(f"expected one subgraph, got {len(model.subgraphs)}")
    subgraph = model.subgraphs[0]

    require_builtin(model, subgraph.operators[args.dequantize_op], schema.BuiltinOperator.DEQUANTIZE, args.dequantize_op)
    require_builtin(model, subgraph.operators[args.left_conv_op], schema.BuiltinOperator.CONV_2D, args.left_conv_op)
    require_builtin(model, subgraph.operators[args.right_conv_op], schema.BuiltinOperator.CONV_2D, args.right_conv_op)

    source_quant = clone_quant(subgraph.tensors[args.source_tensor].quantization)
    left_quant = clone_quant(subgraph.tensors[args.left_output_quant].quantization)
    right_quant = clone_quant(subgraph.tensors[args.right_output_quant].quantization)

    for tensor_index in SOURCE_SCALE_TENSORS:
        set_tensor_int8(subgraph.tensors[tensor_index], source_quant)
    for tensor_index in LEFT_OUTPUT_TENSORS:
        set_tensor_int8(subgraph.tensors[tensor_index], left_quant)
    for tensor_index in RIGHT_OUTPUT_TENSORS:
        set_tensor_int8(subgraph.tensors[tensor_index], right_quant)

    quantize_opcode = find_or_add_opcode(model, schema.BuiltinOperator.QUANTIZE, version=2)
    dequantize_op = subgraph.operators[args.dequantize_op]
    dequantize_op.opcodeIndex = quantize_opcode
    dequantize_op.builtinOptionsType = 0
    dequantize_op.builtinOptions = None

    input_scale = single_scale(subgraph.tensors[89])
    left_conv = subgraph.operators[args.left_conv_op]
    right_conv = subgraph.operators[args.right_conv_op]
    left_weight_scales = quantize_weight(model, subgraph.tensors[left_conv.inputs[1]], input_scale)
    right_weight_scales = quantize_weight(model, subgraph.tensors[right_conv.inputs[1]], input_scale)

    left_bias = append_int32_bias(model, subgraph, "rewritten_int8_bias_convolution_4", input_scale, left_weight_scales)
    right_bias = append_int32_bias(model, subgraph, "rewritten_int8_bias_convolution_5", input_scale, right_weight_scales)
    replace_bias_input(left_conv, left_bias)
    replace_bias_input(right_conv, right_bias)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), b"TFL3")
    output_path.write_bytes(bytes(builder.Output()))

    print("rewritten_float_island=1")
    print(f"output={output_path}")
    print(f"bytes={output_path.stat().st_size}")
    print(f"new_bias_tensors={left_bias},{right_bias}")


if __name__ == "__main__":
    main()
