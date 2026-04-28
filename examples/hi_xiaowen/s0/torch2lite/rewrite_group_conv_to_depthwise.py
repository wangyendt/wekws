#!/usr/bin/env python3
"""Rewrite depthwise-equivalent TFLite Conv2D ops to DepthwiseConv2D.

onnx2tf lowers PyTorch grouped Conv2d(dim, dim, groups=dim) to TFLite
CONV_2D with weight shape [C, H, 1, 1]. For the FSMN memory layer this is
mathematically a depthwise convolution. Rewriting the op lets TFLM use the
depthwise kernel, which usually needs less scratch arena than generic Conv2D.
"""

import argparse
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


TYPE_TO_DTYPE = {
    schema.TensorType.INT8: np.int8,
    schema.TensorType.FLOAT32: np.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source .tflite")
    parser.add_argument("--output", required=True, help="rewritten .tflite")
    parser.add_argument(
        "--min-converted",
        type=int,
        default=1,
        help="fail if fewer ops are rewritten",
    )
    return parser.parse_args()


def buffer_array(buffer_data, dtype, shape):
    raw = np.asarray(buffer_data, dtype=np.uint8).tobytes()
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def add_depthwise_opcode(model):
    opcode = schema.OperatorCodeT()
    opcode.builtinCode = schema.BuiltinOperator.DEPTHWISE_CONV_2D
    opcode.deprecatedBuiltinCode = schema.BuiltinOperator.DEPTHWISE_CONV_2D
    opcode.version = 3
    model.operatorCodes.append(opcode)
    return len(model.operatorCodes) - 1


def is_depthwise_equivalent(subgraph, model, op):
    code = model.operatorCodes[op.opcodeIndex]
    if code.builtinCode != schema.BuiltinOperator.CONV_2D or len(op.inputs) < 2:
        return False
    input_tensor = subgraph.tensors[op.inputs[0]]
    weight_tensor = subgraph.tensors[op.inputs[1]]
    if len(input_tensor.shape) != 4 or len(weight_tensor.shape) != 4:
        return False
    input_c = int(input_tensor.shape[3])
    out_c, _, filter_w, filter_in_c = [int(x) for x in weight_tensor.shape]
    return (
        input_c == out_c
        and filter_in_c == 1
        and filter_w == 1
        and weight_tensor.type in TYPE_TO_DTYPE
    )


def rewrite_op(subgraph, model, op, depthwise_opcode_index):
    weight_tensor = subgraph.tensors[op.inputs[1]]
    dtype = TYPE_TO_DTYPE[weight_tensor.type]
    old_weight = buffer_array(
        model.buffers[weight_tensor.buffer].data,
        dtype,
        weight_tensor.shape,
    )

    # Conv2D OHWI [C, H, 1, 1] -> Depthwise OHWI [1, H, 1, C].
    new_weight = np.transpose(old_weight, (3, 1, 2, 0)).copy()
    model.buffers[weight_tensor.buffer].data = np.frombuffer(
        new_weight.tobytes(),
        dtype=np.uint8,
    )
    weight_tensor.shape = list(new_weight.shape)
    if weight_tensor.quantization is not None:
        weight_tensor.quantization.quantizedDimension = 3

    conv_opts = op.builtinOptions
    depth_opts = schema.DepthwiseConv2DOptionsT()
    depth_opts.padding = conv_opts.padding
    depth_opts.strideW = conv_opts.strideW
    depth_opts.strideH = conv_opts.strideH
    depth_opts.depthMultiplier = 1
    depth_opts.fusedActivationFunction = conv_opts.fusedActivationFunction
    depth_opts.dilationWFactor = conv_opts.dilationWFactor
    depth_opts.dilationHFactor = conv_opts.dilationHFactor

    op.opcodeIndex = depthwise_opcode_index
    op.builtinOptionsType = schema.BuiltinOptions.DepthwiseConv2DOptions
    op.builtinOptions = depth_opts


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    data = input_path.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))
    depthwise_opcode_index = add_depthwise_opcode(model)

    converted = 0
    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            if is_depthwise_equivalent(subgraph, model, op):
                rewrite_op(subgraph, model, op, depthwise_opcode_index)
                converted += 1

    if converted < args.min_converted:
        raise RuntimeError(
            f"converted {converted} ops, expected at least {args.min_converted}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), b"TFL3")
    output_path.write_bytes(bytes(builder.Output()))
    print(f"converted={converted}")
    print(f"output={output_path}")
    print(f"bytes={output_path.stat().st_size}")


if __name__ == "__main__":
    main()
