# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import inspect
import logging
from typing import cast

import torch
from loguru import logger
from torch import nn

from yolox.models import Yolox
from yolox.models.yolox import YoloxModule
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
import onnx
from onnxsim import simplify
from onnxruntime.transformers.float16 import convert_float_to_float16

logger = logging.getLogger(__name__)
MIN_DYNAMO_OPSET = 18


def make_parser():
    parser = argparse.ArgumentParser("yolox onnx-deploy")
    parser.add_argument("-n", "--name", type=str, required=True, help="model name")
    parser.add_argument(
        "--onnx-name", type=str, required=True, help="name of onnx output file"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="whether the input shape should be dynamic or not",
    )
    parser.add_argument("--onnxsim", action="store_false", help="simplify onnx or not")
    parser.add_argument(
        "--decode-in-inference", action="store_true", help="decode in inference or not"
    )
    parser.add_argument(
        "--half", action="store_true", help="export model in half precision (FP16)"
    )
    parser.add_argument(
        "--keep-io",
        action="store_true",
        help="keep input/output tensors as FP32 when using --half (applies keep_io_types flag)",
    )
    parser.add_argument(
        "--disable-dynamo",
        action="store_true",
        help="force the legacy torch.onnx exporter (required for opset < 18)",
    )

    return parser


def main():
    args = make_parser().parse_args()

    model = Yolox.from_pretrained(args.name)
    model.module.eval()
    model.module = cast(YoloxModule, replace_module(model.module, nn.SiLU, SiLU))
    model.module.head.decode_in_inference = args.decode_in_inference

    # Always export to FP32 first
    test_size = getattr(model.processor.config, "test_size", (640, 640))
    height, width = int(test_size[0]), int(test_size[1])
    dummy_input = torch.randn(args.batch_size, 3, height, width)
    export_args = (dummy_input,)

    supports_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    disable_dynamo = args.opset < MIN_DYNAMO_OPSET or args.disable_dynamo

    if args.opset < MIN_DYNAMO_OPSET:
        logger.info(
            "Disabling dynamo exporter because requested opset %s < %s",
            args.opset,
            MIN_DYNAMO_OPSET,
        )
    if args.disable_dynamo:
        logger.info("Dynamo exporter explicitly disabled via --disable-dynamo")

    export_kwargs = {}
    if supports_dynamo:
        export_kwargs["dynamo"] = not disable_dynamo
        logger.info(
            "Using %s torch.onnx exporter",
            "dynamo" if export_kwargs["dynamo"] else "legacy",
        )
    elif not disable_dynamo:
        logger.info(
            "Current PyTorch does not expose the dynamo flag; using legacy exporter"
        )

    dynamic_axes = (
        {args.input: {0: "batch"}, args.output: {0: "batch"}} if args.dynamic else None
    )

    torch.onnx.export(
        model.module,
        export_args,
        args.onnx_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        **export_kwargs,
    )
    logger.info(f"ONNX model has been successfully created: {args.onnx_name}")

    # Simplify before any optional dtype conversions so downstream tools operate on a clean graph
    if args.onnxsim:
        logger.info("Simplify ONNX model")
        onnx_model = onnx.load(args.onnx_name)
        model_simp, valid = simplify(onnx_model)
        if not valid:
            logger.error("Simplified ONNX model could not be validated")
            return
        onnx.save(model_simp, args.onnx_name)
        logger.info(
            f"Simplified ONNX model has been successfully created: {args.onnx_name}"
        )

    # Convert to FP16 if requested using ONNX float16 conversion
    if args.half:
        logger.info("Converting ONNX model to FP16")
        onnx_model = onnx.load(args.onnx_name)
        onnx_model_fp16 = convert_float_to_float16(
            onnx_model,
            keep_io_types=args.keep_io,
        )
        onnx.save(onnx_model_fp16, args.onnx_name)
        logger.info(f"FP16 ONNX model has been successfully created: {args.onnx_name}")


if __name__ == "__main__":
    main()
