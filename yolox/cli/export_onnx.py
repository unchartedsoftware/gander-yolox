# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import logging
from pathlib import Path
from typing import cast

import torch
from loguru import logger
from torch import nn

from yolox.models import Yolox
from yolox.models.yolox import YoloxModule
from yolox.config import YoloxConfig
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
import onnx
from onnxsim import simplify
from onnxruntime.transformers.float16 import convert_float_to_float16

logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser("yolox onnx-deploy")
    parser.add_argument("-n", "--name", type=str, required=True, help="model name")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="config file path"
    )
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

    return parser


def main():
    args = make_parser().parse_args()

    model = Yolox.from_pretrained(args.name, args.config)
    model.module.eval()
    model.module = cast(YoloxModule, replace_module(model.module, nn.SiLU, SiLU))
    model.module.head.decode_in_inference = args.decode_in_inference

    # Always export to FP32 first
    dummy_input = torch.randn(args.batch_size, 3, 640, 640)

    torch.onnx.export(
        model.module,
        dummy_input,
        args.onnx_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes=(
            {args.input: {0: "batch"}, args.output: {0: "batch"}}
            if args.dynamic
            else None
        ),
        opset_version=args.opset,
    )
    logger.info(f"ONNX model has been successfully created: {args.onnx_name}")

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

    # Simplify as final step
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


if __name__ == "__main__":
    main()
