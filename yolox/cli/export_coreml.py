
import argparse
import importlib
import logging
import re
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from yolox.models import Yolox
from yolox.models.network_blocks import SiLU
from yolox.models.yolox import YoloxModule
from yolox.utils import replace_module

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("yolox coreml-deploy")
    parser.add_argument("-n", "--name", type=str, required=True, help="model name")
    parser.add_argument(
        "--mlpackage-name",
        type=str,
        required=True,
        help="output CoreML package path (should end with .mlpackage)",
    )
    parser.add_argument(
        "--input",
        default="images",
        type=str,
        help="input name (CoreML feature name)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "batch size used for tracing; when using --dynamic this is also the max batch size (capped at "
            "--batch-size)"
        ),
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="make batch dimension flexible in the CoreML model",
    )
    parser.add_argument(
        "--decode-in-inference",
        action="store_true",
        help="decode in inference or not",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="export CoreML model with FP16 compute precision (mlprogram)",
    )

    return parser


def main() -> None:
    args = make_parser().parse_args()

    try:
        ct: Any = importlib.import_module("coremltools")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "coremltools is required for CoreML export. Install it with `uv sync --group coreml`. "
            f"Original error: {exc!r}"
        ) from exc

    output_path = Path(args.mlpackage_name)
    precision_suffix = "16" if args.half else "32"

    # Always export as an .mlpackage directory, and encode precision in the name.
    # Example: yolox_tiny_32.mlpackage / yolox_tiny_16.mlpackage
    base_name = output_path.stem if output_path.suffix else output_path.name
    base_name = base_name.replace("-", "_")
    base_name = re.sub(r"[-_](16|32)$", "", base_name)
    output_path = output_path.with_name(f"{base_name}_{precision_suffix}.mlpackage")

    model = Yolox.from_pretrained(args.name)
    model.module.eval()
    model.module = cast(YoloxModule, replace_module(model.module, nn.SiLU, SiLU))
    model.module.head.decode_in_inference = args.decode_in_inference

    # CoreML conversion expects CPU execution during tracing/conversion.
    model.module.to("cpu")

    # Keep input size consistent with the ONNX exporter.
    trace_batch = int(args.batch_size)
    if trace_batch < 1:
        raise SystemExit("--batch-size must be >= 1")

    test_size = getattr(model.processor.config, "test_size", (640, 640))
    height, width = int(test_size[0]), int(test_size[1])
    dummy_input = torch.randn(trace_batch, 3, height, width, device="cpu")

    with torch.no_grad():
        traced = torch.jit.trace(model.module, dummy_input)

    if args.dynamic:
        batch_dim = ct.RangeDim(1, int(args.batch_size))
    else:
        batch_dim = int(args.batch_size)

    inputs = [
        ct.TensorType(
            name=args.input,
            shape=(batch_dim, 3, height, width),
        )
    ]

    convert_kwargs = {
        "inputs": inputs,
        "convert_to": "mlprogram",
    }
    if args.half:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16

    mlmodel: Any = ct.convert(traced, **convert_kwargs)
    mlmodel.save(str(output_path))

    logger.info("CoreML model package has been successfully created: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
