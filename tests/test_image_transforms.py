from __future__ import annotations

import tempfile
from pathlib import Path
import sys

from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.image_transforms import (
    AspectPreservingResizePad,
    build_image_transform,
    pad_and_stack_image_tensors,
)


def test_aspect_preserving_resize_pad_returns_square_image() -> None:
    image = Image.new("RGB", (640, 320), color="white")
    transformed = AspectPreservingResizePad(image_size=448)(image)
    assert transformed.size == (448, 448)


def test_build_image_transform_pad_preserve_outputs_expected_tensor_shape() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "sample.png"
        Image.new("RGB", (300, 500), color="gray").save(image_path)
        image = Image.open(image_path).convert("RGB")

        tensor = build_image_transform(448, preprocessing="pad_preserve")(image)

        assert tuple(tensor.shape) == (3, 448, 448)


def test_build_image_transform_qwen_hybrid_aligns_to_patch_size() -> None:
    image = Image.new("RGB", (640, 320), color="white")
    tensor = build_image_transform(
        448,
        preprocessing="qwen_hybrid",
        dynamic_buckets=[384, 448, 512],
        patch_size=16,
    )(image)

    assert tuple(tensor.shape[:1]) == (3,)
    assert tensor.shape[1] % 16 == 0
    assert tensor.shape[2] % 16 == 0
    assert max(tensor.shape[1], tensor.shape[2]) in {384, 448, 512}


def test_pad_and_stack_image_tensors_pads_to_batch_max_shape() -> None:
    first = torch.zeros(3, 384, 256)
    second = torch.zeros(3, 448, 320)

    stacked = pad_and_stack_image_tensors([first, second], patch_size=16)

    assert tuple(stacked.shape) == (2, 3, 448, 320)


if __name__ == "__main__":
    test_aspect_preserving_resize_pad_returns_square_image()
    test_build_image_transform_pad_preserve_outputs_expected_tensor_shape()
    test_build_image_transform_qwen_hybrid_aligns_to_patch_size()
    test_pad_and_stack_image_tensors_pads_to_batch_max_shape()
    print({"status": "ok"})
