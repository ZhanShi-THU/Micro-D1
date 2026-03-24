from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from data.image_transforms import AspectPreservingResizePad, build_image_transform


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


if __name__ == "__main__":
    test_aspect_preserving_resize_pad_returns_square_image()
    test_build_image_transform_pad_preserve_outputs_expected_tensor_shape()
    print({"status": "ok"})
