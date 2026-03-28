from __future__ import annotations

from typing import Iterable, Literal, Sequence

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PAD_FILL = tuple(int(channel * 255) for channel in IMAGENET_MEAN)


class AspectPreservingResizePad:
    def __init__(
        self,
        image_size: int,
        fill: tuple[int, int, int] = DEFAULT_PAD_FILL,
    ) -> None:
        self.image_size = int(image_size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {image.size}.")

        scale = self.image_size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        image = TF.resize(
            image,
            size=[resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top

        return TF.pad(image, padding=[left, top, right, bottom], fill=self.fill)


class QwenHybridResize:
    def __init__(
        self,
        image_size: int,
        dynamic_buckets: Sequence[int] | None = None,
        patch_size: int = 16,
    ) -> None:
        self.image_size = int(image_size)
        if dynamic_buckets:
            self.dynamic_buckets = sorted({int(value) for value in dynamic_buckets if int(value) > 0})
        else:
            self.dynamic_buckets = [self.image_size]
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")

    def _select_bucket(self, width: int, height: int) -> int:
        longest_edge = max(width, height)
        clamped = min(max(longest_edge, self.dynamic_buckets[0]), self.dynamic_buckets[-1])
        return min(self.dynamic_buckets, key=lambda bucket: (abs(bucket - clamped), bucket))

    def _align_dimension(self, value: int, *, max_value: int) -> int:
        aligned = max(self.patch_size, int(round(value / self.patch_size)) * self.patch_size)
        if aligned > max_value:
            aligned = max(self.patch_size, (max_value // self.patch_size) * self.patch_size)
        return max(1, aligned)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {image.size}.")

        bucket = self._select_bucket(width, height)
        scale = bucket / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized_width = self._align_dimension(resized_width, max_value=bucket)
        resized_height = self._align_dimension(resized_height, max_value=bucket)

        return TF.resize(
            image,
            size=[resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )


class ImagePreprocessor:
    def __init__(
        self,
        image_size: int,
        preprocessing: Literal["resize", "pad_preserve", "qwen_hybrid"] | str = "resize",
        dynamic_buckets: Sequence[int] | None = None,
        patch_size: int = 16,
    ) -> None:
        self.mode = str(preprocessing).strip().lower()
        self.image_size = int(image_size)
        self.dynamic_buckets = [int(value) for value in dynamic_buckets] if dynamic_buckets is not None else None
        self.patch_size = int(patch_size)

        if self.mode == "resize":
            self.spatial_transform = transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        elif self.mode == "pad_preserve":
            self.spatial_transform = AspectPreservingResizePad(image_size=image_size)
        elif self.mode == "qwen_hybrid":
            self.spatial_transform = QwenHybridResize(
                image_size=image_size,
                dynamic_buckets=dynamic_buckets,
                patch_size=patch_size,
            )
        else:
            raise ValueError(
                "Unsupported image preprocessing mode. Expected one of: resize, pad_preserve, qwen_hybrid. "
                f"Received: {preprocessing!r}"
            )

        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.tensor_transform(self.spatial_transform(image))


def pad_and_stack_image_tensors(
    image_tensors: Iterable[torch.Tensor],
    *,
    patch_size: int = 1,
    pad_value: float = 0.0,
) -> torch.Tensor:
    tensors = list(image_tensors)
    if not tensors:
        raise ValueError("image_tensors must not be empty.")

    max_height = max(tensor.shape[-2] for tensor in tensors)
    max_width = max(tensor.shape[-1] for tensor in tensors)

    if patch_size > 1:
        max_height = ((max_height + patch_size - 1) // patch_size) * patch_size
        max_width = ((max_width + patch_size - 1) // patch_size) * patch_size

    padded_tensors = []
    for tensor in tensors:
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor shape [C, H, W], got {tuple(tensor.shape)}.")
        height = tensor.shape[-2]
        width = tensor.shape[-1]
        pad_bottom = max_height - height
        pad_right = max_width - width
        padded_tensors.append(TF.pad(tensor, padding=[0, 0, pad_right, pad_bottom], fill=pad_value))
    return torch.stack(padded_tensors)


def build_image_transform(
    image_size: int,
    preprocessing: Literal["resize", "pad_preserve", "qwen_hybrid"] | str = "resize",
    dynamic_buckets: Sequence[int] | None = None,
    patch_size: int = 16,
):
    return ImagePreprocessor(
        image_size=image_size,
        preprocessing=preprocessing,
        dynamic_buckets=dynamic_buckets,
        patch_size=patch_size,
    )
