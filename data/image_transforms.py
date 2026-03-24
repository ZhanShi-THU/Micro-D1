from __future__ import annotations

from typing import Literal

from PIL import Image
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


def build_image_transform(
    image_size: int,
    preprocessing: Literal["resize", "pad_preserve"] | str = "resize",
):
    mode = str(preprocessing).strip().lower()
    if mode == "resize":
        spatial_transform = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    elif mode == "pad_preserve":
        spatial_transform = AspectPreservingResizePad(image_size=image_size)
    else:
        raise ValueError(
            "Unsupported image preprocessing mode. Expected one of: resize, pad_preserve. "
            f"Received: {preprocessing!r}"
        )

    return transforms.Compose(
        [
            spatial_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
