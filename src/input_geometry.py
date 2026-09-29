"""Aspect-preserving input geometry for vehicle crops.

The active baseline deliberately remains untouched.  This module contains an
alternative preprocessing path that can be selected by later experiments
without stretching a vehicle crop or discarding its edges with a centre crop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageSize: TypeAlias = int | tuple[int, int]
RGBFill: TypeAlias = tuple[int, int, int]


def _target_hw(size: ImageSize) -> tuple[int, int]:
    """Normalize an integer or ``(height, width)`` target size."""

    if isinstance(size, bool):
        raise TypeError("size must be an integer or a (height, width) tuple")
    if isinstance(size, int):
        height = width = size
    elif isinstance(size, tuple) and len(size) == 2:
        height, width = size
        if isinstance(height, bool) or isinstance(width, bool):
            raise TypeError("target dimensions must be integers")
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("target dimensions must be integers")
    else:
        raise TypeError("size must be an integer or a (height, width) tuple")
    if height <= 0 or width <= 0:
        raise ValueError("target dimensions must be positive")
    return height, width


def normalized_zero_fill(mean: Sequence[float] = IMAGENET_MEAN) -> RGBFill:
    """Return the 8-bit RGB colour closest to zero after normalization.

    For the ImageNet mean this is ``(124, 116, 104)``.  Rounding to the nearest
    byte keeps every normalized padding channel within roughly 0.01 of zero.
    """

    if len(mean) != 3:
        raise ValueError("mean must contain exactly three RGB values")
    values = tuple(float(value) for value in mean)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("mean values must be finite and lie in [0, 1]")
    return tuple(int(round(value * 255.0)) for value in values)  # type: ignore[return-value]


def letterbox_resize(
    image: Image.Image,
    size: ImageSize,
    *,
    fill: RGBFill | None = None,
    interpolation: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    """Resize a PIL image into ``(height, width)`` without aspect distortion.

    The largest fitting image is centred on an RGB canvas.  When an odd number
    of padding pixels is required, the extra pixel is deterministically placed
    on the bottom or right.  Inputs are converted to RGB because the model and
    ImageNet normalization both expect three channels.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    target_height, target_width = _target_hw(size)
    if fill is None:
        fill = normalized_zero_fill()
    if len(fill) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 255
        for value in fill
    ):
        raise ValueError("fill must contain three integer RGB values in [0, 255]")

    source = image.convert("RGB")
    source_width, source_height = source.size
    if source_width <= 0 or source_height <= 0:  # defensive; PIL forbids this
        raise ValueError("source image dimensions must be positive")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, int(round(source_width * scale))))
    resized_height = max(1, min(target_height, int(round(source_height * scale))))
    resized = source.resize((resized_width, resized_height), resample=interpolation)

    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas = Image.new("RGB", (target_width, target_height), color=fill)
    canvas.paste(resized, (left, top))
    return canvas


class LetterboxResize:
    """Pickle-friendly callable wrapper around :func:`letterbox_resize`."""

    def __init__(
        self,
        size: ImageSize,
        *,
        fill: RGBFill | None = None,
        interpolation: Image.Resampling = Image.Resampling.BICUBIC,
    ) -> None:
        self.size = _target_hw(size)
        self.fill = normalized_zero_fill() if fill is None else fill
        # Validate fill eagerly rather than failing inside a DataLoader worker.
        if len(self.fill) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.fill
        ):
            raise ValueError(
                "fill must contain three integer RGB values in [0, 255]"
            )
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        return letterbox_resize(
            image,
            self.size,
            fill=self.fill,
            interpolation=self.interpolation,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size!r}, "
            f"fill={self.fill!r}, interpolation={self.interpolation!r})"
        )


def _normalization_values(
    mean: Sequence[float], std: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normalized_mean = tuple(float(value) for value in mean)
    normalized_std = tuple(float(value) for value in std)
    if len(normalized_mean) != 3 or len(normalized_std) != 3:
        raise ValueError("mean and std must contain exactly three RGB values")
    if not all(math.isfinite(value) for value in (*normalized_mean, *normalized_std)):
        raise ValueError("mean and std values must be finite")
    if any(value <= 0 for value in normalized_std):
        raise ValueError("std values must be positive")
    return normalized_mean, normalized_std


def build_letterbox_eval_transform(
    image_size: ImageSize = 384,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    interpolation: Image.Resampling = Image.Resampling.BICUBIC,
) -> transforms.Compose:
    """Build deterministic aspect-preserving evaluation preprocessing."""

    normalized_mean, normalized_std = _normalization_values(mean, std)
    return transforms.Compose(
        [
            LetterboxResize(
                image_size,
                fill=normalized_zero_fill(normalized_mean),
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalized_mean, std=normalized_std),
        ]
    )


def build_letterbox_train_transform(
    image_size: ImageSize = 384,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    horizontal_flip_probability: float = 0.5,
    random_erasing_probability: float = 0.25,
    interpolation: Image.Resampling = Image.Resampling.BICUBIC,
) -> transforms.Compose:
    """Build augmentations that retain the complete vehicle and its aspect."""

    normalized_mean, normalized_std = _normalization_values(mean, std)
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must lie in [0, 1]")
    if not 0.0 <= random_erasing_probability <= 1.0:
        raise ValueError("random_erasing_probability must lie in [0, 1]")
    fill = normalized_zero_fill(normalized_mean)
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.20, hue=0.06
            ),
            transforms.RandomRotation(
                degrees=8,
                interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            ),
            LetterboxResize(
                image_size,
                fill=fill,
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalized_mean, std=normalized_std),
            transforms.RandomErasing(
                p=random_erasing_probability,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LetterboxResize",
    "build_letterbox_eval_transform",
    "build_letterbox_train_transform",
    "letterbox_resize",
    "normalized_zero_fill",
]
