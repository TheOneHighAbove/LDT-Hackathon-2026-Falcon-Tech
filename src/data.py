"""Dataset and image preprocessing utilities for vehicle ReID.

The annotations use an ``(x, y, width, height)`` box in source-image pixel
coordinates.  Cropping is deliberately performed before any augmentation so
that the model cannot learn from unrelated parts of a camera frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .input_geometry import (
    build_letterbox_eval_transform,
    build_letterbox_train_transform,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = ("image_id", "x", "y", "w", "h")


def _normalized_image_size(
    image_size: int | Sequence[int], *, name: str = "image_size"
) -> int | tuple[int, int]:
    """Validate an integer or ``[height, width]`` input size."""

    if isinstance(image_size, bool):
        raise TypeError(f"{name} must be an integer or [height, width]")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError(f"{name} must be positive")
        return image_size
    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        values = list(image_size)
        if len(values) != 2:
            raise ValueError(f"{name} must contain [height, width]")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError(f"{name} dimensions must be integers")
        height, width = int(values[0]), int(values[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} dimensions must be positive")
        return height, width
    raise TypeError(f"{name} must be an integer or [height, width]")


def _size_hw(image_size: int | tuple[int, int]) -> tuple[int, int]:
    return (image_size, image_size) if isinstance(image_size, int) else image_size


def _resize_mode(value: str) -> str:
    mode = str(value).lower()
    if mode not in {"direct", "letterbox"}:
        raise ValueError("resize_mode must be either 'direct' or 'letterbox'")
    return mode


def _annotations_to_frame(annotations: Any) -> pd.DataFrame:
    """Load a CSV, dataframe, or one/many row mappings into a dataframe."""

    if isinstance(annotations, (str, Path)):
        frame = pd.read_csv(annotations)
    elif isinstance(annotations, pd.DataFrame):
        frame = annotations.copy(deep=True)
    elif isinstance(annotations, pd.Series):
        frame = annotations.to_frame().T
    elif isinstance(annotations, Mapping):
        frame = pd.DataFrame([annotations])
    elif isinstance(annotations, Sequence) and not isinstance(
        annotations, (str, bytes)
    ):
        frame = pd.DataFrame(list(annotations))
    else:
        raise TypeError(
            "annotations must be a CSV path, pandas DataFrame/Series, "
            "or a sequence/mapping of annotation rows"
        )
    return frame


def validate_annotations(
    annotations: Any,
    *,
    require_vehicle_id: bool = False,
    require_camera_id: bool = False,
    check_unique_image_ids: bool = True,
) -> pd.DataFrame:
    """Validate annotations and return a defensive, normalized copy.

    Negative ``x``/``y`` values are permitted because boxes are clipped to the
    source image.  Width and height must be finite and strictly positive.
    Vehicle and camera identifiers intentionally remain in their original
    dtype, allowing either integer or string IDs.
    """

    frame = _annotations_to_frame(annotations)
    required = set(REQUIRED_COLUMNS)
    if require_vehicle_id:
        required.add("vehicle_id")
    if require_camera_id:
        required.add("camera_id")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"annotation columns are missing: {missing}")
    if frame.empty:
        raise ValueError("annotations are empty")

    raw_image_ids = frame["image_id"]
    if raw_image_ids.isna().any():
        raise ValueError("image_id contains missing values")
    image_ids = raw_image_ids.astype(str).str.strip()
    if image_ids.eq("").any():
        raise ValueError("image_id contains an empty value")
    if check_unique_image_ids and image_ids.duplicated().any():
        duplicates = image_ids[image_ids.duplicated(keep=False)].unique()[:5]
        raise ValueError(f"image_id must be unique; duplicates include {duplicates!r}")
    frame.loc[:, "image_id"] = image_ids

    try:
        numeric_boxes = frame.loc[:, ["x", "y", "w", "h"]].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox columns x, y, w, h must be numeric") from exc
    boxes = numeric_boxes.to_numpy(dtype=np.float64)
    if not np.isfinite(boxes).all():
        raise ValueError("bbox columns contain NaN or infinite values")
    if (boxes[:, 2:] <= 0).any():
        raise ValueError("bbox width and height must be strictly positive")
    frame.loc[:, ["x", "y", "w", "h"]] = numeric_boxes

    for column in ("vehicle_id", "camera_id"):
        if column in frame.columns and frame[column].isna().any():
            raise ValueError(f"{column} contains missing values")

    return frame.reset_index(drop=True)


def _padding_pair(padding: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(padding, tuple):
        if len(padding) != 2:
            raise ValueError("padding tuple must be (horizontal, vertical)")
        horizontal, vertical = (float(padding[0]), float(padding[1]))
    else:
        horizontal = vertical = float(padding)
    if (
        not math.isfinite(horizontal)
        or not math.isfinite(vertical)
        or horizontal < 0
        or vertical < 0
    ):
        raise ValueError("padding must contain finite, non-negative fractions")
    return horizontal, vertical


def padded_bbox(
    image_size: tuple[int, int],
    bbox: Sequence[float],
    padding: float | tuple[float, float] = 0.1,
) -> tuple[int, int, int, int]:
    """Expand an xywh box fractionally and clip it to a PIL image size.

    Returns integer ``(left, top, right, bottom)`` coordinates suitable for
    :meth:`PIL.Image.Image.crop`.  An entirely out-of-frame box is rejected.
    """

    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly x, y, w, h")
    image_width, image_height = int(image_size[0]), int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    x, y, width, height = (float(value) for value in bbox)
    values = np.asarray((x, y, width, height), dtype=np.float64)
    if not np.isfinite(values).all() or width <= 0 or height <= 0:
        raise ValueError("bbox must be finite with positive width and height")
    horizontal, vertical = _padding_pair(padding)

    left = max(0, min(image_width, math.floor(x - horizontal * width)))
    top = max(0, min(image_height, math.floor(y - vertical * height)))
    right = max(0, min(image_width, math.ceil(x + width + horizontal * width)))
    bottom = max(0, min(image_height, math.ceil(y + height + vertical * height)))
    if right <= left or bottom <= top:
        raise ValueError(
            f"bbox {tuple(bbox)!r} is empty after clipping to image size "
            f"{image_size!r}"
        )
    return left, top, right, bottom


def crop_vehicle(
    image: Image.Image,
    bbox: Sequence[float],
    padding: float | tuple[float, float] = 0.1,
) -> Image.Image:
    """Return an RGB vehicle crop with padded, clipped coordinates."""

    box = padded_bbox(image.size, bbox, padding)
    return image.convert("RGB").crop(box)


def build_train_transform(
    image_size: int | Sequence[int] = 256,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    crop_scale: tuple[float, float] = (0.82, 1.0),
    crop_ratio: tuple[float, float] = (0.82, 1.22),
    horizontal_flip_probability: float = 0.5,
    random_erasing_probability: float = 0.25,
    resize_mode: str = "direct",
) -> transforms.Compose:
    """Strong but geometry-preserving training augmentation pipeline."""

    normalized_size = _normalized_image_size(image_size)
    mode = _resize_mode(resize_mode)
    if mode == "letterbox":
        # Letterbox deliberately retains the entire crop; crop_scale and
        # crop_ratio only apply to the legacy RandomResizedCrop path.
        return build_letterbox_train_transform(
            normalized_size,
            mean=mean,
            std=std,
            horizontal_flip_probability=horizontal_flip_probability,
            random_erasing_probability=random_erasing_probability,
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                normalized_size,
                scale=crop_scale,
                ratio=crop_ratio,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.20, hue=0.06
            ),
            transforms.RandomRotation(
                degrees=8,
                interpolation=InterpolationMode.BILINEAR,
                fill=(124, 116, 104),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
            transforms.RandomErasing(
                p=random_erasing_probability,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def build_eval_transform(
    image_size: int | Sequence[int] = 256,
    *,
    resize_size: int | Sequence[int] | None = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    resize_mode: str = "direct",
) -> transforms.Compose:
    """Deterministic resize/center-crop/ImageNet-normalization pipeline."""

    normalized_size = _normalized_image_size(image_size)
    mode = _resize_mode(resize_mode)
    if mode == "letterbox":
        if resize_size is not None:
            raise ValueError("resize_size is not used with letterbox resize_mode")
        return build_letterbox_eval_transform(
            normalized_size,
            mean=mean,
            std=std,
        )

    target_height, target_width = _size_hw(normalized_size)
    if resize_size is None:
        normalized_resize: int | tuple[int, int]
        if isinstance(normalized_size, int):
            normalized_resize = int(math.ceil(normalized_size / 0.875))
        else:
            normalized_resize = (
                int(math.ceil(target_height / 0.875)),
                int(math.ceil(target_width / 0.875)),
            )
    else:
        normalized_resize = _normalized_image_size(resize_size, name="resize_size")
    resize_height, resize_width = _size_hw(normalized_resize)
    if resize_height < target_height or resize_width < target_width:
        raise ValueError("resize_size must be at least image_size in each dimension")
    return transforms.Compose(
        [
            transforms.Resize(
                (resize_height, resize_width),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(normalized_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


# A concise alias used by some training/inference entry points.
build_val_transform = build_eval_transform


class VehicleDataset(Dataset[dict[str, Any]]):
    """Vehicle crops backed by the provided annotation CSV format.

    ``annotations`` may also be a dataframe, a row mapping/Series, or a list of
    row mappings.  Every item is a dictionary so metadata stays available for
    loss functions, cross-camera validation, and deterministic output ordering.
    ``label`` is a contiguous training class index while ``vehicle_id`` retains
    the source identifier.  Unlabelled query/gallery rows receive ``-1``.
    """

    def __init__(
        self,
        annotations: Any,
        image_dir: str | Path,
        *,
        transform: Any | None = None,
        train: bool = False,
        image_size: int | Sequence[int] = 256,
        bbox_padding: float | tuple[float, float] = 0.1,
        image_suffix: str = ".jpg",
        pid_to_label: Mapping[Any, int] | None = None,
        require_camera_id: bool = False,
        check_unique_image_ids: bool = True,
    ) -> None:
        super().__init__()
        self.annotations = validate_annotations(
            annotations,
            require_vehicle_id=train,
            require_camera_id=require_camera_id,
            check_unique_image_ids=check_unique_image_ids,
        )
        self.image_dir = Path(image_dir).expanduser().resolve()
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image directory does not exist: {self.image_dir}")
        if not image_suffix.startswith("."):
            image_suffix = f".{image_suffix}"
        self.image_suffix = image_suffix
        self.bbox_padding = _padding_pair(bbox_padding)
        self.transform = (
            transform
            if transform is not None
            else (
                build_train_transform(image_size)
                if train
                else build_eval_transform(image_size)
            )
        )

        if "vehicle_id" in self.annotations:
            unique_pids = list(pd.unique(self.annotations["vehicle_id"]))
            try:
                unique_pids = sorted(unique_pids)
            except TypeError:
                unique_pids = sorted(unique_pids, key=lambda value: str(value))
            if pid_to_label is None:
                self.pid_to_label = {
                    pid: label for label, pid in enumerate(unique_pids)
                }
            else:
                self.pid_to_label = dict(pid_to_label)
                missing_pids = [pid for pid in unique_pids if pid not in self.pid_to_label]
                if missing_pids:
                    raise ValueError(
                        "pid_to_label is missing vehicle IDs: "
                        f"{missing_pids[:5]!r}"
                    )
                labels = list(self.pid_to_label.values())
                if any(not isinstance(label, (int, np.integer)) for label in labels):
                    raise ValueError("pid_to_label values must be integer labels")
                if any(int(label) < 0 for label in labels):
                    raise ValueError("pid_to_label values must be non-negative")
                if len(set(int(label) for label in labels)) != len(labels):
                    raise ValueError("pid_to_label values must be unique")
            self.labels = [
                int(self.pid_to_label[pid])
                for pid in self.annotations["vehicle_id"].tolist()
            ]
            self.pids = self.annotations["vehicle_id"].tolist()
        else:
            self.pid_to_label = {}
            self.labels = [-1] * len(self.annotations)
            self.pids = [-1] * len(self.annotations)
        self.camera_ids = (
            self.annotations["camera_id"].tolist()
            if "camera_id" in self.annotations
            else [-1] * len(self.annotations)
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _image_path(self, image_id: str) -> Path:
        relative = Path(image_id)
        if not relative.suffix:
            relative = relative.with_suffix(self.image_suffix)
        candidate = (self.image_dir / relative).resolve()
        try:
            candidate.relative_to(self.image_dir)
        except ValueError as exc:
            raise ValueError(f"image_id escapes image directory: {image_id!r}") from exc
        return candidate

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.annotations.iloc[index]
        image_id = str(row["image_id"])
        image_path = self._image_path(image_id)
        if not image_path.is_file():
            raise FileNotFoundError(f"image file does not exist: {image_path}")
        try:
            with Image.open(image_path) as source:
                source.load()
                crop = crop_vehicle(
                    source,
                    (row["x"], row["y"], row["w"], row["h"]),
                    self.bbox_padding,
                )
        except (OSError, ValueError) as exc:
            raise type(exc)(f"failed to load/crop image_id={image_id!r}: {exc}") from exc

        image = self.transform(crop) if self.transform is not None else crop
        pid = self.pids[index]
        camera_id = self.camera_ids[index]
        return {
            "image": image,
            "label": self.labels[index],
            "vehicle_id": pid,
            "camera_id": camera_id,
            "image_id": image_id,
            "bbox": torch.tensor(
                [row["x"], row["y"], row["w"], row["h"]],
                dtype=torch.float32,
            ),
            "index": index,
        }


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "VehicleDataset",
    "build_eval_transform",
    "build_train_transform",
    "build_val_transform",
    "crop_vehicle",
    "padded_bbox",
    "validate_annotations",
]
