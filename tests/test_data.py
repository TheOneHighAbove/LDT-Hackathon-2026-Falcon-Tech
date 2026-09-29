from __future__ import annotations

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data import (
    VehicleDataset,
    build_eval_transform,
    build_train_transform,
    padded_bbox,
    validate_annotations,
)


def test_padded_bbox_clips_to_image_boundaries() -> None:
    assert padded_bbox((100, 80), (-10, -5, 30, 20), padding=0.1) == (
        0,
        0,
        23,
        17,
    )
    with pytest.raises(ValueError, match="empty after clipping"):
        padded_bbox((100, 80), (120, 10, 5, 5), padding=0.0)


def test_dataset_accepts_a_row_and_returns_clipped_crop(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 80), color=(20, 40, 60)).save(image_dir / "vehicle.jpg")
    row = pd.Series(
        {
            "image_id": "vehicle",
            "x": -10,
            "y": -5,
            "w": 30,
            "h": 20,
            "vehicle_id": 7,
            "camera_id": 2,
        }
    )
    dataset = VehicleDataset(
        row,
        image_dir,
        transform=lambda image: image.copy(),
        bbox_padding=0.1,
    )
    item = dataset[0]
    assert item["image"].size == (23, 17)
    assert item["image"].mode == "RGB"
    assert item["vehicle_id"] == 7
    assert item["camera_id"] == 2
    assert item["label"] == 0
    assert item["image_id"] == "vehicle"
    assert torch.equal(item["bbox"], torch.tensor([-10.0, -5.0, 30.0, 20.0]))


def test_default_eval_transform_has_expected_shape(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (90, 60), color=128).save(image_dir / "gray.jpg")
    row = {"image_id": "gray", "x": 0, "y": 0, "w": 90, "h": 60}
    item = VehicleDataset(row, image_dir, image_size=64, bbox_padding=0)[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["image"].dtype == torch.float32


def test_required_augmentations_are_present_and_executable() -> None:
    train_transform = build_train_transform(64)
    names = {type(operation).__name__ for operation in train_transform.transforms}
    assert {
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "ColorJitter",
        "RandomRotation",
        "RandomErasing",
    }.issubset(names)
    assert train_transform(Image.new("RGB", (100, 80))).shape == (3, 64, 64)
    assert build_eval_transform(64)(Image.new("RGB", (100, 80))).shape == (
        3,
        64,
        64,
    )


def test_rectangular_input_size_and_letterbox_mode() -> None:
    image = Image.new("RGB", (100, 40), color=(255, 255, 255))
    direct = build_eval_transform([40, 64])(image)
    letterbox = build_eval_transform([40, 64], resize_mode="letterbox")(image)

    assert direct.shape == (3, 40, 64)
    assert letterbox.shape == (3, 40, 64)
    # The wide source needs vertical padding only in the aspect-preserving path.
    assert torch.max(torch.abs(letterbox[:, 0, 0])).item() < 0.01
    assert torch.min(letterbox[:, 20, 20]).item() > 1.0


def test_letterbox_train_transform_replaces_random_resized_crop() -> None:
    transform = build_train_transform(
        (40, 64),
        resize_mode="letterbox",
        horizontal_flip_probability=0.0,
        random_erasing_probability=0.0,
    )
    names = {type(operation).__name__ for operation in transform.transforms}
    assert "LetterboxResize" in names
    assert "RandomResizedCrop" not in names
    assert transform(Image.new("RGB", (70, 30))).shape == (3, 40, 64)


@pytest.mark.parametrize(
    ("image_size", "resize_mode"),
    [([64], "direct"), ([64, 0], "direct"), ([64, 64], "stretch")],
)
def test_invalid_extended_transform_configuration_is_rejected(
    image_size, resize_mode
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_eval_transform(image_size, resize_mode=resize_mode)


@pytest.mark.parametrize(
    "frame, message",
    [
        (
            pd.DataFrame(
                [
                    {"image_id": "a", "x": 0, "y": 0, "w": 1, "h": 1},
                    {"image_id": "a", "x": 0, "y": 0, "w": 1, "h": 1},
                ]
            ),
            "unique",
        ),
        (
            pd.DataFrame(
                [{"image_id": "a", "x": 0, "y": 0, "w": 0, "h": 1}]
            ),
            "strictly positive",
        ),
    ],
)
def test_annotation_validation_rejects_unsafe_rows(frame, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_annotations(frame)


def test_dataset_rejects_path_traversal(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    row = {"image_id": "../outside", "x": 0, "y": 0, "w": 1, "h": 1}
    dataset = VehicleDataset(row, image_dir, transform=lambda image: image)
    with pytest.raises(ValueError, match="escapes image directory"):
        dataset[0]
