from __future__ import annotations

import pickle

import pytest
import torch
from PIL import Image

from src.input_geometry import (
    LetterboxResize,
    build_letterbox_eval_transform,
    build_letterbox_train_transform,
    letterbox_resize,
    normalized_zero_fill,
)


def test_letterbox_preserves_aspect_and_centres_odd_padding() -> None:
    image = Image.new("RGB", (8, 4), color=(255, 0, 0))
    output = letterbox_resize(image, (10, 10), fill=(1, 2, 3))

    assert output.size == (10, 10)
    # 8x4 -> 10x5; two rows above and three below are deterministic padding.
    assert output.getpixel((0, 0)) == (1, 2, 3)
    assert output.getpixel((5, 2)) == (255, 0, 0)
    assert output.getpixel((5, 6)) == (255, 0, 0)
    assert output.getpixel((5, 7)) == (1, 2, 3)


def test_rectangular_target_uses_height_width_convention() -> None:
    image = Image.new("RGB", (4, 8), color=(0, 255, 0))
    output = letterbox_resize(image, (6, 10), fill=(9, 9, 9))

    assert output.size == (10, 6)
    # 4x8 -> 3x6, horizontally centred at x=3..5.
    assert output.getpixel((2, 3)) == (9, 9, 9)
    assert output.getpixel((3, 3)) == (0, 255, 0)
    assert output.getpixel((5, 3)) == (0, 255, 0)
    assert output.getpixel((6, 3)) == (9, 9, 9)


def test_letterbox_is_deterministic_and_wrapper_is_pickleable() -> None:
    pixels = bytes(range(3 * 4 * 5))
    image = Image.frombytes("RGB", (4, 5), pixels)
    resize = pickle.loads(pickle.dumps(LetterboxResize(17)))

    first = resize(image)
    second = resize(image)
    assert first.tobytes() == second.tobytes()


def test_imagenet_padding_normalizes_close_to_zero() -> None:
    image = Image.new("RGB", (8, 2), color=(255, 255, 255))
    tensor = build_letterbox_eval_transform(16)(image)

    assert normalized_zero_fill() == (124, 116, 104)
    assert tensor.shape == (3, 16, 16)
    assert torch.max(torch.abs(tensor[:, 0, 0])).item() < 0.01
    assert torch.min(tensor[:, 7, :]).item() > 1.0


def test_train_transform_retains_full_output_shape() -> None:
    transform = build_letterbox_train_transform(
        (32, 48),
        horizontal_flip_probability=0.0,
        random_erasing_probability=0.0,
    )
    output = transform(Image.new("RGB", (100, 50), color=(90, 120, 150)))
    assert output.shape == (3, 32, 48)
    assert output.dtype == torch.float32


@pytest.mark.parametrize("size", [0, -1, (10, 0), (10, 2.5), True])
def test_invalid_target_size_is_rejected(size) -> None:
    with pytest.raises((TypeError, ValueError)):
        LetterboxResize(size)


def test_invalid_fill_and_statistics_are_rejected() -> None:
    with pytest.raises(ValueError, match="fill"):
        LetterboxResize(16, fill=(0, 0, 300))
    with pytest.raises(ValueError, match="three"):
        normalized_zero_fill((0.5, 0.5))
    with pytest.raises(ValueError, match="positive"):
        build_letterbox_eval_transform(16, std=(1.0, 0.0, 1.0))
