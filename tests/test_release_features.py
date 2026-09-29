from __future__ import annotations

import numpy as np
from PIL import Image

from src.release_features import color_descriptor


def _reference_color_descriptor(crop: Image.Image) -> np.ndarray:
    array = np.asarray(crop.resize((64, 64)).convert("HSV"), dtype=np.uint8)
    regions = (
        array,
        array[:32, :32],
        array[:32, 32:],
        array[32:, :32],
        array[32:, 32:],
    )
    values: list[np.ndarray] = []
    for region in regions:
        for channel, bins in ((0, 16), (1, 8), (2, 8)):
            histogram = np.histogram(
                region[..., channel], bins=bins, range=(0, 256)
            )[0].astype(np.float32)
            histogram /= max(float(histogram.sum()), 1.0)
            values.append(np.sqrt(histogram))
    descriptor = np.concatenate(values)
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor.astype(np.float32, copy=False)


def test_color_descriptor_is_bit_exact_with_locked_histogram() -> None:
    rng = np.random.default_rng(20260927)
    for height, width in ((5, 7), (64, 64), (101, 37)):
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        actual = color_descriptor(image)
        expected = _reference_color_descriptor(image)

        assert np.array_equal(actual, expected)
        assert actual.dtype == np.float32
        assert actual.shape == (160,)
