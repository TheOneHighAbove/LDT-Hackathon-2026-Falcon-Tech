from __future__ import annotations

import pytest
import torch
from torch import nn

from src.pooling import GeM, GeM2d


def test_p_one_matches_average_pooling_for_positive_features() -> None:
    features = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]]
    )
    pool = GeM2d(p=1.0, trainable=False)

    actual = pool(features)
    expected = features.mean(dim=(-2, -1))
    assert actual.shape == (1, 2)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_non_flattened_output_and_alias() -> None:
    assert GeM is GeM2d
    output = GeM2d(p=3.0, flatten=False)(torch.ones(2, 5, 3, 7))
    assert output.shape == (2, 5, 1, 1)
    assert torch.allclose(output, torch.ones_like(output), atol=1e-6)


def test_larger_exponent_moves_towards_spatial_maximum() -> None:
    features = torch.tensor([[[[1.0, 2.0, 8.0]]]])
    mean = GeM2d(p=1.0, trainable=False)(features)
    gem = GeM2d(p=8.0, trainable=False)(features)
    maximum = features.amax(dim=(-2, -1))
    assert torch.all(mean < gem)
    assert torch.all(gem < maximum)


def test_trainable_exponent_receives_finite_gradient_and_remains_positive() -> None:
    pool = GeM2d(p=3.0, trainable=True)
    features = torch.rand(2, 3, 4, 5, requires_grad=True) + 0.1
    pool(features).sum().backward()

    assert isinstance(pool.raw_p, nn.Parameter)
    assert pool.raw_p.grad is not None
    assert torch.isfinite(pool.raw_p.grad)
    assert features.grad_fn is not None

    with torch.no_grad():
        pool.raw_p.fill_(-100.0)
    assert pool.p.item() > 0.0
    assert torch.isfinite(pool(features)).all()


def test_frozen_exponent_is_a_buffer_not_a_parameter() -> None:
    pool = GeM2d(p=3.0, trainable=False)
    assert not list(pool.parameters())
    assert "raw_p" in dict(pool.named_buffers())


def test_half_precision_extremes_stay_finite_and_preserve_dtype() -> None:
    features = torch.tensor(
        [[[[1e-4, 1.0], [1000.0, 65000.0]]]], dtype=torch.float16
    )
    output = GeM2d(p=12.0)(features)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": 0.0},
        {"p": float("nan")},
        {"eps": 0.0},
        {"min_p": 0.0},
        {"p": 0.001, "min_p": 0.001},
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        GeM2d(**kwargs)


def test_invalid_feature_shape_and_dtype_are_rejected() -> None:
    pool = GeM2d()
    with pytest.raises(ValueError, match="shape"):
        pool(torch.ones(2, 3, 4))
    with pytest.raises(TypeError, match="floating"):
        pool(torch.ones(2, 3, 4, 5, dtype=torch.int64))
