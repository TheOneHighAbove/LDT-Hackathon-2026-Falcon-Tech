from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from src.config import validate_config
from src.train import _make_optimizer_scheduler, _optimizer_learning_rates


class _OptimizerFixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(5, 7), nn.ReLU())
        self.projection = nn.Linear(7, 4, bias=False)
        self.bnneck = nn.BatchNorm1d(4)
        self.pool = nn.Sequential(nn.Linear(7, 3), nn.Linear(3, 4, bias=False))


def _optimizer_config(*, local_lr: float | None = None) -> dict:
    training = {
        "lr_backbone": 1e-5,
        "lr_head": 2e-4,
        "lr_classifier": 3e-4,
        "weight_decay": 0.01,
        "warmup_epochs": 0,
        "min_lr_ratio": 0.1,
    }
    if local_lr is not None:
        training["lr_local_head"] = local_lr
    return {"training": training}


def _validation_config(*, pooling: str = "global_local") -> dict:
    return {
        "seed": 1,
        "paths": {},
        "model": {
            "backbone": "unused",
            "embedding_dim": 8,
            "input_size": 64,
            "pooling": pooling,
        },
        "data": {},
        "training": {
            "identities_per_batch": 2,
            "instances_per_identity": 2,
        },
        "inference": {},
    }


def _parameter_ids(parameters) -> list[int]:
    return [id(parameter) for parameter in parameters]


def test_legacy_optimizer_layout_and_learning_rate_history_are_unchanged() -> None:
    model = _OptimizerFixtureModel()
    criterion = nn.Linear(4, 6, bias=False)
    config = _optimizer_config()

    optimizer, _ = _make_optimizer_scheduler(model, criterion, config, epochs=5)

    assert len(optimizer.param_groups) == 3
    assert _parameter_ids(optimizer.param_groups[0]["params"]) == _parameter_ids(
        model.backbone.parameters()
    )
    assert _parameter_ids(optimizer.param_groups[1]["params"]) == _parameter_ids(
        list(model.projection.parameters())
        + list(model.bnneck.parameters())
        + list(model.pool.parameters())
    )
    assert _parameter_ids(optimizer.param_groups[2]["params"]) == _parameter_ids(
        criterion.parameters()
    )
    assert _optimizer_learning_rates(optimizer, config) == {
        "lr_backbone": 1e-5,
        "lr_head": 2e-4,
        "lr_classifier": 3e-4,
    }


def test_local_head_group_has_requested_lr_and_covers_every_parameter_once() -> None:
    model = _OptimizerFixtureModel()
    criterion = nn.Linear(4, 6)
    config = _optimizer_config(local_lr=7e-4)

    optimizer, _ = _make_optimizer_scheduler(model, criterion, config, epochs=5)

    assert len(optimizer.param_groups) == 4
    assert _parameter_ids(optimizer.param_groups[1]["params"]) == _parameter_ids(
        list(model.projection.parameters()) + list(model.bnneck.parameters())
    )
    assert _parameter_ids(optimizer.param_groups[2]["params"]) == _parameter_ids(
        model.pool.parameters()
    )
    assert _parameter_ids(optimizer.param_groups[3]["params"]) == _parameter_ids(
        criterion.parameters()
    )
    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    expected = [
        parameter
        for parameter in list(model.parameters()) + list(criterion.parameters())
        if parameter.requires_grad
    ]
    assert len(_parameter_ids(grouped)) == len(set(_parameter_ids(grouped)))
    assert set(_parameter_ids(grouped)) == set(_parameter_ids(expected))
    assert _optimizer_learning_rates(optimizer, config) == {
        "lr_backbone": 1e-5,
        "lr_head": 2e-4,
        "lr_local_head": 7e-4,
        "lr_classifier": 3e-4,
    }


def test_local_head_optimizer_and_scheduler_resume_preserve_group_layout() -> None:
    config = _optimizer_config(local_lr=7e-4)
    model = _OptimizerFixtureModel()
    criterion = nn.Linear(4, 6)
    optimizer, scheduler = _make_optimizer_scheduler(
        model, criterion, config, epochs=5
    )
    sum(parameter.sum() for parameter in model.parameters()).backward()
    sum(parameter.sum() for parameter in criterion.parameters()).backward()
    optimizer.step()
    scheduler.step()
    optimizer_state = deepcopy(optimizer.state_dict())
    scheduler_state = deepcopy(scheduler.state_dict())

    restored_model = _OptimizerFixtureModel()
    restored_criterion = nn.Linear(4, 6)
    restored_optimizer, restored_scheduler = _make_optimizer_scheduler(
        restored_model, restored_criterion, config, epochs=5
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)

    assert len(restored_optimizer.param_groups) == 4
    assert _optimizer_learning_rates(
        restored_optimizer, config
    ) == _optimizer_learning_rates(optimizer, config)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_positive_local_head_lr_is_valid_only_for_global_local_pooling() -> None:
    config = _validation_config()
    config["training"]["lr_local_head"] = 5e-4
    validate_config(config)

    config["model"]["pooling"] = "avg"
    with pytest.raises(ValueError, match="only valid.*global_local"):
        validate_config(config)


@pytest.mark.parametrize(
    "value",
    [0.0, -1e-4, float("inf"), float("-inf"), float("nan"), True, "0.001"],
)
def test_invalid_local_head_learning_rate_is_rejected(value) -> None:
    config = _validation_config()
    config["training"]["lr_local_head"] = value
    with pytest.raises((TypeError, ValueError), match="lr_local_head"):
        validate_config(config)


def test_absent_local_head_lr_retains_valid_legacy_pooling_configs() -> None:
    for pooling in ("avg", "gem", "global_local"):
        validate_config(_validation_config(pooling=pooling))
