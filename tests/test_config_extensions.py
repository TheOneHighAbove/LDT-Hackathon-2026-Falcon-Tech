from __future__ import annotations

import pytest
import torch

from src.config import (
    normalize_cross_batch_memory_config,
    normalize_input_size,
    normalize_multi_similarity_config,
    validate_config,
)
from src.engine import (
    build_inference_checkpoint,
    create_model_from_config,
    load_inference_checkpoint,
)


def minimal_config() -> dict:
    return {
        "seed": 1,
        "paths": {},
        "model": {
            "backbone": "resnet18",
            "embedding_dim": 8,
            "input_size": 64,
        },
        "data": {},
        "training": {"identities_per_batch": 2, "instances_per_identity": 2},
        "inference": {},
    }


def test_rectangular_input_and_gem_config_are_valid():
    config = minimal_config()
    config["model"].update(
        {"input_size": [64, 96], "pooling": "gem", "gem_p": 4.0}
    )
    config["data"]["resize_mode"] = "letterbox"
    validate_config(config)
    assert normalize_input_size(config["model"]["input_size"]) == (64, 96)

    model = create_model_from_config(config, pretrained=False)
    assert model.pooling == "gem"
    assert model.gem_p == 4.0


def test_global_local_config_is_valid_and_factory_forwards_parameters():
    config = minimal_config()
    config["model"].update(
        {
            "pooling": "global_local",
            "local_heads": 3,
            "local_temperature": 0.7,
        }
    )

    validate_config(config)
    model = create_model_from_config(config, pretrained=False)

    assert model.pooling == "global_local"
    assert model.local_heads == 3
    assert model.local_temperature == 0.7
    assert model.pool.num_heads == 3
    assert model.pool.temperature == 0.7


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "input_size", [64]),
        ("model", "input_size", [64, 16]),
        ("model", "pooling", "median"),
        ("model", "gem_p", 0.0),
        ("data", "resize_mode", "crop"),
    ],
)
def test_invalid_experimental_configuration_is_rejected(section, key, value):
    config = minimal_config()
    config[section][key] = value
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("local_heads", 0),
        ("local_heads", -1),
        ("local_heads", True),
        ("local_heads", 2.5),
        ("local_temperature", 0.0),
        ("local_temperature", -0.5),
        ("local_temperature", True),
        ("local_temperature", "0.5"),
        ("local_temperature", float("inf")),
        ("local_temperature", float("nan")),
    ],
)
def test_invalid_global_local_configuration_is_rejected(key, value):
    config = minimal_config()
    config["model"].update({"pooling": "global_local", key: value})
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


def test_checkpoint_metadata_serializes_geometry_and_gem_model():
    config = minimal_config()
    config["model"]["pooling"] = "gem"
    model = create_model_from_config(config, pretrained=False)
    checkpoint = build_inference_checkpoint(
        model,
        input_size=[64, 96],
        resize_mode="letterbox",
        bbox_padding=0.05,
        refusal_threshold=0.3,
    )

    assert checkpoint["preprocessing"]["input_size"] == [64, 96]
    assert checkpoint["preprocessing"]["resize_mode"] == "letterbox"
    assert checkpoint["model"]["pooling"] == "gem"


def test_gem_checkpoint_round_trip_is_strict_and_offline(tmp_path):
    config = minimal_config()
    config["model"]["pooling"] = "gem"
    model = create_model_from_config(config, pretrained=False)
    checkpoint = build_inference_checkpoint(
        model,
        input_size=[64, 96],
        resize_mode="letterbox",
        bbox_padding=0.05,
        refusal_threshold=0.3,
    )
    path = tmp_path / "gem.pt"
    torch.save(checkpoint, path)

    restored, restored_checkpoint = load_inference_checkpoint(path)
    assert restored.pooling == "gem"
    assert restored_checkpoint["preprocessing"]["input_size"] == [64, 96]
    assert torch.allclose(restored.pool.p, model.pool.p)


def test_missing_new_fields_retain_baseline_defaults():
    config = minimal_config()
    validate_config(config)
    model = create_model_from_config(config, pretrained=False)
    checkpoint = build_inference_checkpoint(
        model,
        input_size=64,
        bbox_padding=0.05,
        refusal_threshold=0.3,
    )

    assert model.pooling == "avg"
    assert checkpoint["preprocessing"]["input_size"] == 64
    assert checkpoint["preprocessing"]["resize_mode"] == "direct"
    assert "pooling" not in checkpoint["model"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("arcface_subcenters", 0),
        ("arcface_subcenters", True),
        ("triplet_margin_mode", "hinge-ish"),
        ("triplet_positive_mining", "same_camera"),
    ],
)
def test_invalid_metric_learning_configuration_is_rejected(key, value):
    config = minimal_config()
    config["training"][key] = value
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


def test_cross_batch_memory_config_defaults_and_valid_values():
    assert normalize_cross_batch_memory_config(None) == {
        "enabled": False,
        "capacity": 2048,
    }
    config = minimal_config()
    config["training"]["cross_batch_memory"] = {
        "enabled": True,
        "capacity": 4096,
    }
    validate_config(config)


@pytest.mark.parametrize(
    "value",
    [
        True,
        {"enabled": 1},
        {"enabled": True, "capacity": 0},
        {"enabled": True, "capacity": True},
        {"enabled": True, "capacity": 32, "capcity": 64},
    ],
)
def test_invalid_cross_batch_memory_config_is_rejected(value):
    config = minimal_config()
    config["training"]["cross_batch_memory"] = value
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


def test_enabled_cross_batch_memory_requires_triplet_objective():
    config = minimal_config()
    config["training"].update(
        {
            "cross_batch_memory": {"enabled": True, "capacity": 16},
            "triplet_weight": 0.0,
        }
    )
    with pytest.raises(ValueError, match="positive.*triplet_weight"):
        validate_config(config)


def test_multi_similarity_config_defaults_and_valid_values():
    assert normalize_multi_similarity_config(None) == {
        "alpha": 2.0,
        "beta": 50.0,
        "base": 0.5,
        "epsilon": 0.1,
    }
    config = minimal_config()
    config["training"].update(
        {
            "metric_loss": "multi_similarity",
            "multi_similarity": {
                "alpha": 3,
                "beta": 40,
                "base": 0.4,
                "epsilon": 0.2,
            },
            "cross_batch_memory": {"enabled": True, "capacity": 32},
        }
    )
    validate_config(config)


@pytest.mark.parametrize(
    "value",
    [
        True,
        {"alpha": 0.0},
        {"beta": float("inf")},
        {"base": 1.01},
        {"epsilon": -0.01},
        {"epsilon": 2.01},
        {"alpha": 2.0, "gamma": 1.0},
    ],
)
def test_invalid_multi_similarity_config_is_rejected(value):
    config = minimal_config()
    config["training"]["multi_similarity"] = value
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


def test_unknown_metric_loss_is_rejected():
    config = minimal_config()
    config["training"]["metric_loss"] = "circle"
    with pytest.raises(ValueError, match="training.metric_loss"):
        validate_config(config)
