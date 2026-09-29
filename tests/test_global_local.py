from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from src.pooling import SpatialSaliencyLocalBranch


def test_saliency_branch_is_neutral_and_attention_is_normalized() -> None:
    branch = SpatialSaliencyLocalBranch(8, 12, num_heads=3)
    features = torch.randn(2, 8, 4, 5)

    weights = branch.attention_weights(features)
    residual = branch(features)

    assert weights.shape == (2, 3, 20)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3))
    assert residual.shape == (2, 12)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_saliency_branch_propagates_gradients_after_gate_opens() -> None:
    branch = SpatialSaliencyLocalBranch(6, 10, num_heads=2)
    with torch.no_grad():
        branch.residual_gate.fill_(0.25)
    features = torch.randn(3, 6, 3, 4, requires_grad=True)

    branch(features).square().mean().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for parameter in branch.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(branch.saliency.weight.grad) > 0
    assert torch.count_nonzero(branch.projection.weight.grad) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": 0, "embedding_dim": 8},
        {"channels": 4, "embedding_dim": 0},
        {"channels": 4, "embedding_dim": 8, "num_heads": 0},
        {"channels": 4, "embedding_dim": 8, "temperature": 0.0},
        {"channels": 4, "embedding_dim": 8, "temperature": float("nan")},
    ],
)
def test_saliency_branch_rejects_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        SpatialSaliencyLocalBranch(**kwargs)


def test_saliency_branch_rejects_wrong_shape_channels_and_dtype() -> None:
    branch = SpatialSaliencyLocalBranch(4, 8)
    with pytest.raises(ValueError, match="shape"):
        branch(torch.rand(2, 4, 3))
    with pytest.raises(ValueError, match="channels"):
        branch(torch.rand(2, 5, 3, 3))
    with pytest.raises(TypeError, match="floating"):
        branch(torch.ones(2, 4, 3, 3, dtype=torch.int64))


try:
    import timm  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent dependency
    pytestmark = pytest.mark.skip(reason=f"working timm required: {exc}")

from src.engine import load_inference_checkpoint, save_inference_checkpoint
from src.model import VehicleReIDModel
from src.train import _initialize_model_from_checkpoint


def test_global_local_model_shape_gradient_and_metadata() -> None:
    model = VehicleReIDModel(
        "resnet18",
        embedding_dim=24,
        pretrained=False,
        pooling="global_local",
        local_heads=3,
        local_temperature=0.75,
    ).train()
    with torch.no_grad():
        model.pool.residual_gate.fill_(0.2)
    result = model(torch.randn(2, 3, 64, 64), return_dict=True)

    assert result["embeddings"].shape == (2, 24)
    assert result["features"].shape == (2, 24)
    assert torch.allclose(
        torch.linalg.vector_norm(result["embeddings"], dim=1), torch.ones(2)
    )
    result["features"].square().mean().backward()
    assert model.pool.saliency.weight.grad is not None
    assert torch.count_nonzero(model.pool.saliency.weight.grad) > 0
    assert model.get_config() == {
        "backbone_name": "resnet18",
        "embedding_dim": 24,
        "pretrained": False,
        "backbone_kwargs": {},
        "pooling": "global_local",
        "local_heads": 3,
        "local_temperature": 0.75,
    }


def test_avg_warm_start_preserves_shared_state_and_exact_output() -> None:
    torch.manual_seed(11)
    avg = VehicleReIDModel(
        "resnet18", embedding_dim=16, pretrained=False, pooling="avg"
    ).eval()
    torch.manual_seed(29)
    global_local = VehicleReIDModel(
        "resnet18",
        embedding_dim=16,
        pretrained=False,
        pooling="global_local",
        local_heads=2,
    ).eval()
    with torch.no_grad():
        global_local.pool.residual_gate.fill_(0.9)

    global_local.warm_start_from_avg_state_dict(avg.state_dict())

    assert global_local.pool.residual_gate.item() == 0.0
    for name, value in avg.state_dict().items():
        assert torch.equal(global_local.state_dict()[name], value), name
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(avg(images), global_local(images))


def test_training_init_automatically_uses_guarded_avg_warm_start() -> None:
    avg = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="avg"
    ).eval()
    global_local = VehicleReIDModel(
        "resnet18",
        embedding_dim=8,
        pretrained=False,
        pooling="global_local",
        local_heads=2,
    ).eval()
    with torch.no_grad():
        global_local.pool.residual_gate.fill_(0.8)
    checkpoint = {"model": avg.get_config(), "model_state": avg.state_dict()}

    report = _initialize_model_from_checkpoint(
        global_local, checkpoint, allow_partial_init=False
    )

    assert report["mode"] == "avg_to_global_local_strict"
    assert report["source_pooling"] == "avg"
    assert report["target_pooling"] == "global_local"
    assert report["unexpected_keys"] == []
    assert report["missing_keys"]
    assert all(name.startswith("pool.") for name in report["missing_keys"])
    assert global_local.pool.residual_gate.item() == 0.0
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(avg(images), global_local(images))


def test_training_init_preserves_strict_and_explicit_partial_behavior() -> None:
    source = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="avg"
    )
    strict_target = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="avg"
    )
    checkpoint = {"model": source.get_config(), "model_state": source.state_dict()}
    strict_report = _initialize_model_from_checkpoint(strict_target, checkpoint)
    assert strict_report["mode"] == "strict"
    assert strict_report["missing_keys"] == []
    assert strict_report["unexpected_keys"] == []

    partial_target = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="avg"
    )
    partial_state = OrderedDict(source.state_dict())
    partial_state.pop("projection.weight")
    partial_report = _initialize_model_from_checkpoint(
        partial_target,
        {"model": source.get_config(), "model_state": partial_state},
        allow_partial_init=True,
    )
    assert partial_report["mode"] == "partial"
    assert partial_report["missing_keys"] == ["projection.weight"]
    assert partial_report["unexpected_keys"] == []


def test_avg_warm_start_rejects_partial_or_wrong_state_before_mutation() -> None:
    avg = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="avg"
    )
    global_local = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="global_local"
    )
    original_projection = global_local.projection.weight.detach().clone()
    broken = OrderedDict(avg.state_dict())
    broken.pop("projection.weight")

    with pytest.raises(ValueError, match="incompatible"):
        global_local.warm_start_from_avg_state_dict(broken)

    assert torch.equal(global_local.projection.weight, original_projection)


def test_global_local_checkpoint_roundtrip(tmp_path) -> None:
    model = VehicleReIDModel(
        "resnet18",
        embedding_dim=12,
        pretrained=False,
        pooling="global_local",
        local_heads=2,
        local_temperature=0.8,
    ).eval()
    with torch.no_grad():
        model.pool.residual_gate.fill_(0.35)
        model.pool.saliency.weight.normal_(std=0.01)
    path = tmp_path / "global_local.pt"
    save_inference_checkpoint(
        path,
        model,
        input_size=64,
        bbox_padding=0.05,
        refusal_threshold=0.8,
    )

    restored, checkpoint = load_inference_checkpoint(path)

    assert checkpoint["model"]["pooling"] == "global_local"
    assert checkpoint["model"]["local_heads"] == 2
    assert checkpoint["model"]["local_temperature"] == 0.8
    assert restored.pooling == "global_local"
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name
    images = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.allclose(model(images), restored(images), atol=1e-7)
