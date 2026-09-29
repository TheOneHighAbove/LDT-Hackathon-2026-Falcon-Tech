import inspect

import pytest


torch = pytest.importorskip("torch", reason="model tests require PyTorch")
try:
    import timm  # noqa: F401
except Exception as exc:  # timm can also fail when torchvision is incompatible
    pytest.skip(
        f"model tests require a working timm installation: {exc}",
        allow_module_level=True,
    )

from src.model import DEFAULT_BACKBONE, VehicleReIDModel


def test_model_defaults_are_offline_safe():
    signature = inspect.signature(VehicleReIDModel)
    assert signature.parameters["backbone_name"].default == DEFAULT_BACKBONE
    assert signature.parameters["embedding_dim"].default == 512
    assert signature.parameters["pretrained"].default is False


def test_eval_batch_one_returns_normalized_float_embedding_without_download():
    model = VehicleReIDModel("resnet18", embedding_dim=32, pretrained=False)
    model.eval()
    images = torch.randn(1, 3, 64, 64)

    with torch.inference_mode():
        embeddings = model(images)

    assert embeddings.shape == (1, 32)
    assert embeddings.dtype == torch.float32
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(1),
        atol=1e-5,
    )


def test_training_dictionary_contains_bnneck_and_projection_features():
    model = VehicleReIDModel("resnet18", embedding_dim=24, pretrained=False)
    model.train()
    output = model(torch.randn(2, 3, 64, 64), return_dict=True)

    assert output["embeddings"].shape == (2, 24)
    assert output["features"].shape == (2, 24)
    assert output["bn_features"].shape == (2, 24)
    assert model.projection.bias is None


def test_checkpoint_config_never_requests_pretrained_download():
    model = VehicleReIDModel("resnet18", embedding_dim=16, pretrained=False)
    config = model.get_config()
    assert config == {
        "backbone_name": "resnet18",
        "embedding_dim": 16,
        "pretrained": False,
        "backbone_kwargs": {},
    }


def test_reserved_backbone_kwargs_are_rejected_before_model_creation():
    with pytest.raises(ValueError, match="cannot override"):
        VehicleReIDModel(
            "resnet18",
            pretrained=False,
            backbone_kwargs={"num_classes": 10},
        )


def test_gem_model_uses_trainable_spatial_pool_and_records_metadata():
    model = VehicleReIDModel(
        "resnet18",
        embedding_dim=16,
        pretrained=False,
        pooling="gem",
        gem_p=4.0,
    ).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))

    assert output.shape == (1, 16)
    assert isinstance(model.pool.raw_p, torch.nn.Parameter)
    assert model.get_config() == {
        "backbone_name": "resnet18",
        "embedding_dim": 16,
        "pretrained": False,
        "backbone_kwargs": {},
        "pooling": "gem",
        "gem_p": 4.0,
        "gem_eps": 1e-6,
    }


def test_gem_accepts_nhwc_spatial_backbone_output():
    model = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="gem"
    ).eval()

    class NHWCBackbone(torch.nn.Module):
        def forward(self, images):
            return torch.rand(
                images.shape[0], 2, 3, model.projection.in_features,
                device=images.device,
            )

    model.backbone = NHWCBackbone()
    with torch.inference_mode():
        output = model(torch.rand(2, 3, 32, 32))
    assert output.shape == (2, 8)


def test_gem_rejects_non_spatial_backbone_output():
    model = VehicleReIDModel(
        "resnet18", embedding_dim=8, pretrained=False, pooling="gem"
    )
    model.backbone = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1)
    )
    with pytest.raises(ValueError, match="4D spatial"):
        model(torch.rand(2, model.projection.in_features, 2, 2))


def test_invalid_pooling_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        VehicleReIDModel("resnet18", pooling="median")
