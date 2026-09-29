"""Embedding model used by the vehicle ReID training and inference pipelines.

The default forward contract is intentionally small: ``model(images)`` returns
only L2-normalized float32 embeddings.  Training code can request the
intermediate representations with ``return_dict=True`` so that metric losses
can use the pre-BN projection while ArcFace uses the normalized embedding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pooling import GeM2d, SpatialSaliencyLocalBranch


DEFAULT_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"


def _import_timm() -> Any:
    """Import timm lazily so loss-only/offline tooling can import ``src``."""

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        raise ImportError(
            "VehicleReIDModel requires the optional dependency 'timm'. "
            "Install the project requirements before constructing the model."
        ) from exc
    return timm


class VehicleReIDModel(nn.Module):
    """A timm backbone followed by a linear projection and BNNeck.

    Parameters
    ----------
    backbone_name:
        Any timm model that supports ``num_classes=0`` and ``global_pool='avg'``.
    embedding_dim:
        Size of the retrieval descriptor.
    pretrained:
        Explicit switch for loading timm weights.  It defaults to ``False`` so
        checkpoint-based inference is safe in a network-isolated container.
    backbone_kwargs:
        Optional extra arguments forwarded to ``timm.create_model``.  The
        arguments controlled by this class cannot be overridden here.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        embedding_dim: int = 512,
        *,
        pretrained: bool = False,
        backbone_kwargs: Mapping[str, Any] | None = None,
        pooling: str = "avg",
        gem_p: float = 3.0,
        gem_eps: float = 1e-6,
        local_heads: int = 4,
        local_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not backbone_name:
            raise ValueError("backbone_name must be a non-empty timm model name")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be a bool")
        if (
            isinstance(local_heads, bool)
            or not isinstance(local_heads, int)
            or local_heads <= 0
        ):
            raise ValueError("local_heads must be a positive integer")
        if isinstance(local_temperature, bool) or not isinstance(
            local_temperature, (int, float)
        ):
            raise TypeError("local_temperature must be a finite positive number")
        local_temperature = float(local_temperature)
        if not math.isfinite(local_temperature) or local_temperature <= 0:
            raise ValueError("local_temperature must be finite and positive")
        pooling = str(pooling).lower()
        if pooling not in {"avg", "gem", "global_local"}:
            raise ValueError(
                "pooling must be one of 'avg', 'gem', or 'global_local'"
            )

        extra_kwargs = dict(backbone_kwargs or {})
        reserved = {"pretrained", "num_classes", "global_pool"}.intersection(
            extra_kwargs
        )
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"backbone_kwargs cannot override: {names}")

        timm = _import_timm()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            # global_local deliberately keeps the exact established avg head;
            # its spatial branch reads forward_features separately.  GeM must
            # receive a spatial feature map from the regular forward instead.
            global_pool="" if pooling == "gem" else "avg",
            **extra_kwargs,
        )

        feature_dim = getattr(self.backbone, "num_features", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError(
                f"Backbone {backbone_name!r} does not expose a valid num_features"
            )

        self.backbone_name = backbone_name
        self.embedding_dim = int(embedding_dim)
        self.pretrained = pretrained
        self.backbone_kwargs = extra_kwargs
        self.pooling = pooling
        self.gem_p = float(gem_p)
        self.gem_eps = float(gem_eps)
        self.local_heads = local_heads
        self.local_temperature = local_temperature
        if pooling == "gem":
            self.pool = GeM2d(
                p=self.gem_p,
                eps=self.gem_eps,
                trainable=True,
                flatten=True,
            )
        elif pooling == "global_local":
            self.pool = SpatialSaliencyLocalBranch(
                feature_dim,
                self.embedding_dim,
                num_heads=self.local_heads,
                temperature=self.local_temperature,
            )
        else:
            self.pool = nn.Identity()

        self.projection = nn.Linear(feature_dim, self.embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)
        self._reset_head_parameters()

    def _reset_head_parameters(self) -> None:
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_out")
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)

    def warm_start_from_avg_state_dict(
        self, state_dict: Mapping[str, Tensor]
    ) -> None:
        """Safely initialize ``global_local`` from a legacy ``avg`` state.

        Unlike a general ``strict=False`` load, this method accepts *only* the
        new local-branch keys as missing and validates all shared tensor shapes
        before mutating the module.  Thus a wrong backbone or embedding size
        cannot silently produce a partially initialized experiment.
        """

        if self.pooling != "global_local":
            raise ValueError(
                "avg warm-start is only defined for pooling='global_local'"
            )
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")

        target = self.state_dict()
        provided_keys = set(state_dict)
        target_keys = set(target)
        local_keys = {f"pool.{name}" for name in self.pool.state_dict()}
        missing = target_keys - provided_keys
        unexpected = provided_keys - target_keys
        if missing != local_keys or unexpected:
            raise ValueError(
                "avg warm-start state is incompatible: expected exactly local "
                f"keys to be missing; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for name in sorted(provided_keys):
            value = state_dict[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"state_dict[{name!r}] must be a tensor")
            if value.shape != target[name].shape:
                raise ValueError(
                    f"avg warm-start tensor {name!r} has shape "
                    f"{tuple(value.shape)}, expected {tuple(target[name].shape)}"
                )

        # Reset before loading so the method is neutral even when called on a
        # previously trained global-local instance.
        self.pool.reset_parameters()
        incompatible = self.load_state_dict(state_dict, strict=False)
        if set(incompatible.missing_keys) != local_keys or incompatible.unexpected_keys:
            raise RuntimeError("validated avg warm-start produced unexpected keys")

    @staticmethod
    def _to_feature_vector(features: Tensor) -> Tensor:
        """Coerce common timm outputs to one vector per image."""

        if not isinstance(features, Tensor):
            raise TypeError(
                "The selected timm backbone must return a Tensor when created "
                "with num_classes=0 and global_pool='avg'"
            )
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        elif features.ndim == 3:
            # Defensive fallback for token backbones that ignore global_pool.
            features = features.mean(dim=1)
        elif features.ndim != 2:
            raise ValueError(
                f"Expected a 2D pooled feature tensor, got shape {tuple(features.shape)}"
            )
        return features

    def extract_projected_features(self, images: Tensor) -> Tensor:
        """Return the projection before BNNeck (useful for triplet loss)."""

        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [batch, channels, height, width], got "
                f"{tuple(images.shape)}"
            )
        if self.pooling == "global_local":
            forward_features = getattr(self.backbone, "forward_features", None)
            forward_head = getattr(self.backbone, "forward_head", None)
            if not callable(forward_features) or not callable(forward_head):
                raise TypeError(
                    "global_local pooling requires a timm backbone with "
                    "forward_features and forward_head methods"
                )
            spatial_output = forward_features(images)
            global_output = forward_head(spatial_output)
            backbone_features = self._to_feature_vector(global_output)
            spatial_features = self._to_spatial_feature_map(
                spatial_output, self.projection.in_features, mode="global_local"
            )
            local_residual = self.pool(spatial_features)
        else:
            backbone_output = self.backbone(images)
        if self.pooling == "gem":
            if not isinstance(backbone_output, Tensor) or backbone_output.ndim != 4:
                shape = (
                    tuple(backbone_output.shape)
                    if isinstance(backbone_output, Tensor)
                    else type(backbone_output).__name__
                )
                raise ValueError(
                    "GeM pooling requires a 4D spatial backbone feature map, "
                    f"got {shape}"
                )
            spatial_features = self._to_spatial_feature_map(
                backbone_output, self.projection.in_features, mode="GeM"
            )
            backbone_features = self.pool(spatial_features)
        elif self.pooling == "avg":
            backbone_features = self._to_feature_vector(backbone_output)
        if backbone_features.shape[1] != self.projection.in_features:
            raise RuntimeError(
                "Backbone output dimension changed unexpectedly: expected "
                f"{self.projection.in_features}, got {backbone_features.shape[1]}"
            )
        projected = self.projection(backbone_features)
        if self.pooling == "global_local":
            projected = projected + local_residual
        return projected

    @staticmethod
    def _to_spatial_feature_map(
        features: Tensor, feature_dim: int, *, mode: str
    ) -> Tensor:
        """Convert NCHW/NHWC timm features to the NCHW branch contract."""

        if not isinstance(features, Tensor) or features.ndim != 4:
            shape = (
                tuple(features.shape)
                if isinstance(features, Tensor)
                else type(features).__name__
            )
            raise ValueError(
                f"{mode} pooling requires a 4D spatial backbone feature map, "
                f"got {shape}"
            )
        if features.shape[1] == feature_dim:
            return features
        if features.shape[-1] == feature_dim:
            return features.permute(0, 3, 1, 2)
        raise ValueError(
            "Cannot identify the channel dimension of the spatial backbone "
            f"output {tuple(features.shape)}; expected num_features="
            f"{feature_dim} in dimension 1 or -1"
        )

    def forward(
        self, images: Tensor, *, return_dict: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Create retrieval embeddings.

        ``return_dict=False`` is the inference API.  The returned tensor is
        always float32, including under mixed precision, which keeps cosine
        ranking numerically stable.  In evaluation mode BatchNorm1d supports a
        batch size of one using its stored running statistics.
        """

        features = self.extract_projected_features(images)
        bn_features = self.bnneck(features)
        embeddings = F.normalize(bn_features.float(), p=2, dim=1, eps=1e-12)

        if return_dict:
            return {
                "embeddings": embeddings,
                "features": features,
                "bn_features": bn_features,
            }
        return embeddings

    def get_config(self) -> dict[str, Any]:
        """Return the architecture metadata required to rebuild a checkpoint."""

        config = {
            "backbone_name": self.backbone_name,
            "embedding_dim": self.embedding_dim,
            # Restoring a trained checkpoint must never trigger a download.
            "pretrained": False,
            "backbone_kwargs": dict(self.backbone_kwargs),
        }
        # Omitting the default retains exact metadata compatibility with the
        # original checkpoints, whose missing pooling field also means avg.
        if self.pooling == "gem":
            config.update(
                {
                    "pooling": "gem",
                    "gem_p": self.gem_p,
                    "gem_eps": self.gem_eps,
                }
            )
        elif self.pooling == "global_local":
            config.update(
                {
                    "pooling": "global_local",
                    "local_heads": self.local_heads,
                    "local_temperature": self.local_temperature,
                }
            )
        return config


# Short alias retained for concise configuration-driven factory code.
ReIDModel = VehicleReIDModel


__all__ = ["DEFAULT_BACKBONE", "ReIDModel", "VehicleReIDModel"]
