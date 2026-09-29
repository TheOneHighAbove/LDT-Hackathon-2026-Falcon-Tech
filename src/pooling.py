"""Numerically stable spatial pooling layers for retrieval models."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    """Stable inverse of softplus for positive scalar initialization."""

    return value if value > 20.0 else math.log(math.expm1(value))


class GeM2d(nn.Module):
    """Generalized mean pooling over the two spatial dimensions.

    ``p`` is represented with a softplus transform, so a trainable exponent
    remains positive throughout optimization.  Pooling is evaluated in
    log-space and at least float32, preventing intermediate overflow for
    mixed-precision feature maps.  Like the conventional ReID implementation,
    negative activations are clamped to ``eps`` before taking the power.

    Parameters
    ----------
    p:
        Initial positive exponent.  ``p=1`` is average pooling and increasing
        values approach max pooling.
    eps:
        Positive floor applied to feature activations.
    trainable:
        Whether the exponent is an optimized parameter or a frozen buffer.
    flatten:
        Return ``[N, C]`` when true; otherwise retain ``[N, C, 1, 1]``.
    min_p:
        Strict lower bound for the exponent under the softplus transform.
    """

    def __init__(
        self,
        p: float = 3.0,
        *,
        eps: float = 1e-6,
        trainable: bool = True,
        flatten: bool = True,
        min_p: float = 1e-3,
    ) -> None:
        super().__init__()
        p = float(p)
        eps = float(eps)
        min_p = float(min_p)
        if not math.isfinite(p) or p <= 0:
            raise ValueError("p must be finite and positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(min_p) or min_p <= 0:
            raise ValueError("min_p must be finite and positive")
        if p <= min_p:
            raise ValueError("p must be greater than min_p")

        raw = torch.tensor(_inverse_softplus(p - min_p), dtype=torch.float32)
        if trainable:
            self.raw_p = nn.Parameter(raw)
        else:
            self.register_buffer("raw_p", raw)
        self.eps = eps
        self.flatten = bool(flatten)
        self.min_p = min_p

    @property
    def p(self) -> Tensor:
        """Current positive exponent (a differentiable scalar tensor)."""

        return F.softplus(self.raw_p) + self.min_p

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [batch, channels, height, width], "
                f"got {tuple(features.shape)}"
            )
        height, width = features.shape[-2:]
        if height <= 0 or width <= 0:
            raise ValueError("spatial feature dimensions must be positive")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")

        original_dtype = features.dtype
        work_dtype = (
            torch.float32
            if original_dtype in (torch.float16, torch.bfloat16)
            else original_dtype
        )
        work = features.to(dtype=work_dtype).clamp_min(self.eps)
        exponent = self.p.to(device=features.device, dtype=work_dtype)

        # log(mean(x**p)) / p is equivalent to GeM but does not materialize
        # potentially overflowing powers for float16 or unusually large p.
        log_power = work.log() * exponent
        log_mean_power = torch.logsumexp(
            log_power, dim=(-2, -1), keepdim=True
        ) - math.log(height * width)
        pooled = torch.exp(log_mean_power / exponent)
        pooled = pooled.to(dtype=original_dtype)
        return pooled.flatten(1) if self.flatten else pooled

    def extra_repr(self) -> str:
        return (
            f"p={self.p.detach().item():.4g}, eps={self.eps:g}, "
            f"trainable={isinstance(self.raw_p, nn.Parameter)}, "
            f"flatten={self.flatten}"
        )


class SpatialSaliencyLocalBranch(nn.Module):
    """Learn view-agnostic local residuals from a spatial feature map.

    Each head predicts a content-dependent saliency distribution over all
    spatial locations.  Unlike fixed horizontal stripes, this does not assume
    that the same vehicle part always appears at the same image height, which
    is important when front, rear, and side views are mixed.

    The branch projects the concatenated attended descriptors into the final
    embedding space.  A learnable scalar residual gate is initialized to zero,
    so adding this module to an average-pooling checkpoint initially leaves the
    old descriptor *exactly* unchanged.  The projection is initialized
    asymmetrically; the gate learns first and subsequently exposes distinct
    gradients to the saliency heads instead of locking them into a symmetric
    all-zero solution.

    Parameters
    ----------
    channels:
        Number of channels in the backbone feature map.
    embedding_dim:
        Dimension of the residual added to the global projected descriptor.
    num_heads:
        Number of independently learned spatial saliency maps.
    temperature:
        Positive softmax temperature for the saliency maps.
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        *,
        num_heads: int = 4,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        self.channels = channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.temperature = temperature
        # A per-head bias would add the same constant at every spatial
        # location and therefore cancel exactly inside the spatial softmax.
        self.saliency = nn.Conv2d(channels, num_heads, kernel_size=1, bias=False)
        self.projection = nn.Linear(
            channels * num_heads, embedding_dim, bias=False
        )
        self.residual_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the neutral, average-checkpoint-compatible initialization."""

        # Uniform attention is the least opinionated starting point.  The
        # randomly initialized projection breaks symmetry between the heads,
        # while the zero gate guarantees that it cannot perturb the old model.
        nn.init.zeros_(self.saliency.weight)
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_out")
        nn.init.zeros_(self.residual_gate)

    def attention_weights(self, features: Tensor) -> Tensor:
        """Return normalized ``[N, heads, H*W]`` saliency weights."""

        self._validate_features(features)
        logits = self.saliency(features).flatten(2)
        # Always normalize logits in float32.  This avoids fp16 overflow for a
        # sharply trained attention map while preserving the feature dtype for
        # the batched matrix multiplication below.
        weights = F.softmax(logits.float() / self.temperature, dim=-1)
        return weights.to(dtype=features.dtype)

    def _validate_features(self, features: Tensor) -> None:
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [batch, channels, height, width], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} feature channels, got "
                f"{features.shape[1]}"
            )
        if features.shape[-2] <= 0 or features.shape[-1] <= 0:
            raise ValueError("spatial feature dimensions must be positive")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")

    def forward(self, features: Tensor) -> Tensor:
        weights = self.attention_weights(features)
        values = features.flatten(2).transpose(1, 2)
        attended = torch.bmm(weights, values)
        residual = self.projection(attended.flatten(1))
        gate = torch.tanh(self.residual_gate).to(dtype=residual.dtype)
        return gate * residual

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, temperature={self.temperature:g}, "
            f"gate={torch.tanh(self.residual_gate.detach()).item():.4g}"
        )


# Familiar concise name while retaining the dimension-explicit public class.
GeM = GeM2d


__all__ = ["GeM", "GeM2d", "SpatialSaliencyLocalBranch"]
