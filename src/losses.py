"""ArcFace classification and batch-hard metric losses for vehicle ReID."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    """Additive angular-margin (ArcFace) classification head.

    Passing ``labels=None`` returns scaled cosine logits without applying the
    margin, which is useful for diagnostics.  Margin calculations are performed
    in float32 for stability under automatic mixed precision.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale: float = 30.0,
        margin: float = 0.50,
        easy_margin: bool = False,
        num_subcenters: int = 1,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not 0.0 <= margin < math.pi:
            raise ValueError("margin must be in [0, pi)")
        if (
            isinstance(num_subcenters, bool)
            or not isinstance(num_subcenters, int)
            or num_subcenters < 1
        ):
            raise ValueError("num_subcenters must be a positive integer")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self.num_subcenters = int(num_subcenters)

        # K=1 deliberately retains the historical [classes, embedding_dim]
        # state-dict shape, so existing checkpoints load without conversion.
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features * self.num_subcenters,
                self.in_features,
            )
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_correction = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: Tensor, labels: Tensor | None = None) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.in_features:
            raise ValueError(
                f"embeddings must have shape [batch, {self.in_features}], got "
                f"{tuple(embeddings.shape)}"
            )

        cosine = F.linear(
            F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12),
            F.normalize(self.weight.float(), p=2, dim=1, eps=1e-12),
        )
        if self.num_subcenters > 1:
            cosine = cosine.reshape(
                embeddings.shape[0], self.out_features, self.num_subcenters
            ).max(dim=2).values
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.scale

        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"labels must have shape [{embeddings.shape[0]}], got "
                f"{tuple(labels.shape)}"
            )
        labels = labels.to(device=cosine.device, dtype=torch.long)
        if labels.numel() and (
            torch.any(labels < 0).item() or torch.any(labels >= self.out_features).item()
        ):
            raise ValueError("labels contain a class outside [0, out_features)")

        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0.0, phi, cosine)
        else:
            phi = torch.where(
                cosine > self.threshold,
                phi,
                cosine - self.margin_correction,
            )

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scale={self.scale}, margin={self.margin}, "
            f"easy_margin={self.easy_margin}, "
            f"num_subcenters={self.num_subcenters}"
        )


class ArcFace(ArcMarginProduct):
    """Descriptive alias for :class:`ArcMarginProduct`."""


class BatchHardTripletLoss(nn.Module):
    """Configurable triplet loss with hardest pair mining per batch.

    Anchors without another positive sample (or without any negative class) are
    ignored.  If the whole batch has no valid anchors, a differentiable scalar
    zero is returned instead of NaN.  ``cross_camera_preferred`` restricts an
    anchor's positives to other cameras when possible and falls back to all
    same-identity positives only for anchors that have no cross-camera sample.
    ``cross_camera_only`` is strict and ignores such anchors instead.
    """

    SUPPORTED_METRICS = {"cosine", "euclidean"}
    SUPPORTED_MARGIN_MODES = {"fixed", "soft"}
    SUPPORTED_POSITIVE_MINING = {
        "all",
        "cross_camera_preferred",
        "cross_camera_only",
    }

    def __init__(
        self,
        margin: float = 0.30,
        *,
        metric: str = "cosine",
        normalize_embeddings: bool = True,
        margin_mode: str = "fixed",
        positive_mining: str = "all",
    ) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be non-negative")
        metric = metric.lower()
        if metric not in self.SUPPORTED_METRICS:
            choices = ", ".join(sorted(self.SUPPORTED_METRICS))
            raise ValueError(f"metric must be one of: {choices}")
        margin_mode = margin_mode.lower()
        if margin_mode not in self.SUPPORTED_MARGIN_MODES:
            choices = ", ".join(sorted(self.SUPPORTED_MARGIN_MODES))
            raise ValueError(f"margin_mode must be one of: {choices}")
        positive_mining = positive_mining.lower()
        if positive_mining not in self.SUPPORTED_POSITIVE_MINING:
            choices = ", ".join(sorted(self.SUPPORTED_POSITIVE_MINING))
            raise ValueError(f"positive_mining must be one of: {choices}")
        self.margin = float(margin)
        self.metric = metric
        self.normalize_embeddings = bool(normalize_embeddings)
        self.margin_mode = margin_mode
        self.positive_mining = positive_mining

    def pairwise_distance(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D [batch, dim], got {tuple(embeddings.shape)}"
            )
        vectors = embeddings.float()
        if self.metric == "cosine" or self.normalize_embeddings:
            vectors = F.normalize(vectors, p=2, dim=1, eps=1e-12)
        if self.metric == "cosine":
            # Small negative round-off would otherwise alter the hard-positive max.
            return (1.0 - vectors @ vectors.transpose(0, 1)).clamp_min(0.0)
        return torch.cdist(vectors, vectors, p=2)

    def _cross_distance(self, anchors: Tensor, candidates: Tensor) -> Tensor:
        """Pairwise distances from current anchors to current/memory candidates."""

        anchor_vectors = anchors.float()
        candidate_vectors = candidates.float()
        if self.metric == "cosine" or self.normalize_embeddings:
            anchor_vectors = F.normalize(anchor_vectors, p=2, dim=1, eps=1e-12)
            candidate_vectors = F.normalize(
                candidate_vectors, p=2, dim=1, eps=1e-12
            )
        if self.metric == "cosine":
            return (1.0 - anchor_vectors @ candidate_vectors.transpose(0, 1)).clamp_min(
                0.0
            )
        return torch.cdist(anchor_vectors, candidate_vectors, p=2)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        camera_ids: Tensor | None = None,
        *,
        memory_embeddings: Tensor | None = None,
        memory_labels: Tensor | None = None,
        memory_camera_ids: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D [batch, dim], got {tuple(embeddings.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"labels must have shape [{embeddings.shape[0]}], got "
                f"{tuple(labels.shape)}"
            )
        if camera_ids is not None and (
            camera_ids.ndim != 1 or camera_ids.shape[0] != embeddings.shape[0]
        ):
            raise ValueError(
                f"camera_ids must have shape [{embeddings.shape[0]}], got "
                f"{tuple(camera_ids.shape)}"
            )
        if self.positive_mining != "all" and camera_ids is None:
            raise ValueError(
                f"camera_ids are required for positive_mining={self.positive_mining!r}"
            )
        if memory_embeddings is None:
            if memory_labels is not None or memory_camera_ids is not None:
                raise ValueError(
                    "memory labels/cameras require memory_embeddings"
                )
        else:
            if memory_embeddings.ndim != 2:
                raise ValueError(
                    "memory_embeddings must be 2D [items, dim], got "
                    f"{tuple(memory_embeddings.shape)}"
                )
            if memory_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    "memory_embeddings feature dimension must match embeddings"
                )
            if memory_labels is None or memory_labels.ndim != 1 or (
                memory_labels.shape[0] != memory_embeddings.shape[0]
            ):
                raise ValueError(
                    "memory_labels must have one item per memory embedding"
                )
            if memory_camera_ids is not None and (
                memory_camera_ids.ndim != 1
                or memory_camera_ids.shape[0] != memory_embeddings.shape[0]
            ):
                raise ValueError(
                    "memory_camera_ids must have one item per memory embedding"
                )
            if self.positive_mining != "all" and memory_camera_ids is None:
                raise ValueError(
                    f"memory_camera_ids are required for "
                    f"positive_mining={self.positive_mining!r}"
                )
        # Preserve a gradient path even for an empty or invalid PK batch.
        differentiable_zero = embeddings.sum() * 0.0
        if embeddings.shape[0] < 2:
            # A singleton can still become a valid anchor when its positive and
            # a negative are both present in cross-batch memory.
            if memory_embeddings is None or memory_embeddings.shape[0] == 0:
                return differentiable_zero

        labels = labels.to(device=embeddings.device)
        candidate_embeddings = embeddings
        candidate_labels = labels
        candidate_cameras = (
            camera_ids.to(device=embeddings.device)
            if camera_ids is not None
            else None
        )
        if memory_embeddings is not None and memory_embeddings.shape[0] > 0:
            candidate_embeddings = torch.cat(
                [embeddings, memory_embeddings.to(device=embeddings.device)], dim=0
            )
            assert memory_labels is not None
            candidate_labels = torch.cat(
                [labels, memory_labels.to(device=embeddings.device)], dim=0
            )
            if camera_ids is not None:
                if memory_camera_ids is None:
                    # Cameras are irrelevant in ``all`` mode, so omitting them
                    # from memory remains a supported lightweight use case.
                    if self.positive_mining != "all":
                        raise ValueError(
                            "memory_camera_ids are required for cross-camera mining"
                        )
                    candidate_cameras = None
                else:
                    candidate_cameras = torch.cat(
                        [
                            camera_ids.to(device=embeddings.device),
                            memory_camera_ids.to(device=embeddings.device),
                        ],
                        dim=0,
                    )

        distances = self._cross_distance(embeddings, candidate_embeddings)
        same_identity = labels[:, None].eq(candidate_labels[None, :])
        # Current samples occupy the first B candidate positions.  Exclude only
        # their exact self-pair; queued tensors are detached historical samples.
        self_mask = torch.zeros_like(same_identity)
        self_mask[:, : embeddings.shape[0]] = torch.eye(
            embeddings.shape[0], device=embeddings.device, dtype=torch.bool
        )
        positive_mask = same_identity & ~self_mask
        if self.positive_mining != "all":
            assert camera_ids is not None
            cameras = camera_ids.to(device=embeddings.device)
            assert candidate_cameras is not None
            cross_camera_positive = positive_mask & ~cameras[:, None].eq(
                candidate_cameras[None, :]
            )
            if self.positive_mining == "cross_camera_preferred":
                has_cross_camera = cross_camera_positive.any(dim=1, keepdim=True)
                positive_mask = torch.where(
                    has_cross_camera, cross_camera_positive, positive_mask
                )
            else:
                positive_mask = cross_camera_positive
        negative_mask = ~same_identity
        valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not torch.any(valid_anchor).item():
            return differentiable_zero

        hard_positive = distances.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
        hard_negative = distances.masked_fill(~negative_mask, torch.inf).min(dim=1).values
        distance_difference = hard_positive - hard_negative
        if self.margin_mode == "fixed":
            losses = F.relu(distance_difference + self.margin)
        else:
            # Smooth, margin-free ranking objective.  Unlike the hinge loss it
            # keeps a small gradient after the fixed-margin constraint is met.
            losses = F.softplus(distance_difference)
        return losses[valid_anchor].mean()

    def extra_repr(self) -> str:
        return (
            f"margin={self.margin}, metric={self.metric!r}, "
            f"normalize_embeddings={self.normalize_embeddings}, "
            f"margin_mode={self.margin_mode!r}, "
            f"positive_mining={self.positive_mining!r}"
        )


# Conventional name used by configuration files and training scripts.
TripletLoss = BatchHardTripletLoss


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity metric loss with optional cross-batch candidates.

    Embeddings are L2-normalized internally and compared by cosine similarity.
    The pair-mining rule follows the original Multi-Similarity formulation:
    positives close to the hardest negative and negatives close to the hardest
    positive are retained.  Only current-batch samples act as anchors; queued
    XBM features are detached candidates and never stale anchors.
    """

    SUPPORTED_POSITIVE_MINING = BatchHardTripletLoss.SUPPORTED_POSITIVE_MINING

    def __init__(
        self,
        *,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        epsilon: float = 0.1,
        positive_mining: str = "all",
    ) -> None:
        super().__init__()
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(base)) or not -1.0 <= float(base) <= 1.0:
            raise ValueError("base must be finite and in [-1, 1]")
        if (
            not math.isfinite(float(epsilon))
            or not 0.0 <= float(epsilon) <= 2.0
        ):
            raise ValueError("epsilon must be finite and in [0, 2]")
        positive_mining = positive_mining.lower()
        if positive_mining not in self.SUPPORTED_POSITIVE_MINING:
            choices = ", ".join(sorted(self.SUPPORTED_POSITIVE_MINING))
            raise ValueError(f"positive_mining must be one of: {choices}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.base = float(base)
        self.epsilon = float(epsilon)
        self.positive_mining = positive_mining

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        camera_ids: Tensor | None = None,
        *,
        memory_embeddings: Tensor | None = None,
        memory_labels: Tensor | None = None,
        memory_camera_ids: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D [batch, dim], got {tuple(embeddings.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"labels must have shape [{embeddings.shape[0]}], got "
                f"{tuple(labels.shape)}"
            )
        if camera_ids is not None and (
            camera_ids.ndim != 1 or camera_ids.shape[0] != embeddings.shape[0]
        ):
            raise ValueError(
                f"camera_ids must have shape [{embeddings.shape[0]}], got "
                f"{tuple(camera_ids.shape)}"
            )
        if self.positive_mining != "all" and camera_ids is None:
            raise ValueError(
                f"camera_ids are required for positive_mining={self.positive_mining!r}"
            )
        if memory_embeddings is None:
            if memory_labels is not None or memory_camera_ids is not None:
                raise ValueError("memory labels/cameras require memory_embeddings")
        else:
            if memory_embeddings.ndim != 2:
                raise ValueError("memory_embeddings must be 2D [items, dim]")
            if memory_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    "memory_embeddings feature dimension must match embeddings"
                )
            if memory_labels is None or memory_labels.ndim != 1 or (
                memory_labels.shape[0] != memory_embeddings.shape[0]
            ):
                raise ValueError(
                    "memory_labels must have one item per memory embedding"
                )
            if memory_camera_ids is not None and (
                memory_camera_ids.ndim != 1
                or memory_camera_ids.shape[0] != memory_embeddings.shape[0]
            ):
                raise ValueError(
                    "memory_camera_ids must have one item per memory embedding"
                )
            if self.positive_mining != "all" and memory_camera_ids is None:
                raise ValueError(
                    "memory_camera_ids are required for cross-camera mining"
                )

        differentiable_zero = embeddings.sum() * 0.0
        if embeddings.shape[0] == 0:
            return differentiable_zero

        labels = labels.to(device=embeddings.device)
        candidates = embeddings
        candidate_labels = labels
        candidate_cameras = (
            camera_ids.to(device=embeddings.device)
            if camera_ids is not None
            else None
        )
        if memory_embeddings is not None and memory_embeddings.shape[0] > 0:
            candidates = torch.cat(
                [embeddings, memory_embeddings.to(device=embeddings.device)], dim=0
            )
            assert memory_labels is not None
            candidate_labels = torch.cat(
                [labels, memory_labels.to(device=embeddings.device)], dim=0
            )
            if camera_ids is not None and memory_camera_ids is not None:
                candidate_cameras = torch.cat(
                    [
                        camera_ids.to(device=embeddings.device),
                        memory_camera_ids.to(device=embeddings.device),
                    ],
                    dim=0,
                )

        anchors = F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12)
        candidates = F.normalize(candidates.float(), p=2, dim=1, eps=1e-12)
        similarities = anchors @ candidates.transpose(0, 1)
        same_identity = labels[:, None].eq(candidate_labels[None, :])
        self_mask = torch.zeros_like(same_identity)
        self_mask[:, : embeddings.shape[0]] = torch.eye(
            embeddings.shape[0], device=embeddings.device, dtype=torch.bool
        )
        positive_mask = same_identity & ~self_mask
        if self.positive_mining != "all":
            assert camera_ids is not None and candidate_cameras is not None
            cameras = camera_ids.to(device=embeddings.device)
            cross_camera_positive = positive_mask & ~cameras[:, None].eq(
                candidate_cameras[None, :]
            )
            if self.positive_mining == "cross_camera_preferred":
                has_cross_camera = cross_camera_positive.any(dim=1, keepdim=True)
                positive_mask = torch.where(
                    has_cross_camera, cross_camera_positive, positive_mask
                )
            else:
                positive_mask = cross_camera_positive
        negative_mask = ~same_identity
        valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not torch.any(valid_anchor).item():
            return differentiable_zero

        min_positive = similarities.masked_fill(~positive_mask, torch.inf).min(
            dim=1
        ).values
        max_negative = similarities.masked_fill(~negative_mask, -torch.inf).max(
            dim=1
        ).values
        mined_positive = positive_mask & (
            similarities < max_negative[:, None] + self.epsilon
        )
        mined_negative = negative_mask & (
            similarities > min_positive[:, None] - self.epsilon
        )

        positive_logits = (-self.alpha * (similarities - self.base)).masked_fill(
            ~mined_positive, -torch.inf
        )
        negative_logits = (self.beta * (similarities - self.base)).masked_fill(
            ~mined_negative, -torch.inf
        )
        # A zero logit represents the additive 1 in log(1 + sum(exp(.))).
        zeros = similarities.new_zeros((similarities.shape[0], 1))
        positive_loss = torch.logsumexp(
            torch.cat([zeros, positive_logits], dim=1), dim=1
        ) / self.alpha
        negative_loss = torch.logsumexp(
            torch.cat([zeros, negative_logits], dim=1), dim=1
        ) / self.beta
        return (positive_loss + negative_loss)[valid_anchor].mean()

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, beta={self.beta}, base={self.base}, "
            f"epsilon={self.epsilon}, positive_mining={self.positive_mining!r}"
        )


class CrossBatchMemory(nn.Module):
    """Bounded FIFO queue of detached metric embeddings.

    Queue tensors are intentionally non-persistent buffers: they follow the
    criterion device but are absent from ``state_dict``.  Training resets the
    queue at every epoch boundary, so saving/resuming between epochs is exact
    and checkpoints do not carry stale feature representations.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__()
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self.register_buffer(
            "_embeddings", torch.empty((0, 0), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_labels", torch.empty((0,), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_camera_ids", torch.empty((0,), dtype=torch.long), persistent=False
        )
        self._stores_camera_ids = False

    @property
    def count(self) -> int:
        return int(self._labels.numel())

    def reset(self) -> None:
        """Discard all queued samples while retaining the current device."""

        self._embeddings = self._embeddings.new_empty((0, 0))
        self._labels = self._labels.new_empty((0,))
        self._camera_ids = self._camera_ids.new_empty((0,))
        self._stores_camera_ids = False

    def snapshot(self) -> tuple[Tensor, Tensor, Tensor | None]:
        cameras = self._camera_ids if self._stores_camera_ids else None
        return self._embeddings, self._labels, cameras

    @torch.no_grad()
    def enqueue(
        self,
        embeddings: Tensor,
        labels: Tensor,
        camera_ids: Tensor | None = None,
    ) -> None:
        """Append a batch after detaching it and retain the newest items."""

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D [batch, dim]")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must have one item per embedding")
        if camera_ids is not None and (
            camera_ids.ndim != 1 or camera_ids.shape[0] != embeddings.shape[0]
        ):
            raise ValueError("camera_ids must have one item per embedding")
        if embeddings.shape[0] == 0:
            return
        has_cameras = camera_ids is not None
        if self.count and has_cameras != self._stores_camera_ids:
            raise ValueError(
                "camera_ids must be supplied consistently while memory is non-empty"
            )

        vectors = embeddings.detach().to(dtype=torch.float32)
        batch_labels = labels.detach().to(device=vectors.device, dtype=torch.long)
        batch_cameras = (
            camera_ids.detach().to(device=vectors.device, dtype=torch.long)
            if camera_ids is not None
            else None
        )
        if self.count:
            if self._embeddings.shape[1] != vectors.shape[1]:
                raise ValueError("embedding dimension changed without a memory reset")
            vectors = torch.cat([self._embeddings.to(vectors.device), vectors], dim=0)
            batch_labels = torch.cat(
                [self._labels.to(vectors.device), batch_labels], dim=0
            )
            if batch_cameras is not None:
                batch_cameras = torch.cat(
                    [self._camera_ids.to(vectors.device), batch_cameras], dim=0
                )

        self._embeddings = vectors[-self.capacity :]
        self._labels = batch_labels[-self.capacity :]
        if batch_cameras is not None:
            self._camera_ids = batch_cameras[-self.capacity :]
        else:
            self._camera_ids = self._camera_ids.new_empty((0,))
        self._stores_camera_ids = has_cameras

    def extra_repr(self) -> str:
        return f"capacity={self.capacity}, count={self.count}"


class ReIDLoss(nn.Module):
    """Weighted ArcFace cross-entropy plus batch-hard triplet loss.

    The ArcFace class weights are trainable parameters of this module.  Training
    code must therefore add ``criterion.parameters()`` to the optimizer.
    ``embeddings`` may be either a tensor or the dictionary returned by
    ``VehicleReIDModel(..., return_dict=True)``.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.50,
        easy_margin: bool = False,
        arcface_subcenters: int = 1,
        triplet_margin: float = 0.30,
        triplet_metric: str = "cosine",
        normalize_triplet_embeddings: bool = True,
        triplet_margin_mode: str = "fixed",
        triplet_positive_mining: str = "all",
        metric_loss: str = "triplet",
        multi_similarity_alpha: float = 2.0,
        multi_similarity_beta: float = 50.0,
        multi_similarity_base: float = 0.5,
        multi_similarity_epsilon: float = 0.1,
        label_smoothing: float = 0.10,
        classification_weight: float = 1.0,
        triplet_weight: float = 1.0,
        cross_batch_memory_capacity: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if classification_weight < 0.0 or triplet_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if classification_weight == 0.0 and triplet_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")
        if (
            isinstance(cross_batch_memory_capacity, bool)
            or not isinstance(cross_batch_memory_capacity, int)
            or cross_batch_memory_capacity < 0
        ):
            raise ValueError(
                "cross_batch_memory_capacity must be a non-negative integer"
            )
        if cross_batch_memory_capacity > 0 and triplet_weight == 0.0:
            raise ValueError("cross-batch memory requires a positive triplet_weight")

        self.arcface = ArcMarginProduct(
            embedding_dim,
            num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
            easy_margin=easy_margin,
            num_subcenters=arcface_subcenters,
        )
        metric_loss = str(metric_loss).lower()
        if metric_loss == "triplet":
            self.metric_objective: nn.Module = BatchHardTripletLoss(
                triplet_margin,
                metric=triplet_metric,
                normalize_embeddings=normalize_triplet_embeddings,
                margin_mode=triplet_margin_mode,
                positive_mining=triplet_positive_mining,
            )
        elif metric_loss == "multi_similarity":
            self.metric_objective = MultiSimilarityLoss(
                alpha=multi_similarity_alpha,
                beta=multi_similarity_beta,
                base=multi_similarity_base,
                epsilon=multi_similarity_epsilon,
                positive_mining=triplet_positive_mining,
            )
        else:
            raise ValueError("metric_loss must be 'triplet' or 'multi_similarity'")
        self.metric_loss_type = metric_loss
        self.label_smoothing = float(label_smoothing)
        self.classification_weight = float(classification_weight)
        self.triplet_weight = float(triplet_weight)
        self.cross_batch_memory = (
            CrossBatchMemory(cross_batch_memory_capacity)
            if cross_batch_memory_capacity > 0
            else None
        )

    @property
    def cross_batch_memory_count(self) -> int:
        return (
            self.cross_batch_memory.count
            if self.cross_batch_memory is not None
            else 0
        )

    def reset_cross_batch_memory(self) -> None:
        """Reset the optional queue; called once at every epoch boundary."""

        if self.cross_batch_memory is not None:
            self.cross_batch_memory.reset()

    @staticmethod
    def _unpack_embeddings(
        output: Tensor | Mapping[str, Tensor],
        metric_embeddings: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(output, Tensor):
            return output, output if metric_embeddings is None else metric_embeddings
        if "embeddings" not in output:
            raise KeyError("model output dictionary must contain 'embeddings'")
        embeddings = output["embeddings"]
        if metric_embeddings is None:
            metric_embeddings = output.get("features", embeddings)
        return embeddings, metric_embeddings

    def compute_components(
        self,
        embeddings: Tensor | Mapping[str, Tensor],
        labels: Tensor,
        *,
        metric_embeddings: Tensor | None = None,
        camera_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        classification_embeddings, triplet_embeddings = self._unpack_embeddings(
            embeddings, metric_embeddings
        )
        logits = self.arcface(classification_embeddings, labels)
        labels_for_ce = labels.to(device=logits.device, dtype=torch.long)
        classification_loss = F.cross_entropy(
            logits,
            labels_for_ce,
            label_smoothing=self.label_smoothing,
        )
        memory_embeddings = memory_labels = memory_camera_ids = None
        # Evaluation must be stateless: it neither consumes nor updates stale
        # training features.  The normal training loop sets criterion.train().
        if (
            self.training
            and self.cross_batch_memory is not None
            and self.cross_batch_memory.count > 0
        ):
            memory_embeddings, memory_labels, memory_camera_ids = (
                self.cross_batch_memory.snapshot()
            )
        metric_loss = self.metric_objective(
            triplet_embeddings,
            labels,
            camera_ids,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
            memory_camera_ids=memory_camera_ids,
        )
        if self.training and self.cross_batch_memory is not None:
            self.cross_batch_memory.enqueue(
                triplet_embeddings, labels, camera_ids
            )
        total = (
            self.classification_weight * classification_loss
            + self.triplet_weight * metric_loss
        )
        return {
            "loss": total,
            "classification_loss": classification_loss,
            "arcface_loss": classification_loss,
            "metric_loss": metric_loss,
            # Backward-compatible alias for existing logs and integrations.
            "triplet_loss": metric_loss,
            "logits": logits,
        }

    def forward(
        self,
        embeddings: Tensor | Mapping[str, Tensor],
        labels: Tensor,
        *,
        metric_embeddings: Tensor | None = None,
        camera_ids: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        components = self.compute_components(
            embeddings,
            labels,
            metric_embeddings=metric_embeddings,
            camera_ids=camera_ids,
        )
        return components if return_details else components["loss"]


__all__ = [
    "ArcFace",
    "ArcMarginProduct",
    "BatchHardTripletLoss",
    "CrossBatchMemory",
    "MultiSimilarityLoss",
    "ReIDLoss",
    "TripletLoss",
]
