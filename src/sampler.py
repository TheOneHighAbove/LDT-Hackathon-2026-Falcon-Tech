"""Deterministic camera-aware P x K and offline hard-ID sampling."""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from torch.utils.data import Sampler


def _to_list(values: Sequence[Any] | Iterable[Any], name: str) -> list[Any]:
    if hasattr(values, "tolist"):
        result = values.tolist()
        if not isinstance(result, list):
            result = list(result)
    else:
        result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if not isinstance(value, Hashable):
            raise TypeError(f"all {name} values must be hashable")
    return result


def _identity_token(value: Any) -> bytes:
    """Return an unambiguous, process-independent token for an identity."""

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return f"{type_name}:{value!r}".encode("utf-8")


def _neighbor_index_fingerprint(
    identities: Sequence[Any], rows: Sequence[Sequence[int]]
) -> str:
    digest = hashlib.sha256(b"vehicle-reid-hard-identity-map-v1\0")
    for identity, neighbors in zip(identities, rows, strict=True):
        token = _identity_token(identity)
        digest.update(len(token).to_bytes(8, "little"))
        digest.update(token)
        encoded = np.asarray(list(neighbors), dtype="<i8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded.tobytes())
    return digest.hexdigest()


def build_identity_neighbor_map(
    embeddings: np.ndarray,
    vehicle_ids: Sequence[Any] | Iterable[Any],
    *,
    neighbors_per_identity: int = 32,
    chunk_size: int = 1024,
) -> tuple[dict[Any, list[Any]], dict[str, int | float | str]]:
    """Build a deterministic cosine-nearest-ID map from image embeddings.

    Image embeddings are averaged per identity and the resulting centroids are
    L2-normalized.  The returned map never includes the identity itself.  A
    stable full sort makes exact cosine ties deterministic in first-observed ID
    order; chunking bounds the temporary similarity matrix.
    """

    ids = _to_list(vehicle_ids, "vehicle_ids")
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if matrix.shape[0] != len(ids):
        raise ValueError("embeddings and vehicle_ids must have equal length")
    if matrix.shape[1] <= 0:
        raise ValueError("embeddings must have a positive feature dimension")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite numeric values")
    if (
        isinstance(neighbors_per_identity, bool)
        or not isinstance(neighbors_per_identity, (int, np.integer))
        or int(neighbors_per_identity) <= 0
    ):
        raise ValueError("neighbors_per_identity must be a positive integer")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or int(chunk_size) <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    identities = list(dict.fromkeys(ids))
    if len(identities) < 2:
        raise ValueError("hard-identity mining requires at least two identities")
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    row_identity_indices = np.fromiter(
        (identity_to_index[identity] for identity in ids),
        dtype=np.int64,
        count=len(ids),
    )
    # Float64 accumulation avoids order-sensitive centroid drift while the
    # normalized result returns to compact float32 for cosine search.
    centroid_sums = np.zeros(
        (len(identities), matrix.shape[1]), dtype=np.float64
    )
    np.add.at(centroid_sums, row_identity_indices, matrix.astype(np.float64, copy=False))
    counts = np.bincount(row_identity_indices, minlength=len(identities))
    centroids = centroid_sums / counts[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms[:, 0] <= 1e-12)[:5]
        raise ValueError(
            "identity centroids must have non-zero norm; invalid identity positions "
            f"include {bad.tolist()}"
        )
    centroids = (centroids / norms).astype(np.float32, copy=False)

    neighbor_count = min(int(neighbors_per_identity), len(identities) - 1)
    neighbor_rows: list[list[int]] = []
    top1_similarities: list[float] = []
    for start in range(0, len(identities), int(chunk_size)):
        stop = min(start + int(chunk_size), len(identities))
        similarities = centroids[start:stop] @ centroids.T
        local_rows = np.arange(stop - start)
        similarities[local_rows, np.arange(start, stop)] = -np.inf
        # mergesort is stable, hence exact ties follow first-observed ID order.
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbor_count]
        for local_index, neighbors in enumerate(order):
            row = [int(index) for index in neighbors]
            neighbor_rows.append(row)
            top1_similarities.append(
                float(similarities[local_index, row[0]])
            )

    mapping = {
        identity: [identities[index] for index in neighbor_rows[position]]
        for position, identity in enumerate(identities)
    }
    top1 = np.asarray(top1_similarities, dtype=np.float64)
    report: dict[str, int | float | str] = {
        "identity_count": len(identities),
        "image_count": len(ids),
        "embedding_dim": int(matrix.shape[1]),
        "neighbors_per_identity": neighbor_count,
        "mean_top1_similarity": float(top1.mean()),
        "median_top1_similarity": float(np.median(top1)),
        "min_top1_similarity": float(top1.min()),
        "max_top1_similarity": float(top1.max()),
        "fingerprint": _neighbor_index_fingerprint(identities, neighbor_rows),
    }
    return mapping, report


class CameraAwarePKBatchSampler(Sampler[list[int]]):
    """Yield batches containing exactly ``P`` identities and ``K`` samples/ID.

    Samples from distinct cameras are used first for every identity.  Once all
    available samples have been consumed, replacement is used only as needed.
    Iteration uses a local random generator, so it does not perturb NumPy's
    process-wide RNG.  Calling :meth:`set_epoch` changes the sequence in the
    same manner as PyTorch's distributed samplers.
    """

    def __init__(
        self,
        vehicle_ids: Sequence[Any] | Iterable[Any],
        camera_ids: Sequence[Any] | Iterable[Any] | None = None,
        *,
        identities_per_batch: int = 16,
        instances_per_identity: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.vehicle_ids = _to_list(vehicle_ids, "vehicle_ids")
        if camera_ids is None:
            self.camera_ids = [0] * len(self.vehicle_ids)
        else:
            self.camera_ids = _to_list(camera_ids, "camera_ids")
            if len(self.camera_ids) != len(self.vehicle_ids):
                raise ValueError("vehicle_ids and camera_ids must have equal length")

        if not isinstance(identities_per_batch, (int, np.integer)) or isinstance(
            identities_per_batch, bool
        ) or identities_per_batch <= 0:
            raise ValueError("identities_per_batch must be a positive integer")
        if not isinstance(instances_per_identity, (int, np.integer)) or isinstance(
            instances_per_identity, bool
        ) or instances_per_identity <= 0:
            raise ValueError("instances_per_identity must be a positive integer")
        self.P = int(identities_per_batch)
        self.K = int(instances_per_identity)
        self.seed = int(seed)
        self.epoch = 0

        pid_to_camera_indices: dict[Any, dict[Any, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, (pid, camera_id) in enumerate(
            zip(self.vehicle_ids, self.camera_ids, strict=True)
        ):
            pid_to_camera_indices[pid][camera_id].append(index)
        self._pid_to_camera_indices = {
            pid: {camera: indices[:] for camera, indices in camera_map.items()}
            for pid, camera_map in pid_to_camera_indices.items()
        }
        # dict preserves first-observed order and supports mixed-type IDs.
        self.identities = list(self._pid_to_camera_indices)
        if len(self.identities) < self.P:
            raise ValueError(
                f"P={self.P} requires at least {self.P} identities, "
                f"but only {len(self.identities)} are available"
            )

        if batches_per_epoch is None:
            batches_per_epoch = max(
                1, math.ceil(len(self.vehicle_ids) / (self.P * self.K))
            )
        if not isinstance(batches_per_epoch, (int, np.integer)) or isinstance(
            batches_per_epoch, bool
        ) or int(batches_per_epoch) <= 0:
            raise ValueError("batches_per_epoch must be a positive integer")
        self.batches_per_epoch = int(batches_per_epoch)

    @property
    def batch_size(self) -> int:
        return self.P * self.K

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed subsequent deterministic iterations."""

        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _sample_identity(
        self, pid: Any, rng: np.random.Generator
    ) -> list[int]:
        camera_map = self._pid_to_camera_indices[pid]
        cameras = list(camera_map)
        rng.shuffle(cameras)
        queues: dict[Any, list[int]] = {}
        for camera in cameras:
            queue = camera_map[camera][:]
            rng.shuffle(queue)
            queues[camera] = queue

        chosen: list[int] = []
        # Round-robin across cameras maximizes camera diversity in the prefix.
        while len(chosen) < self.K and any(queues[camera] for camera in cameras):
            for camera in cameras:
                if queues[camera] and len(chosen) < self.K:
                    chosen.append(queues[camera].pop())

        if len(chosen) < self.K:
            all_indices = [
                index
                for indices in camera_map.values()
                for index in indices
            ]
            extra = rng.choice(
                np.asarray(all_indices, dtype=np.int64),
                size=self.K - len(chosen),
                replace=True,
            )
            chosen.extend(int(index) for index in np.atleast_1d(extra))
        return chosen

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        identity_indices = np.arange(len(self.identities), dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            selected = rng.choice(identity_indices, size=self.P, replace=False)
            batch: list[int] = []
            for pid_index in selected:
                pid = self.identities[int(pid_index)]
                batch.extend(self._sample_identity(pid, rng))
            yield batch

    def state_dict(self) -> dict[str, Any]:
        """Return the minimal state needed to reproduce the next iteration."""

        return {"seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "epoch" not in state:
            raise ValueError("sampler state is missing 'epoch'")
        self.set_epoch(int(state["epoch"]))
        if "seed" in state and int(state["seed"]) != self.seed:
            raise ValueError("sampler state seed does not match this sampler")


class HardIdentityMiningPKBatchSampler(CameraAwarePKBatchSampler):
    """PK sampler mixing an offline hard-ID group with random identities.

    A batch starts from a uniformly sampled anchor identity.  A configured
    fraction of its identity slots is filled from the anchor's offline nearest
    neighbors; the remaining slots are sampled uniformly from all other IDs.
    Images are still drawn by the base camera-aware sampler, and losses are
    therefore computed exclusively from fresh examples in the current batch.

    Until :meth:`set_hard_neighbors` is called, iteration is byte-for-byte the
    base random PK policy.  This permits an initial random warm-up epoch.
    """

    def __init__(
        self,
        vehicle_ids: Sequence[Any] | Iterable[Any],
        camera_ids: Sequence[Any] | Iterable[Any] | None = None,
        *,
        identities_per_batch: int = 16,
        instances_per_identity: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
        hard_fraction: float = 0.5,
    ) -> None:
        super().__init__(
            vehicle_ids,
            camera_ids,
            identities_per_batch=identities_per_batch,
            instances_per_identity=instances_per_identity,
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
        if self.P < 3:
            raise ValueError(
                "hard-identity sampling requires at least three identities per batch"
            )
        if isinstance(hard_fraction, bool):
            raise TypeError("hard_fraction must be numeric")
        try:
            fraction = float(hard_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("hard_fraction must be finite and in (0, 1)") from exc
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("hard_fraction must be finite and in (0, 1)")
        self.hard_fraction = fraction
        self.hard_group_size = max(2, min(self.P - 1, round(self.P * fraction)))
        self._identity_to_index = {
            identity: index for index, identity in enumerate(self.identities)
        }
        self._hard_neighbor_indices: dict[int, tuple[int, ...]] = {}
        self._last_sampling_report: dict[str, int | float | bool | str | None] = {
            "active": False,
            "batches": 0,
            "target_hard_group_size": self.hard_group_size,
            "achieved_hard_group_fraction": 0.0,
            "neighbor_fingerprint": None,
        }

    @property
    def has_hard_neighbors(self) -> bool:
        return bool(self._hard_neighbor_indices)

    @property
    def neighbor_fingerprint(self) -> str | None:
        if not self._hard_neighbor_indices:
            return None
        rows = [self._hard_neighbor_indices.get(index, ()) for index in range(len(self.identities))]
        return _neighbor_index_fingerprint(self.identities, rows)

    def set_hard_neighbors(
        self, neighbors: Mapping[Any, Sequence[Any] | Iterable[Any]]
    ) -> None:
        """Install a validated nearest-neighbor map using source identity IDs."""

        if not isinstance(neighbors, Mapping):
            raise TypeError("hard identity neighbors must be a mapping")
        unknown_anchors = [identity for identity in neighbors if identity not in self._identity_to_index]
        if unknown_anchors:
            raise ValueError(
                "hard identity map contains unknown anchor IDs: "
                f"{unknown_anchors[:5]!r}"
            )
        encoded: dict[int, tuple[int, ...]] = {}
        for identity in self.identities:
            raw_neighbors = list(neighbors.get(identity, ()))
            seen: set[int] = set()
            indices: list[int] = []
            own_index = self._identity_to_index[identity]
            for neighbor in raw_neighbors:
                if neighbor not in self._identity_to_index:
                    raise ValueError(
                        f"hard identity map for {identity!r} contains unknown ID "
                        f"{neighbor!r}"
                    )
                index = self._identity_to_index[neighbor]
                if index != own_index and index not in seen:
                    seen.add(index)
                    indices.append(index)
            if indices:
                encoded[own_index] = tuple(indices)
        if not encoded:
            raise ValueError("hard identity map has no usable cross-identity neighbors")
        self._hard_neighbor_indices = encoded

    def clear_hard_neighbors(self) -> None:
        """Restore random PK sampling, primarily for controlled ablations."""

        self._hard_neighbor_indices = {}

    def __iter__(self):
        if not self.has_hard_neighbors:
            self._last_sampling_report = {
                "active": False,
                "batches": 0,
                "target_hard_group_size": self.hard_group_size,
                "achieved_hard_group_fraction": 0.0,
                "neighbor_fingerprint": None,
            }
            yield from super().__iter__()
            return

        rng = np.random.default_rng(self.seed + self.epoch)
        all_indices = np.arange(len(self.identities), dtype=np.int64)
        selected_hard_neighbors = 0
        completed_batches = 0
        for _ in range(self.batches_per_epoch):
            anchor = int(rng.integers(len(self.identities)))
            candidates = np.asarray(
                self._hard_neighbor_indices.get(anchor, ()), dtype=np.int64
            )
            requested_neighbors = self.hard_group_size - 1
            if len(candidates) > requested_neighbors:
                hard_neighbors = [
                    int(index)
                    for index in rng.choice(
                        candidates, size=requested_neighbors, replace=False
                    )
                ]
            else:
                hard_neighbors = [int(index) for index in candidates]

            chosen = [anchor, *hard_neighbors]
            chosen_set = set(chosen)
            remaining = np.asarray(
                [int(index) for index in all_indices if int(index) not in chosen_set],
                dtype=np.int64,
            )
            missing = self.P - len(chosen)
            random_identities = [
                int(index)
                for index in rng.choice(remaining, size=missing, replace=False)
            ]
            selected = np.asarray([*chosen, *random_identities], dtype=np.int64)
            rng.shuffle(selected)

            batch: list[int] = []
            for identity_index in selected:
                identity = self.identities[int(identity_index)]
                batch.extend(self._sample_identity(identity, rng))
            selected_hard_neighbors += len(hard_neighbors)
            completed_batches += 1
            self._last_sampling_report = {
                "active": True,
                "batches": completed_batches,
                "target_hard_group_size": self.hard_group_size,
                "achieved_hard_group_fraction": (
                    completed_batches + selected_hard_neighbors
                )
                / (completed_batches * self.P),
                "neighbor_fingerprint": self.neighbor_fingerprint,
            }
            yield batch

    def sampling_report(self) -> dict[str, int | float | bool | str | None]:
        return dict(self._last_sampling_report)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "hard_fraction": self.hard_fraction,
                "hard_neighbor_indices": [
                    list(self._hard_neighbor_indices.get(index, ()))
                    for index in range(len(self.identities))
                ],
            }
        )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        if "hard_fraction" in state and not math.isclose(
            float(state["hard_fraction"]), self.hard_fraction, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("sampler state hard_fraction does not match this sampler")
        raw_rows = state.get("hard_neighbor_indices", [])
        if not isinstance(raw_rows, (list, tuple)):
            raise ValueError("sampler hard_neighbor_indices state must be a sequence")
        if raw_rows and len(raw_rows) != len(self.identities):
            raise ValueError(
                "sampler hard-neighbor state identity count does not match this sampler"
            )
        restored: dict[int, tuple[int, ...]] = {}
        for anchor, raw_neighbors in enumerate(raw_rows):
            if not isinstance(raw_neighbors, (list, tuple)):
                raise ValueError("each sampler hard-neighbor row must be a sequence")
            values: list[int] = []
            seen: set[int] = set()
            for raw_index in raw_neighbors:
                if isinstance(raw_index, bool) or not isinstance(
                    raw_index, (int, np.integer)
                ):
                    raise ValueError("sampler hard-neighbor indices must be integers")
                index = int(raw_index)
                if not 0 <= index < len(self.identities) or index == anchor:
                    raise ValueError("sampler hard-neighbor index is out of range or self")
                if index not in seen:
                    seen.add(index)
                    values.append(index)
            if values:
                restored[anchor] = tuple(values)
        self._hard_neighbor_indices = restored


# Conventional shorter name for configuration files and training scripts.
PKBatchSampler = CameraAwarePKBatchSampler


__all__ = [
    "build_identity_neighbor_map",
    "CameraAwarePKBatchSampler",
    "HardIdentityMiningPKBatchSampler",
    "PKBatchSampler",
]
