from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.config import (
    normalize_hard_identity_mining_config,
    validate_config,
)
from src.sampler import (
    CameraAwarePKBatchSampler,
    HardIdentityMiningPKBatchSampler,
    build_identity_neighbor_map,
)
from src.train import _refresh_hard_identity_map, _training_state


def _minimal_config() -> dict:
    return {
        "seed": 7,
        "paths": {},
        "model": {"backbone": "resnet18", "embedding_dim": 4, "input_size": 64},
        "data": {},
        "training": {"identities_per_batch": 4, "instances_per_identity": 2},
        "inference": {},
    }


def _identity_fixture() -> tuple[list[int], list[int]]:
    pids = [pid for pid in range(6) for _ in range(2)]
    cameras = [camera for _ in range(6) for camera in (0, 1)]
    return pids, cameras


def test_identity_neighbor_map_uses_normalized_centroids_deterministically() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.9, 0.1],
            [1.0, 0.1],
            [-1.0, 0.0],
            [-0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    pids = ["a", "a", "b", "b", "c", "c", "d", "d"]

    first, first_report = build_identity_neighbor_map(
        embeddings, pids, neighbors_per_identity=2, chunk_size=2
    )
    second, second_report = build_identity_neighbor_map(
        embeddings, pids, neighbors_per_identity=2, chunk_size=4
    )

    assert first == second
    assert first["a"][0] == "b"
    assert first["b"][0] == "a"
    assert all(identity not in neighbors for identity, neighbors in first.items())
    assert first_report["fingerprint"] == second_report["fingerprint"]
    assert first_report["identity_count"] == 4
    assert first_report["image_count"] == 8
    assert first_report["neighbors_per_identity"] == 2


@pytest.mark.parametrize(
    ("embeddings", "pids", "message"),
    [
        (np.ones((2, 2), dtype=np.float32), [1], "equal length"),
        (np.ones((2,), dtype=np.float32), [1, 2], "two-dimensional"),
        (np.asarray([[np.nan, 0.0], [1.0, 0.0]]), [1, 2], "finite"),
        (np.zeros((2, 2), dtype=np.float32), [1, 2], "non-zero norm"),
    ],
)
def test_identity_neighbor_map_rejects_invalid_inputs(embeddings, pids, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_identity_neighbor_map(embeddings, pids)


def test_hard_sampler_is_random_pk_until_map_is_installed() -> None:
    pids, cameras = _identity_fixture()
    baseline = CameraAwarePKBatchSampler(
        pids,
        cameras,
        identities_per_batch=4,
        instances_per_identity=2,
        batches_per_epoch=4,
        seed=19,
    )
    hard = HardIdentityMiningPKBatchSampler(
        pids,
        cameras,
        identities_per_batch=4,
        instances_per_identity=2,
        batches_per_epoch=4,
        seed=19,
        hard_fraction=0.5,
    )

    assert list(hard) == list(baseline)
    assert hard.sampling_report()["active"] is False


def test_hard_sampler_mixes_neighbor_group_with_random_ids_and_is_reproducible() -> None:
    pids, cameras = _identity_fixture()
    sampler = HardIdentityMiningPKBatchSampler(
        pids,
        cameras,
        identities_per_batch=4,
        instances_per_identity=2,
        batches_per_epoch=8,
        seed=23,
        hard_fraction=0.5,
    )
    paired_neighbors = {
        0: [1],
        1: [0],
        2: [3],
        3: [2],
        4: [5],
        5: [4],
    }
    sampler.set_hard_neighbors(paired_neighbors)

    first = list(sampler)
    assert first == list(sampler)
    for batch in first:
        selected_ids = {pids[index] for index in batch}
        assert len(batch) == 8
        assert set(Counter(pids[index] for index in batch).values()) == {2}
        assert any(
            neighbor in selected_ids
            for identity in selected_ids
            for neighbor in paired_neighbors[identity]
        )
        for identity in selected_ids:
            indices = [index for index in batch if pids[index] == identity]
            assert {cameras[index] for index in indices} == {0, 1}

    report = sampler.sampling_report()
    assert report["active"] is True
    assert report["batches"] == 8
    assert report["target_hard_group_size"] == 2
    assert report["achieved_hard_group_fraction"] == pytest.approx(0.5)
    assert isinstance(report["neighbor_fingerprint"], str)


def test_hard_sampler_state_restores_neighbor_map_and_sequence() -> None:
    pids, cameras = _identity_fixture()
    arguments = {
        "identities_per_batch": 4,
        "instances_per_identity": 2,
        "batches_per_epoch": 3,
        "seed": 29,
        "hard_fraction": 0.5,
    }
    first = HardIdentityMiningPKBatchSampler(pids, cameras, **arguments)
    first.set_hard_neighbors(
        {identity: [(identity + 1) % 6, (identity + 2) % 6] for identity in range(6)}
    )
    first.set_epoch(4)

    restored = HardIdentityMiningPKBatchSampler(pids, cameras, **arguments)
    restored.load_state_dict(first.state_dict())

    assert restored.neighbor_fingerprint == first.neighbor_fingerprint
    assert list(restored) == list(first)


def test_hard_identity_config_defaults_validation_and_xbm_exclusion() -> None:
    assert normalize_hard_identity_mining_config(None) == {
        "enabled": False,
        "hard_fraction": 0.5,
        "neighbors_per_identity": 32,
        "warmup_epochs": 1,
        "refresh_interval": 1,
        "embedding_batch_size": 64,
        "tta_horizontal_flip": False,
    }
    config = _minimal_config()
    config["training"]["hard_identity_mining"] = {
        "enabled": True,
        "hard_fraction": 0.6,
        "neighbors_per_identity": 12,
        "warmup_epochs": 2,
        "refresh_interval": 3,
        "embedding_batch_size": 16,
        "tta_horizontal_flip": True,
    }
    validate_config(config)

    config["training"]["cross_batch_memory"] = {"enabled": True, "capacity": 16}
    with pytest.raises(ValueError, match="cannot both be enabled"):
        validate_config(config)


@pytest.mark.parametrize(
    "section",
    [
        True,
        {"enabled": 1},
        {"hard_fraction": 0.0},
        {"hard_fraction": 1.0},
        {"neighbors_per_identity": 0},
        {"warmup_epochs": -1},
        {"refresh_interval": 0},
        {"embedding_batch_size": True},
        {"tta_horizontal_flip": 1},
        {"neigbors_per_identity": 12},
    ],
)
def test_invalid_hard_identity_config_is_rejected(section) -> None:
    config = _minimal_config()
    config["training"]["hard_identity_mining"] = section
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


class _IdentityModel(nn.Module):
    def forward(self, images):
        return F.normalize(images.float(), dim=1)


def test_refresh_installs_map_and_training_state_preserves_it() -> None:
    pids = [0, 0, 1, 1, 2, 2]
    cameras = [0, 1, 0, 1, 0, 1]
    sampler = HardIdentityMiningPKBatchSampler(
        pids,
        cameras,
        identities_per_batch=3,
        instances_per_identity=2,
        batches_per_epoch=1,
        hard_fraction=0.5,
    )
    batch = {
        "image": torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [-1.0, 0.0],
                [-0.9, 0.1],
            ]
        )
    }
    mining_config = normalize_hard_identity_mining_config(
        {"enabled": True, "neighbors_per_identity": 2}
    )
    report = _refresh_hard_identity_map(
        _IdentityModel(),
        [batch],
        sampler,
        pids,
        mining_config,
        epoch=1,
        device=torch.device("cpu"),
        amp=False,
        channels_last=False,
    )

    assert sampler.has_hard_neighbors
    assert report["refresh_epoch"] == 2
    assert report["fingerprint"] == sampler.neighbor_fingerprint

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = _training_state(
        epoch=1,
        model=model,
        criterion=nn.Identity(),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_map=0.2,
        history=[],
        sampler=sampler,
        hard_identity_mining_config=mining_config,
        hard_identity_mining_report=report,
    )
    assert state["sampler_state"]["hard_neighbor_indices"]
    assert state["hard_identity_mining_config"] == mining_config
    assert state["hard_identity_mining_report"]["fingerprint"] == report["fingerprint"]
