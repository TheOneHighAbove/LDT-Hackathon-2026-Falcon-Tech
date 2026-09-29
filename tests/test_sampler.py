from __future__ import annotations

from collections import Counter

import pytest

from src.sampler import CameraAwarePKBatchSampler


def test_pk_batches_are_balanced_camera_aware_and_deterministic() -> None:
    pids = [pid for pid in range(4) for _ in range(4)]
    cameras = [camera for _ in range(4) for camera in (0, 0, 1, 1)]
    sampler = CameraAwarePKBatchSampler(
        pids,
        cameras,
        identities_per_batch=3,
        instances_per_identity=3,
        batches_per_epoch=5,
        seed=17,
    )
    first = list(iter(sampler))
    assert first == list(iter(sampler))
    assert len(first) == len(sampler) == 5
    for batch in first:
        assert len(batch) == 9
        assert sorted(Counter(pids[index] for index in batch).values()) == [3, 3, 3]
        for pid in set(pids[index] for index in batch):
            selected = [index for index in batch if pids[index] == pid]
            assert len({cameras[index] for index in selected}) == 2

    sampler.set_epoch(1)
    second = list(iter(sampler))
    assert second == list(iter(sampler))
    assert first != second


def test_pk_sampler_uses_replacement_only_when_needed() -> None:
    sampler = CameraAwarePKBatchSampler(
        [10, 20],
        [1, 2],
        identities_per_batch=2,
        instances_per_identity=3,
        batches_per_epoch=1,
    )
    batch = next(iter(sampler))
    assert Counter(batch) == {0: 3, 1: 3}


def test_pk_sampler_validates_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        CameraAwarePKBatchSampler([1, 2], [0])
    with pytest.raises(ValueError, match="at least"):
        CameraAwarePKBatchSampler(
            [1, 1, 2, 2], identities_per_batch=3, instances_per_identity=2
        )
