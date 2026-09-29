from __future__ import annotations

import json

import pandas as pd

from src.split import identity_disjoint_split, save_split


def _annotations() -> pd.DataFrame:
    rows = []
    counts = {"singleton": 1, "a": 2, "b": 2, "c": 3, "d": 2}
    image_index = 0
    for pid, count in counts.items():
        for occurrence in range(count):
            rows.append(
                {
                    "image_id": f"image-{image_index}",
                    "vehicle_id": pid,
                    "camera_id": occurrence % 2,
                }
            )
            image_index += 1
    return pd.DataFrame(rows)


def test_identity_split_is_disjoint_deterministic_and_keeps_valid_ids() -> None:
    frame = _annotations()
    train, val = identity_disjoint_split(frame, val_fraction=0.4, seed=42)
    train_again, val_again = identity_disjoint_split(
        frame, val_fraction=0.4, seed=42
    )
    pd.testing.assert_frame_equal(train, train_again)
    pd.testing.assert_frame_equal(val, val_again)
    assert set(train.vehicle_id).isdisjoint(val.vehicle_id)
    assert "singleton" in set(train.vehicle_id)
    assert val.vehicle_id.nunique() == 2
    assert (val.groupby("vehicle_id").size() >= 2).all()
    assert set(train.image_id).union(val.image_id) == set(frame.image_id)


def test_split_can_require_multiple_validation_cameras() -> None:
    frame = _annotations()
    train, val = identity_disjoint_split(
        frame,
        val_fraction=0.4,
        seed=4,
        min_val_cameras=2,
    )
    assert (val.groupby("vehicle_id").camera_id.nunique() >= 2).all()
    assert set(train.vehicle_id).isdisjoint(val.vehicle_id)


def test_save_split_writes_csv_id_lists_and_reproducibility_metadata(tmp_path) -> None:
    train, val = identity_disjoint_split(_annotations(), val_fraction=0.4, seed=42)
    paths = save_split(train, val, tmp_path / "split", seed=42, val_fraction=0.4)
    assert all(path.is_file() for path in paths.values())
    pd.testing.assert_frame_equal(pd.read_csv(paths["train_csv"]), train)
    pd.testing.assert_frame_equal(pd.read_csv(paths["val_csv"]), val)
    assert paths["train_image_ids"].read_text(encoding="utf-8").splitlines() == list(
        train.image_id
    )
    assert paths["val_image_ids"].read_text(encoding="utf-8").splitlines() == list(
        val.image_id
    )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["train_images"] == len(train)
    assert metadata["val_images"] == len(val)
    assert len(metadata["combined_csv_sha256"]) == 64
