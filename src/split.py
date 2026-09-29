"""Reproducible identity-disjoint train/validation splitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SEED = 42


def _load_annotations(
    annotations: str | Path | pd.DataFrame, vehicle_id_column: str
) -> pd.DataFrame:
    if isinstance(annotations, (str, Path)):
        frame = pd.read_csv(annotations)
    elif isinstance(annotations, pd.DataFrame):
        frame = annotations.copy(deep=True)
    else:
        raise TypeError("annotations must be a CSV path or pandas DataFrame")
    missing = {"image_id", vehicle_id_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"annotation columns are missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("annotations are empty")
    for column in ("image_id", vehicle_id_column):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values")
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError("image_id must be unique before creating a split")
    return frame.reset_index(drop=True)


def identity_disjoint_split(
    annotations: str | Path | pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
    min_val_images: int = 2,
    min_val_cameras: int | None = None,
    vehicle_id_column: str = "vehicle_id",
    camera_id_column: str = "camera_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole identities into train and validation partitions.

    IDs with fewer than ``min_val_images`` observations remain in training.
    ``min_val_cameras=2`` can additionally guarantee meaningful cross-camera
    validation when camera labels exist.  Rows retain their original order.
    """

    frame = _load_annotations(annotations, vehicle_id_column)
    if vehicle_id_column not in frame.columns:
        raise ValueError(f"missing vehicle ID column: {vehicle_id_column!r}")
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if not isinstance(min_val_images, (int, np.integer)) or isinstance(
        min_val_images, bool
    ) or int(min_val_images) < 2:
        raise ValueError("min_val_images must be an integer >= 2")
    min_val_images = int(min_val_images)
    if min_val_cameras is not None:
        if not isinstance(min_val_cameras, (int, np.integer)) or isinstance(
            min_val_cameras, bool
        ) or int(min_val_cameras) < 1:
            raise ValueError("min_val_cameras must be a positive integer or None")
        min_val_cameras = int(min_val_cameras)
        if camera_id_column not in frame.columns:
            raise ValueError(
                f"min_val_cameras was requested but {camera_id_column!r} is missing"
            )

    identities = list(pd.unique(frame[vehicle_id_column]))
    if len(identities) < 2:
        raise ValueError("at least two vehicle identities are required for a split")

    counts = frame.groupby(vehicle_id_column, sort=False).size().to_dict()
    eligible = [pid for pid in identities if counts[pid] >= min_val_images]
    if min_val_cameras is not None:
        camera_counts = (
            frame.groupby(vehicle_id_column, sort=False)[camera_id_column]
            .nunique(dropna=True)
            .to_dict()
        )
        eligible = [
            pid for pid in eligible if camera_counts.get(pid, 0) >= min_val_cameras
        ]
    if not eligible:
        raise ValueError(
            "no identity is eligible for validation under the requested constraints"
        )

    # Round to the closest identity count, while keeping both partitions nonempty.
    target_val_identities = max(1, int(round(len(identities) * val_fraction)))
    target_val_identities = min(
        target_val_identities, len(eligible), len(identities) - 1
    )
    rng = np.random.default_rng(int(seed))
    selected_positions = rng.choice(
        len(eligible), size=target_val_identities, replace=False
    )
    val_identities = {eligible[int(position)] for position in selected_positions}

    val_mask = frame[vehicle_id_column].isin(val_identities)
    train = frame.loc[~val_mask].copy().reset_index(drop=True)
    val = frame.loc[val_mask].copy().reset_index(drop=True)
    if train.empty or val.empty:
        raise RuntimeError("split unexpectedly produced an empty partition")
    if set(train[vehicle_id_column]).intersection(val[vehicle_id_column]):
        raise RuntimeError("identity leakage detected between train and validation")
    if (val.groupby(vehicle_id_column).size() < min_val_images).any():
        raise RuntimeError("validation contains an identity with too few images")
    return train, val


def _ordered_unique_text(values: pd.Series) -> str:
    lines = [str(value) for value in pd.unique(values)]
    return "".join(f"{line}\n" for line in lines)


def save_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    val_fraction: float = 0.2,
    vehicle_id_column: str = "vehicle_id",
) -> dict[str, Path]:
    """Save CSV partitions plus explicit image/vehicle ID lists and metadata."""

    for name, frame in (("train", train), ("val", val)):
        missing = {"image_id", vehicle_id_column}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} split is empty")
    overlap = set(train[vehicle_id_column]).intersection(val[vehicle_id_column])
    if overlap:
        raise ValueError(f"train/val identity overlap: {list(overlap)[:5]!r}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_csv": destination / "train.csv",
        "val_csv": destination / "val.csv",
        "train_image_ids": destination / "train_image_ids.txt",
        "val_image_ids": destination / "val_image_ids.txt",
        "train_vehicle_ids": destination / "train_vehicle_ids.txt",
        "val_vehicle_ids": destination / "val_vehicle_ids.txt",
        "metadata": destination / "split_metadata.json",
    }
    train.to_csv(paths["train_csv"], index=False)
    val.to_csv(paths["val_csv"], index=False)
    paths["train_image_ids"].write_text(
        _ordered_unique_text(train["image_id"]), encoding="utf-8"
    )
    paths["val_image_ids"].write_text(
        _ordered_unique_text(val["image_id"]), encoding="utf-8"
    )
    paths["train_vehicle_ids"].write_text(
        _ordered_unique_text(train[vehicle_id_column]), encoding="utf-8"
    )
    paths["val_vehicle_ids"].write_text(
        _ordered_unique_text(val[vehicle_id_column]), encoding="utf-8"
    )

    digest = hashlib.sha256()
    digest.update(train.to_csv(index=False).encode("utf-8"))
    digest.update(val.to_csv(index=False).encode("utf-8"))
    metadata: Mapping[str, Any] = {
        "seed": int(seed),
        "requested_val_fraction": float(val_fraction),
        "train_images": int(len(train)),
        "val_images": int(len(val)),
        "train_identities": int(train[vehicle_id_column].nunique()),
        "val_identities": int(val[vehicle_id_column].nunique()),
        "combined_csv_sha256": digest.hexdigest(),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths


def create_and_save_split(
    annotations: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
    min_val_images: int = 2,
    min_val_cameras: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Create and persist a split in one reproducible operation."""

    train, val = identity_disjoint_split(
        annotations,
        val_fraction=val_fraction,
        seed=seed,
        min_val_images=min_val_images,
        min_val_cameras=min_val_cameras,
    )
    paths = save_split(
        train, val, output_dir, seed=seed, val_fraction=val_fraction
    )
    return train, val, paths


# Backwards-friendly concise alias.
split_by_vehicle_id = identity_disjoint_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="training CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-val-images", type=int, default=2)
    parser.add_argument("--min-val-cameras", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train, val, paths = create_and_save_split(
        args.input,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_val_images=args.min_val_images,
        min_val_cameras=args.min_val_cameras,
    )
    print(
        f"saved {len(train)} train and {len(val)} validation rows to "
        f"{paths['train_csv'].parent}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SEED",
    "create_and_save_split",
    "identity_disjoint_split",
    "save_split",
    "split_by_vehicle_id",
]
