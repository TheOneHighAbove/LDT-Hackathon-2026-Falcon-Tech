"""Audit annotation integrity, identity/camera distribution, images, and BBoxes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .config import configured_path, load_config
from .utils import configure_logging, save_json, sha256_file


def _describe(values: pd.Series) -> dict[str, float]:
    quantiles = values.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {f"p{int(q * 100):02d}": float(value) for q, value in quantiles.items()}


def _frame_summary(frame: pd.DataFrame, image_width: int, image_height: int) -> dict[str, Any]:
    area = frame.w.astype(float) * frame.h.astype(float)
    touches = (
        (frame.x <= 0)
        | (frame.y <= 0)
        | (frame.x + frame.w >= image_width)
        | (frame.y + frame.h >= image_height)
    )
    near_edge = (
        (frame.x <= 5)
        | (frame.y <= 5)
        | (frame.x + frame.w >= image_width - 5)
        | (frame.y + frame.h >= image_height - 5)
    )
    return {
        "rows": int(len(frame)),
        "unique_image_ids": int(frame.image_id.nunique()),
        "duplicate_image_ids": int(frame.image_id.duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "bbox": {
            "x": _describe(frame.x),
            "y": _describe(frame.y),
            "w": _describe(frame.w),
            "h": _describe(frame.h),
            "area_fraction": _describe(area / (image_width * image_height)),
            "nonpositive": int(((frame.w <= 0) | (frame.h <= 0)).sum()),
            "outside_image": int(
                (
                    (frame.x < 0)
                    | (frame.y < 0)
                    | (frame.x + frame.w > image_width)
                    | (frame.y + frame.h > image_height)
                ).sum()
            ),
            "touches_boundary": int(touches.sum()),
            "within_5px_of_boundary": int(near_edge.sum()),
        },
    }


def analyze(config: dict[str, Any], verify_images: bool = True) -> dict[str, Any]:
    csv_paths = {
        "train": configured_path(config, "train_csv"),
        "query": configured_path(config, "query_csv"),
        "gallery": configured_path(config, "gallery_csv"),
    }
    frames = {name: pd.read_csv(path) for name, path in csv_paths.items()}
    image_dir = configured_path(config, "images_dir")
    image_files = sorted(image_dir.glob("*.jpg"))
    dimensions: Counter[tuple[int, int, str]] = Counter()
    corrupt: list[str] = []
    total_bytes = 0
    if verify_images:
        for path in tqdm(image_files, desc="verify JPEG", leave=False):
            total_bytes += path.stat().st_size
            try:
                with Image.open(path) as image:
                    dimensions[(image.width, image.height, image.mode)] += 1
                    image.verify()
            except OSError:
                corrupt.append(path.name)
    else:
        total_bytes = sum(path.stat().st_size for path in image_files)
        if image_files:
            with Image.open(image_files[0]) as image:
                dimensions[(image.width, image.height, image.mode)] = len(image_files)
    if not dimensions:
        raise RuntimeError(f"No readable JPEG files found under {image_dir}")
    most_common_dimension = dimensions.most_common(1)[0][0]
    image_width, image_height = most_common_dimension[:2]

    all_csv_ids = set().union(*(set(frame.image_id.astype(str)) for frame in frames.values()))
    disk_ids = {path.stem for path in image_files}
    identity_counts = frames["train"].groupby("vehicle_id").size()
    camera_counts = frames["train"].groupby("vehicle_id").camera_id.nunique()
    report: dict[str, Any] = {
        "csv": {
            name: {
                **_frame_summary(frame, image_width, image_height),
                "columns": frame.columns.tolist(),
                "sha256": sha256_file(csv_paths[name]),
            }
            for name, frame in frames.items()
        },
        "global": {
            "unique_annotation_image_ids": len(all_csv_ids),
            "image_files": len(image_files),
            "annotation_ids_missing_on_disk": sorted(all_csv_ids - disk_ids)[:100],
            "unannotated_disk_ids": sorted(disk_ids - all_csv_ids)[:100],
            "train_query_overlap": len(
                set(frames["train"].image_id) & set(frames["query"].image_id)
            ),
            "train_gallery_overlap": len(
                set(frames["train"].image_id) & set(frames["gallery"].image_id)
            ),
            "query_gallery_overlap": len(
                set(frames["query"].image_id) & set(frames["gallery"].image_id)
            ),
        },
        "images": {
            "total_bytes": total_bytes,
            "dimension_mode_histogram": {
                f"{width}x{height}_{mode}": count
                for (width, height, mode), count in dimensions.items()
            },
            "corrupt_count": len(corrupt),
            "corrupt_examples": corrupt[:100],
        },
        "train_identities": {
            "num_identities": int(len(identity_counts)),
            "images_per_identity": _describe(identity_counts),
            "images_per_identity_histogram": {
                str(key): int(value)
                for key, value in identity_counts.value_counts().sort_index().items()
            },
            "num_cameras": int(frames["train"].camera_id.nunique()),
            "cameras_per_identity": _describe(camera_counts),
            "cameras_per_identity_histogram": {
                str(key): int(value)
                for key, value in camera_counts.value_counts().sort_index().items()
            },
            "identities_with_at_least_2_images": int((identity_counts >= 2).sum()),
            "identities_with_at_least_2_cameras": int((camera_counts >= 2).sum()),
        },
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--skip-image-validation", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging()
    config = load_config(args.config)
    report = analyze(config, verify_images=not args.skip_image_validation)
    destination = args.output or configured_path(config, "outputs_dir") / "data_analysis.json"
    save_json(report, destination)
    print(f"Data audit saved to {destination}")


if __name__ == "__main__":
    main()
