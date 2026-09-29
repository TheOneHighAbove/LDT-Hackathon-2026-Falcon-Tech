"""Evaluate a production extractor/ranker on locked confirmation protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_score_optimized import extract_features, rank_gallery


CONFIRM_SEEDS = (101, 211, 307, 401, 503)


def _camera_representative_protocol(frame: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    gallery = []
    for _, group in frame.groupby(
        ["vehicle_id", "camera_id"], sort=False
    ):
        indices = group.index.to_numpy()
        gallery.append(int(indices[int(rng.integers(len(indices)))]))
    gallery_indices = np.asarray(gallery, dtype=np.int64)
    is_query = np.ones(len(frame), dtype=bool)
    is_query[gallery_indices] = False
    return np.flatnonzero(is_query), gallery_indices


def _official(order: np.ndarray, query: pd.DataFrame, gallery: pd.DataFrame):
    gallery_pid = gallery.vehicle_id.to_numpy()
    gallery_camera = gallery.camera_id.to_numpy()
    average_precision, rank1, rank5 = [], [], []
    for ranking, row in zip(
        order, query.itertuples(index=False), strict=True
    ):
        positive = (gallery_pid == row.vehicle_id) & (
            gallery_camera != row.camera_id
        )
        count_positive = int(positive.sum())
        if count_positive == 0:
            continue
        junk = (gallery_pid == row.vehicle_id) & (
            gallery_camera == row.camera_id
        )
        clean = ranking[~junk[ranking]][:10]
        relevant = positive[clean]
        precision = np.cumsum(relevant) / np.arange(
            1, len(clean) + 1
        )
        average_precision.append(
            float((precision * relevant).sum() / min(count_positive, 10))
        )
        rank1.append(bool(relevant[:1].any()))
        rank5.append(bool(relevant[:5].any()))
    return {
        "mAP@10": float(np.mean(average_precision)),
        "Rank-1": float(np.mean(rank1)),
        "Rank-5": float(np.mean(rank5)),
        "n_scored": len(average_precision),
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("mAP@10", "Rank-1", "Rank-5", "n_scored")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessing", choices=("pillow", "pillow_shared", "opencv"),
        default="pillow",
    )
    parser.add_argument("--dino-size", type=int, choices=(224, 280), default=280)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/score_optimized_speed/validation.json"),
    )
    args = parser.parse_args()

    frame = pd.read_csv("splits/val.csv").reset_index(drop=True)
    release = json.loads(
        Path("configs/score_optimized_speed.json").read_text(encoding="utf-8")
    )
    arrays, extraction = extract_features(
        frame, Path("dataset/images"), Path("weights/release"),
        args.batch_size, args.workers, preprocessing=args.preprocessing,
        dino_size=args.dino_size, include_clip=False,
    )
    metrics = []
    for seed in CONFIRM_SEEDS:
        query_indices, gallery_indices = _camera_representative_protocol(frame, seed)
        indices = np.concatenate((query_indices, gallery_indices))
        episode = frame.iloc[indices].reset_index(drop=True).copy()
        episode["split"] = ["query"] * len(query_indices) + ["gallery"] * len(gallery_indices)
        episode_arrays = {name: values[indices] for name, values in arrays.items()}
        order, _, _ = rank_gallery(
            episode, episode_arrays, Path("weights"), release,
            include_clip=False, colors=episode_arrays["colors"],
        )
        metrics.append(_official(
            order,
            frame.iloc[query_indices].reset_index(drop=True),
            frame.iloc[gallery_indices].reset_index(drop=True),
        ))
    report = {
        "profile": "speed",
        "preprocessing": args.preprocessing,
        "dino_input_size": args.dino_size,
        "seeds": list(CONFIRM_SEEDS),
        "extraction": extraction,
        "per_seed": metrics,
        "confirmation": _aggregate(metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
