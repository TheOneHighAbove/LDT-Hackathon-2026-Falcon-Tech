"""Create reproducible cross-camera validation error tables and a visual montage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .config import configured_path, load_config
from .data import crop_vehicle
from .utils import save_json


def build_error_table(embeddings: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) != len(frame):
        raise ValueError("embeddings and validation rows must be aligned")
    embeddings = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None
    )
    scores = embeddings @ embeddings.T
    pids = frame.vehicle_id.to_numpy()
    cameras = frame.camera_id.to_numpy()
    image_ids = frame.image_id.astype(str).to_numpy()
    rows = []
    for index in range(len(frame)):
        valid = np.ones(len(frame), dtype=bool)
        valid[index] = False
        valid &= ~((pids == pids[index]) & (cameras == cameras[index]))
        candidates = np.flatnonzero(valid)
        order = candidates[np.argsort(-scores[index, candidates], kind="stable")]
        matches = pids[order] == pids[index]
        positive_positions = np.flatnonzero(matches)
        if len(positive_positions) == 0:
            continue
        top_index = int(order[0])
        best_positive_index = int(order[positive_positions[0]])
        rows.append(
            {
                "query_image_id": image_ids[index],
                "query_vehicle_id": pids[index],
                "query_camera_id": cameras[index],
                "predicted_image_id": image_ids[top_index],
                "predicted_vehicle_id": pids[top_index],
                "predicted_camera_id": cameras[top_index],
                "top1_similarity": float(scores[index, top_index]),
                "top1_correct": bool(matches[0]),
                "first_correct_rank": int(positive_positions[0] + 1),
                "best_positive_image_id": image_ids[best_positive_index],
                "best_positive_camera_id": cameras[best_positive_index],
                "best_positive_similarity": float(scores[index, best_positive_index]),
                "query_bbox_area": float(frame.iloc[index].w * frame.iloc[index].h),
                "query_index": index,
                "predicted_index": top_index,
                "best_positive_index": best_positive_index,
            }
        )
    return pd.DataFrame(rows)


def _render_crop(frame: pd.DataFrame, index: int, images_dir: Path, size: int = 256) -> Image.Image:
    row = frame.iloc[index]
    with Image.open(images_dir / f"{row.image_id}.jpg") as source:
        source.load()
        crop = crop_vehicle(
            source,
            (row.x, row.y, row.w, row.h),
            padding=0.05,
        )
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def create_error_montage(
    errors: pd.DataFrame,
    frame: pd.DataFrame,
    images_dir: Path,
    destination: Path,
    *,
    rows: int = 6,
) -> None:
    wrong = errors.loc[~errors.top1_correct].nlargest(rows, "top1_similarity")
    tile, caption = 256, 52
    canvas = Image.new("RGB", (tile * 3, (tile + caption) * len(wrong)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_number, (_, error) in enumerate(wrong.iterrows()):
        indices = [
            int(error.query_index),
            int(error.predicted_index),
            int(error.best_positive_index),
        ]
        labels = [
            f"QUERY true={error.query_vehicle_id} cam={error.query_camera_id}",
            f"WRONG id={error.predicted_vehicle_id} s={error.top1_similarity:.3f}",
            f"TRUE rank={error.first_correct_rank} s={error.best_positive_similarity:.3f}",
        ]
        y = row_number * (tile + caption)
        for column, (index, label) in enumerate(zip(indices, labels, strict=True)):
            image = _render_crop(frame, index, images_dir, tile)
            x = column * tile
            canvas.paste(image, (x, y))
            draw.text((x + 5, y + tile + 5), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/val_embeddings.npy"))
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = pd.read_csv(configured_path(config, "splits_dir") / "val.csv")
    embeddings = np.load(args.embeddings, allow_pickle=False)
    table = build_error_table(embeddings, frame)
    output_dir = configured_path(config, "outputs_dir")
    table_path = output_dir / "error_analysis.csv"
    table.to_csv(table_path, index=False)
    create_error_montage(
        table,
        frame,
        configured_path(config, "images_dir"),
        output_dir / "error_cases.jpg",
        rows=args.rows,
    )
    wrong = table.loc[~table.top1_correct]
    summary = {
        "queries": len(table),
        "top1_correct": int(table.top1_correct.sum()),
        "top1_errors": int((~table.top1_correct).sum()),
        "rank1": float(table.top1_correct.mean()),
        "median_first_correct_rank": float(table.first_correct_rank.median()),
        "rank95_first_correct": float(table.first_correct_rank.quantile(0.95)),
        "median_wrong_top1_similarity": float(wrong.top1_similarity.median()),
        "median_wrong_best_positive_similarity": float(
            wrong.best_positive_similarity.median()
        ),
    }
    save_json(summary, output_dir / "error_analysis_summary.json")
    print(f"Saved {table_path} and visual montage ({len(wrong)} Rank-1 errors)")


if __name__ == "__main__":
    main()
