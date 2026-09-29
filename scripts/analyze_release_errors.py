"""Produce reproducible error diagnostics for the frozen speed release.

Camera and identity labels are used only to score the labelled validation
split.  They are never passed to release inference or used as ranking inputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from scripts.infer_score_optimized import extract_features, rank_gallery
from scripts.validate_release_profile import (
    CONFIRM_SEEDS,
    _camera_representative_protocol,
)
from src.data import crop_vehicle


def _relative_bbox_area(row: pd.Series, images_dir: Path) -> float:
    with Image.open(images_dir / f"{row.image_id}.jpg") as image:
        image_area = max(1, image.width * image.height)
    return float(max(0, row.w) * max(0, row.h) / image_area)


def _evaluate_episode(
    frame: pd.DataFrame,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    order: np.ndarray,
    score: np.ndarray,
    *,
    seed: int,
) -> list[dict]:
    pids = frame.vehicle_id.to_numpy()
    cameras = frame.camera_id.to_numpy()
    rows: list[dict] = []
    for local_query, (query_index, ranking) in enumerate(
        zip(query_indices, order, strict=True)
    ):
        query_pid = pids[query_index]
        query_camera = cameras[query_index]
        gallery_pid = pids[gallery_indices]
        gallery_camera = cameras[gallery_indices]
        positive = (gallery_pid == query_pid) & (
            gallery_camera != query_camera
        )
        if not bool(positive.any()):
            continue
        junk = (gallery_pid == query_pid) & (
            gallery_camera == query_camera
        )
        clean = ranking[~junk[ranking]]
        relevant = positive[clean[:10]]
        precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
        ap10 = float(
            (precision * relevant).sum() / min(int(positive.sum()), 10)
        )
        relevant_positions = np.flatnonzero(positive[clean])
        first_positive_rank = (
            int(relevant_positions[0] + 1) if len(relevant_positions) else None
        )
        official_top1 = int(clean[0])
        best_positive_local = (
            int(clean[relevant_positions[0]])
            if len(relevant_positions)
            else int(np.flatnonzero(positive)[0])
        )
        rows.append(
            {
                "seed": int(seed),
                "query_index": int(query_index),
                "query_image_id": str(frame.iloc[query_index].image_id),
                "vehicle_id": int(query_pid),
                "camera_id": int(query_camera),
                "ap10": ap10,
                "rank1": bool(relevant[:1].any()),
                "rank5": bool(relevant[:5].any()),
                "rank10": bool(relevant.any()),
                "first_positive_rank": first_positive_rank,
                "top1_index": int(gallery_indices[official_top1]),
                "top1_score": float(score[local_query, official_top1]),
                "best_positive_index": int(
                    gallery_indices[best_positive_local]
                ),
                "best_positive_score": float(
                    score[local_query, best_positive_local]
                ),
            }
        )
    return rows


def _summary(rows: pd.DataFrame) -> dict:
    count = len(rows)
    first_rank = rows.first_positive_rank.fillna(np.inf)
    return {
        "scored_queries": count,
        "mAP@10": float(rows.ap10.mean()),
        "rank1": float(rows.rank1.mean()),
        "rank5": float(rows.rank5.mean()),
        "rank10": float(rows.rank10.mean()),
        "rank1_failures": int((~rows.rank1).sum()),
        "rescued_at_ranks_2_to_5": int(((first_rank >= 2) & (first_rank <= 5)).sum()),
        "rescued_at_ranks_6_to_10": int(((first_rank >= 6) & (first_rank <= 10)).sum()),
        "no_positive_in_top10": int((first_rank > 10).sum()),
    }


def _crop(row: pd.Series, images_dir: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(images_dir / f"{row.image_id}.jpg") as source:
        cropped = crop_vehicle(
            source.convert("RGB"),
            (row.x, row.y, row.w, row.h),
            padding=0.02,
        )
        cropped.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (28, 30, 34))
    canvas.paste(
        cropped,
        ((size[0] - cropped.width) // 2, (size[1] - cropped.height) // 2),
    )
    return canvas


def _render_montage(
    frame: pd.DataFrame,
    cases: list[dict],
    images_dir: Path,
    output: Path,
) -> None:
    tile = (260, 180)
    caption = 48
    margin = 12
    width = margin * 4 + tile[0] * 3
    height = margin * (len(cases) + 1) + (tile[1] + caption) * len(cases)
    canvas = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    labels = ("QUERY", "WRONG TOP-1", "BEST CORRECT")
    for row_number, case in enumerate(cases):
        y = margin + row_number * (tile[1] + caption + margin)
        indices = (
            case["query_index"],
            case["top1_index"],
            case["best_positive_index"],
        )
        for column, (label, index) in enumerate(zip(labels, indices, strict=True)):
            x = margin + column * (tile[0] + margin)
            item = frame.iloc[index]
            canvas.paste(_crop(item, images_dir, tile), (x, y))
            color = (108, 170, 255) if column == 0 else (
                (245, 102, 102) if column == 1 else (97, 205, 138)
            )
            draw.rectangle(
                (x, y, x + tile[0] - 1, y + tile[1] - 1),
                outline=color,
                width=4,
            )
            text = f"{label}  id={item.vehicle_id} cam={item.camera_id}"
            if column == 0:
                text += f"  first+={case['first_positive_rank']}"
            draw.text((x, y + tile[1] + 8), text, fill=(235, 238, 242), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=Path("configs/score_optimized_speed.json"),
    )
    parser.add_argument("--annotations", type=Path, default=Path("splits/val.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/score_optimized_speed/error_analysis.json"),
    )
    parser.add_argument(
        "--montage",
        type=Path,
        default=Path("docs/error_analysis_montage.jpg"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--montage-cases", type=int, default=12)
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations).reset_index(drop=True)
    release = json.loads(args.release_config.read_text(encoding="utf-8"))
    arrays, extraction = extract_features(
        frame,
        args.images_dir,
        args.weights_dir / "release",
        args.batch_size,
        args.workers,
        preprocessing=release["runtime"]["preprocessing"],
        dino_size=int(release["runtime"]["dino_input_size"]),
        include_clip=False,
    )

    evaluations: list[dict] = []
    per_seed: list[dict] = []
    montage_cases: list[dict] = []
    for seed in CONFIRM_SEEDS:
        query_indices, gallery_indices = _camera_representative_protocol(
            frame, seed
        )
        indices = np.concatenate((query_indices, gallery_indices))
        episode = frame.iloc[indices].reset_index(drop=True).copy()
        episode["split"] = ["query"] * len(query_indices) + [
            "gallery"
        ] * len(gallery_indices)
        episode_arrays = {
            name: values[indices] for name, values in arrays.items()
        }
        order, score, _ = rank_gallery(
            episode,
            episode_arrays,
            args.weights_dir,
            release,
            include_clip=False,
            colors=episode_arrays["colors"],
        )
        rows = _evaluate_episode(
            frame,
            query_indices,
            gallery_indices,
            order,
            score,
            seed=seed,
        )
        evaluations.extend(rows)
        seed_frame = pd.DataFrame(rows)
        per_seed.append({"seed": seed, **_summary(seed_frame)})
        if seed == CONFIRM_SEEDS[0]:
            failures = seed_frame.loc[~seed_frame.rank1].copy()
            failures["missing_rank"] = failures.first_positive_rank.fillna(10_000)
            montage_cases = failures.sort_values(
                ["missing_rank", "ap10", "top1_score"],
                ascending=[False, True, False],
            ).head(args.montage_cases).to_dict("records")

    detail = pd.DataFrame(evaluations)
    area_by_image = {
        str(row.image_id): _relative_bbox_area(row, args.images_dir)
        for _, row in frame.iterrows()
    }
    detail["relative_bbox_area"] = detail.query_image_id.map(area_by_image)
    detail["bbox_quartile"] = pd.qcut(
        detail.relative_bbox_area,
        q=4,
        labels=("smallest", "small", "large", "largest"),
        duplicates="drop",
    )
    by_bbox = {
        str(name): _summary(group)
        for name, group in detail.groupby("bbox_quartile", observed=True)
    }

    image_failures = Counter(
        detail.loc[~detail.rank1, "query_image_id"].tolist()
    )
    image_appearances = Counter(detail.query_image_id.tolist())
    stable = []
    for image_id, failures in image_failures.items():
        appearances = image_appearances[image_id]
        if appearances >= 3 and failures == appearances:
            stable.append(
                {
                    "query_image_id": image_id,
                    "failures": failures,
                    "appearances": appearances,
                }
            )
    stable.sort(key=lambda row: (-row["failures"], row["query_image_id"]))

    first_rank = detail.first_positive_rank.fillna(np.inf)
    error_modes = {
        "rank1_wrong_but_positive_at_2_to_5": int(
            ((first_rank >= 2) & (first_rank <= 5)).sum()
        ),
        "positive_only_at_6_to_10": int(
            ((first_rank >= 6) & (first_rank <= 10)).sum()
        ),
        "no_positive_in_top10": int((first_rank > 10).sum()),
        "stable_rank1_failure_images": len(stable),
    }
    report = {
        "schema_version": 1,
        "profile": "speed",
        "scope": (
            "offline diagnostics only; vehicle_id and camera_id are never "
            "used by release inference"
        ),
        "seeds": list(CONFIRM_SEEDS),
        "extraction": extraction,
        "aggregate": _summary(detail),
        "per_seed": per_seed,
        "error_modes": error_modes,
        "by_relative_bbox_area_quartile": by_bbox,
        "stable_rank1_failures": stable[:25],
        "montage_cases": montage_cases,
        "montage": str(args.montage),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _render_montage(frame, montage_cases, args.images_dir, args.montage)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
