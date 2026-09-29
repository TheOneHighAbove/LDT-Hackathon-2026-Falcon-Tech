"""Build the exact production global descriptor used to calibrate refusal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_score_optimized import (
    _normalize,
    _weighted_concat,
    extract_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config", type=Path,
        default=Path("configs/score_optimized_speed.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/score_optimized/val_global_embeddings_speed.npy"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    release = json.loads(args.release_config.read_text(encoding="utf-8"))
    runtime = release["runtime"]
    include_clip = release.get("profile", "quality") == "quality"
    frame = pd.read_csv("splits/val.csv")
    arrays, extraction = extract_features(
        frame, Path("dataset/images"), Path("weights/release"),
        args.batch_size, args.workers,
        preprocessing=runtime["preprocessing"],
        dino_size=int(runtime["dino_input_size"]),
        include_clip=include_clip,
    )
    specialized = _weighted_concat(
        (arrays["os_identity"], arrays["os_metric"]), (0.75, 0.25)
    )
    osnet = _weighted_concat((arrays["conv"], specialized), (0.20, 0.80))
    fused = _weighted_concat((osnet, _normalize(arrays["dino"])), (0.75, 0.25))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, fused.astype(np.float32), allow_pickle=False)
    payload = {
        "description": "exact production 0.75 OSNet/Conv + 0.25 DINOv2 CLS",
        "profile": release["profile"],
        "runtime": runtime,
        "rows": len(fused),
        "dimensions": fused.shape[1],
        "mean_l2_norm": float(np.linalg.norm(fused, axis=1).mean()),
        "uses_camera": False,
        "uses_other_queries": False,
        "extraction": extraction,
    }
    report = args.output.with_suffix(".json")
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
