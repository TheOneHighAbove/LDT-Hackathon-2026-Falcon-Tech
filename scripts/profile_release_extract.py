"""Break strict batch-1 extraction latency into CPU and CUDA stages."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_release_score_optimized import _forward, _models
from scripts.infer_score_optimized import (
    DEVICE,
    FastSharedCropDataset,
    PillowSharedBaseCropDataset,
    SharedCropDataset,
    _backbone_streams,
    _metadata,
)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
    }


@torch.inference_mode()
def main() -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required")
    query = pd.read_csv("dataset/test_query.csv")
    gallery = pd.read_csv("dataset/test_gallery.csv")
    frame = _metadata(query, gallery).iloc[:64]
    preprocessing = os.environ.get("RELEASE_PREPROCESSING", "pillow")
    dataset_type = {
        "pillow": SharedCropDataset,
        "opencv": FastSharedCropDataset,
        "pillow_shared": PillowSharedBaseCropDataset,
    }[preprocessing]
    dino_size = int(os.environ.get("RELEASE_DINO_SIZE", "280"))
    dataset = dataset_type(
        frame, Path("dataset/images"), dino_size=dino_size, include_clip=False
    )
    models = _models(include_clip=False)
    streams = _backbone_streams(include_clip=False)

    warm = dataset[0]
    if preprocessing == "pillow_shared":
        from torch.nn import functional as F
        from scripts.infer_score_optimized import IMAGENET_MEAN, IMAGENET_STD
        padded = warm["padded"].unsqueeze(0).to(DEVICE)
        tight = warm["tight"].unsqueeze(0).to(DEVICE)
        mean = torch.tensor(IMAGENET_MEAN, device=DEVICE)[None, :, None, None]
        std = torch.tensor(IMAGENET_STD, device=DEVICE)[None, :, None, None]
        warm_inputs = {
            "conv": (padded - mean) / std,
            "osnet": (
                F.interpolate(
                    padded, (208, 208), mode="bicubic",
                    align_corners=False, antialias=True,
                ) - mean
            ) / std,
            "dino": (tight - mean) / std,
        }
    else:
        warm_inputs = {
            name: warm[name].unsqueeze(0).to(DEVICE) for name in models
        }
    for _ in range(10):
        _forward(models, warm_inputs, streams)
    torch.cuda.synchronize()

    cpu_ms, cuda_ms, total_ms = [], [], []
    for index in range(len(dataset)):
        total_started = time.perf_counter()
        cpu_started = total_started
        item = dataset[index]
        cpu_finished = time.perf_counter()
        if preprocessing == "pillow_shared":
            from torch.nn import functional as F
            from scripts.infer_score_optimized import IMAGENET_MEAN, IMAGENET_STD
            padded = item["padded"].unsqueeze(0).to(DEVICE)
            tight = item["tight"].unsqueeze(0).to(DEVICE)
            mean = torch.tensor(IMAGENET_MEAN, device=DEVICE)[None, :, None, None]
            std = torch.tensor(IMAGENET_STD, device=DEVICE)[None, :, None, None]
            inputs = {
                "conv": (padded - mean) / std,
                "osnet": (
                    F.interpolate(
                        padded, (208, 208), mode="bicubic",
                        align_corners=False, antialias=True,
                    ) - mean
                ) / std,
                "dino": (tight - mean) / std,
            }
        else:
            inputs = {
                name: item[name].unsqueeze(0).to(DEVICE) for name in models
            }
        cuda_started = time.perf_counter()
        value = _forward(models, inputs, streams)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if not torch.isfinite(value):
            raise RuntimeError("non-finite profile checksum")
        cpu_ms.append(1000.0 * (cpu_finished - cpu_started))
        cuda_ms.append(1000.0 * (finished - cuda_started))
        total_ms.append(1000.0 * (finished - total_started))

    report = {
        "gpu": torch.cuda.get_device_name(DEVICE),
        "profile": "speed",
        "preprocessing": preprocessing,
        "dino_input_size": dino_size,
        "images": len(dataset),
        "jpeg_crop_resize_normalize": _stats(cpu_ms),
        "transfer_forward_pool_sync": _stats(cuda_ms),
        "strict_total": _stats(total_ms),
    }
    output = Path("outputs/score_optimized/extract_stage_profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
