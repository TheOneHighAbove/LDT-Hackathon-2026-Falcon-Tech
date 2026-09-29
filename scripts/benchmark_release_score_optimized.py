"""Benchmark the exact speed extractor with the organizer's fixed protocol.

Latency uses 50 warmups followed by 300 synchronized batch-1 runs and reports
their median. Throughput measures batches 1, 8, 16 and 32 for at least ten
seconds each. JPEG read, BBox crop, color descriptor, transforms, host/device
transfer, all three TorchScript backbones and descriptor pooling are included.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.infer_score_optimized import (
    DEVICE,
    PillowSharedBaseCropDataset,
    _backbone_streams,
    _metadata,
    _pool_tokens,
    _run_backbones,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.release_features import dino_part_pool


LATENCY_WARMUPS = 50
LATENCY_RUNS = 300
THROUGHPUT_BATCHES = (1, 8, 16, 32)
THROUGHPUT_MIN_SECONDS = 10.0


def _models(weights: Path) -> dict[str, torch.jit.ScriptModule]:
    names = {
        "conv": "convnext_global_parts.ts",
        "osnet": "osnet_loss_branch_global_parts.ts",
        "dino": "dinov2_vehicle_cls_tokens.ts",
    }
    return {
        name: torch.jit.load(
            str(weights / filename), map_location=DEVICE
        ).eval()
        for name, filename in names.items()
    }


def _normalization_tensors():
    mean = torch.tensor(
        IMAGENET_MEAN, device=DEVICE, dtype=torch.float32
    )[None, :, None, None]
    std = torch.tensor(
        IMAGENET_STD, device=DEVICE, dtype=torch.float32
    )[None, :, None, None]
    return mean, std


def _prepare(batch, mean: torch.Tensor, std: torch.Tensor) -> dict:
    padded = batch["padded"].to(DEVICE, non_blocking=True)
    tight = batch["tight"].to(DEVICE, non_blocking=True)
    osnet = F.interpolate(
        padded,
        (208, 208),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return {
        "conv": (padded - mean) / std,
        "osnet": (osnet - mean) / std,
        "dino": (tight - mean) / std,
    }


@torch.inference_mode()
def _forward(models, inputs, streams) -> torch.Tensor:
    conv_output, osnet_output, dino_output, _ = _run_backbones(
        models, inputs, streams
    )
    conv, conv_parts = conv_output
    identity, metric, osnet_parts = osnet_output
    dino, dino_tokens = dino_output
    values = (
        conv.float(),
        conv_parts.half(),
        identity.float(),
        metric.float(),
        osnet_parts.half(),
        F.normalize(dino.float(), dim=1),
        F.normalize(dino_tokens.mean(1).float(), dim=1),
        dino_part_pool(dino_tokens, 2),
        dino_part_pool(dino_tokens, 4),
        _pool_tokens(dino_tokens, 5).half(),
        _pool_tokens(dino_tokens, 10).half(),
    )
    return sum(value.reshape(-1)[0] for value in values)


def _loader(
    frame: pd.DataFrame,
    images_dir: Path,
    *,
    batch_size: int,
    workers: int,
) -> DataLoader:
    dataset = PillowSharedBaseCropDataset(
        frame,
        images_dir,
        dino_size=224,
        include_clip=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _next_cycling(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _latency(
    frame: pd.DataFrame,
    images_dir: Path,
    models,
    streams,
    mean,
    std,
) -> dict:
    loader = _loader(frame, images_dir, batch_size=1, workers=0)
    iterator = iter(loader)
    checksum = None
    for _ in range(LATENCY_WARMUPS):
        batch, iterator = _next_cycling(loader, iterator)
        checksum = _forward(models, _prepare(batch, mean, std), streams)
        torch.cuda.synchronize()

    samples = []
    for _ in range(LATENCY_RUNS):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        batch, iterator = _next_cycling(loader, iterator)
        checksum = _forward(models, _prepare(batch, mean, std), streams)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if checksum is None or not bool(torch.isfinite(checksum)):
        raise RuntimeError("non-finite benchmark checksum")
    ordered = sorted(samples)
    return {
        "warmup_runs": LATENCY_WARMUPS,
        "measured_runs": LATENCY_RUNS,
        "batch_size": 1,
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(ordered[int(0.95 * (len(ordered) - 1))]),
        "minimum_ms": float(ordered[0]),
        "maximum_ms": float(ordered[-1]),
    }


def _throughput(
    frame: pd.DataFrame,
    images_dir: Path,
    models,
    streams,
    mean,
    std,
    *,
    batch_size: int,
    workers: int,
) -> dict:
    loader = _loader(
        frame, images_dir, batch_size=batch_size, workers=workers
    )
    iterator = iter(loader)
    # Initialize workers, file cache and the exact batch-shape graph before
    # starting the mandatory ten-second measurement window.
    batch, iterator = _next_cycling(loader, iterator)
    checksum = _forward(models, _prepare(batch, mean, std), streams)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    count = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    elapsed = 0.0
    while elapsed < THROUGHPUT_MIN_SECONDS:
        batch, iterator = _next_cycling(loader, iterator)
        checksum = _forward(models, _prepare(batch, mean, std), streams)
        count += len(batch["image_id"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(checksum)):
        raise RuntimeError("non-finite benchmark checksum")
    return {
        "batch_size": batch_size,
        "workers": workers,
        "images": count,
        "seconds": elapsed,
        "images_per_second": count / elapsed,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", type=Path, default=Path("dataset/images")
    )
    parser.add_argument(
        "--query-csv", type=Path, default=Path("dataset/test_query.csv")
    )
    parser.add_argument(
        "--gallery-csv",
        type=Path,
        default=Path("dataset/test_gallery.csv"),
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("weights/release")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/score_optimized_speed/benchmark.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required")

    query = pd.read_csv(args.query_csv)
    gallery = pd.read_csv(args.gallery_csv)
    frame = _metadata(query, gallery)
    models = _models(args.weights_dir)
    streams = _backbone_streams(include_clip=False)
    mean, std = _normalization_tensors()
    throughput = [
        _throughput(
            frame,
            args.images_dir,
            models,
            streams,
            mean,
            std,
            batch_size=batch_size,
            workers=args.workers,
        )
        for batch_size in THROUGHPUT_BATCHES
    ]
    best = max(throughput, key=lambda row: row["images_per_second"])
    report = {
        "protocol": {
            "latency_warmups": LATENCY_WARMUPS,
            "latency_measured_runs": LATENCY_RUNS,
            "latency_statistic": "median synchronized wall time",
            "throughput_batch_sizes": list(THROUGHPUT_BATCHES),
            "minimum_seconds_per_throughput_batch": THROUGHPUT_MIN_SECONDS,
        },
        "scope": (
            "JPEG read, BBox crop, five-region HSV descriptor, resize and "
            "normalization, host/device transfer, three TorchScript "
            "backbones, all release descriptor pooling"
        ),
        "gpu": torch.cuda.get_device_name(DEVICE),
        "torch": torch.__version__,
        "preprocessing": "pillow_shared",
        "dino_input_size": 224,
        "latency": _latency(
            frame, args.images_dir, models, streams, mean, std
        ),
        "throughput": throughput,
        "best_throughput": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
