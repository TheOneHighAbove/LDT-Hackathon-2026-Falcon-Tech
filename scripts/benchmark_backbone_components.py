"""Benchmark each image backbone used by the retained release profiles.

The benchmark measures CUDA forward time only.  Official latency additionally
includes decode, crop, preprocessing and normalization, so these numbers are
for relative architecture decisions rather than leaderboard claims.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch


DEVICE = torch.device("cuda")


def _timed(
    model: torch.nn.Module,
    forward: Callable[[torch.Tensor], object],
    *,
    size: int,
    batch: int,
    iterations: int,
) -> dict:
    image = torch.randn(batch, 3, size, size, device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            forward(image)
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward(image)
            end.record()
            torch.cuda.synchronize()
            elapsed.append(start.elapsed_time(end))
    median = float(np.median(elapsed))
    return {
        "batch": batch,
        "median_batch_ms": median,
        "milliseconds_per_image": median / batch,
        "images_per_second": 1000.0 * batch / median,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def _release(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _convnext_tiny():
    from src.engine import load_inference_checkpoint

    model, _ = load_inference_checkpoint("weights/best.pt", device=DEVICE)
    return model.eval(), lambda image: model(image), 256, [Path("weights/best.pt")]


def _osnet():
    from scripts.probe_osnet_loss_branch_fusion import _model

    model = _model(Path("weights/osnet_target_loss_branches.pt")).eval()
    return (
        model,
        lambda image: model.forward_branches(image),
        208,
        [Path("weights/osnet_target_loss_branches.pt")],
    )


def _dinov2():
    from scripts.train_dinov2_vehicle_metric import _model

    model = _model()
    payload = torch.load(
        "weights/dinov2_vehicle_metric_compact.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()
    return (
        model,
        lambda image: model.forward_features(image),
        196,
        [Path("weights/dinov2_vehicle_metric_compact.pt")],
    )


def _clip():
    from scripts.train_clip_vehicle_metric import _visual

    model = _visual()
    payload = torch.load(
        "weights/clip_vitb16_vehicle_metric_stage2.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return (
        model,
        lambda image: model(image),
        224,
        [Path("weights/clip_vitb16_vehicle_metric_stage2.pt")],
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    loaders = {
        "osnet": _osnet,
        "dinov2_vitb14": _dinov2,
        "convnext_tiny_local": _convnext_tiny,
        "clip_vitb16": _clip,
    }
    report = {
        "scope": "CUDA forward only; official extract latency also includes image I/O and preprocessing",
        "gpu": torch.cuda.get_device_name(DEVICE),
        "components": {},
    }
    for name, loader in loaders.items():
        try:
            model, forward, size, weights = loader()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            result = {
                "input_size": size,
                "parameters": parameters,
                "declared_weight_bytes": sum(path.stat().st_size for path in weights),
                "batch1": _timed(
                    model, forward, size=size, batch=1, iterations=30
                ),
                "batch32": _timed(
                    model, forward, size=size, batch=32, iterations=15
                ),
            }
            report["components"][name] = result
            print(name, json.dumps(result), flush=True)
            _release(model)
        except Exception as error:  # preserve partial benchmark evidence
            report["components"][name] = {
                "error": f"{type(error).__name__}: {error}"
            }
            print(name, report["components"][name], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    output = Path("outputs/retrieval_v2/backbone_component_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
