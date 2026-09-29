"""Fast cross-camera expert-fusion experiment.

The experiment keeps the trained ConvNeXt frozen, adds a compact public
vehicle-ReID expert, and combines their normalized descriptors.  It trains no
new network and deliberately keeps only the branch that improved validation.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import configured_path, load_config, resolve_path
from .data import IMAGENET_MEAN, IMAGENET_STD, VehicleDataset, build_eval_transform
from .engine import extract_embeddings, load_inference_checkpoint
from .metrics import evaluate_retrieval
from .split import create_and_save_split
from .utils import (
    LOGGER,
    configure_logging,
    save_json,
    seed_everything,
    seed_worker,
    select_device,
    sha256_file,
)


def l2_normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty [N, D] matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 1e-12).any():
        raise ValueError("embeddings must be finite and non-zero")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def weighted_concatenate(
    primary: np.ndarray,
    expert: np.ndarray,
    *,
    primary_weight: float = 0.5,
) -> np.ndarray:
    """Concatenate normalized experts so cosine is their weighted mean."""

    weight = float(primary_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("primary_weight must be in [0, 1]")
    first, second = l2_normalize(primary), l2_normalize(expert)
    if len(first) != len(second):
        raise ValueError("primary and expert embeddings must have equal row counts")
    fused = np.concatenate(
        (math.sqrt(weight) * first, math.sqrt(1.0 - weight) * second), axis=1
    )
    return l2_normalize(fused)


def _retrieval(embeddings: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    return evaluate_retrieval(
        embeddings,
        embeddings,
        frame["vehicle_id"].to_numpy(),
        frame["vehicle_id"].to_numpy(),
        query_camera_ids=frame["camera_id"].to_numpy(),
        gallery_camera_ids=frame["camera_id"].to_numpy(),
        query_image_ids=frame["image_id"].astype(str).to_numpy(),
        gallery_image_ids=frame["image_id"].astype(str).to_numpy(),
        same_source=True,
    )


def _cache_embeddings(
    path: Path,
    frame: pd.DataFrame,
    extract: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_ids = frame["image_id"].astype(str).to_numpy(dtype=np.str_)
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            cached_ids = archive["image_ids"]
            cached = archive["embeddings"]
        if np.array_equal(cached_ids, expected_ids):
            LOGGER.info("reusing cached embeddings: %s", path)
            return l2_normalize(cached), {"cached": True, "seconds": 0.0}
    started = time.perf_counter()
    embeddings = l2_normalize(extract())
    elapsed = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, image_ids=expected_ids, embeddings=embeddings)
    return embeddings, {
        "cached": False,
        "seconds": elapsed,
        "images_per_second": len(frame) / elapsed,
        "pipeline_ms_per_image": 1000.0 * elapsed / len(frame),
    }


def _torch_embeddings(
    model: nn.Module,
    frame: pd.DataFrame,
    *,
    images_dir: Path,
    preprocessing: dict[str, Any],
    batch_size: int,
    workers: int,
    device: torch.device,
    channels_last: bool,
) -> np.ndarray:
    dataset = VehicleDataset(
        frame,
        images_dir,
        transform=build_eval_transform(
            preprocessing["input_size"],
            resize_mode=str(preprocessing.get("resize_mode", "direct")),
        ),
        bbox_padding=float(preprocessing["bbox_padding"]),
        require_camera_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    embeddings, metadata = extract_embeddings(
        model,
        loader,
        device=device,
        amp=device.type == "cuda",
        tta_horizontal_flip=False,
        channels_last=channels_last,
    )
    if metadata["image_id"].tolist() != frame["image_id"].astype(str).tolist():
        raise RuntimeError("ConvNeXt DataLoader changed annotation order")
    return embeddings


class _OpenVinoVehicleExpert:
    def __init__(self, model_path: Path) -> None:
        os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")
        try:
            from openvino import Core
        except ImportError as exc:  # pragma: no cover - optional experiment dependency
            raise ImportError("expert fusion requires openvino") from exc
        core = Core()
        model = core.read_model(str(model_path))
        self.compiled = core.compile_model(
            model,
            "CPU",
            {"PERFORMANCE_HINT": "THROUGHPUT"},
        )

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return np.asarray(self.compiled([images])[0], dtype=np.float32)


def _osnet_session(model_path: Path) -> _OpenVinoVehicleExpert:
    # The 2.18M-parameter expert is deliberately kept on CPU.  OpenVINO is its
    # native runtime, avoids CUDA-version coupling, and can overlap with the
    # main GPU backbone.
    return _OpenVinoVehicleExpert(model_path)


def _osnet_embeddings(
    session: Any,
    frame: pd.DataFrame,
    *,
    images_dir: Path,
    bbox_padding: float,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (208, 208), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = VehicleDataset(
        frame,
        images_dir,
        transform=transform,
        bbox_padding=bbox_padding,
        require_camera_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    chunks: list[np.ndarray] = []
    observed_ids: list[str] = []
    for batch in loader:
        array = np.ascontiguousarray(batch["image"].numpy(), dtype=np.float32)
        chunks.append(session(array))
        observed_ids.extend(str(value) for value in batch["image_id"])
    if observed_ids != frame["image_id"].astype(str).tolist():
        raise RuntimeError("OSNet DataLoader changed annotation order")
    return l2_normalize(np.concatenate(chunks))


def _load_validation(config: dict[str, Any]) -> pd.DataFrame:
    split_dir = configured_path(config, "splits_dir")
    train_path, val_path = split_dir / "train.csv", split_dir / "val.csv"
    if not train_path.is_file() or not val_path.is_file():
        create_and_save_split(
            configured_path(config, "train_csv"),
            split_dir,
            val_fraction=float(config["validation"]["val_fraction"]),
            seed=int(config["seed"]),
            min_val_images=2,
            min_val_cameras=2,
        )
    return pd.read_csv(val_path)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed, deterministic=True)
    device = select_device(args.device)
    output_dir = resolve_path(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_path(config, args.checkpoint)
    osnet_path = resolve_path(config, args.osnet_model)
    if not osnet_path.is_file():
        raise FileNotFoundError(f"OSNet model does not exist: {osnet_path}")
    val_frame = _load_validation(config)
    images_dir = configured_path(config, "images_dir")
    workers = int(config["data"]["num_workers"])

    model, checkpoint = load_inference_checkpoint(checkpoint_path, device=device)
    channels_last = bool(config["model"].get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    preprocessing = dict(checkpoint["preprocessing"])
    conv_digest = sha256_file(checkpoint_path)[:16]
    osnet_digest = sha256_file(osnet_path)[:16]
    cache_dir = output_dir / "cache"

    def conv(frame: pd.DataFrame) -> np.ndarray:
        return _torch_embeddings(
            model,
            frame,
            images_dir=images_dir,
            preprocessing=preprocessing,
            batch_size=int(args.batch_size),
            workers=workers,
            device=device,
            channels_last=channels_last,
        )

    conv_val, conv_val_time = _cache_embeddings(
        cache_dir / f"convnext_val_{conv_digest}.npz",
        val_frame,
        lambda: conv(val_frame),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    session = _osnet_session(osnet_path)

    def osnet(frame: pd.DataFrame) -> np.ndarray:
        return _osnet_embeddings(
            session,
            frame,
            images_dir=images_dir,
            bbox_padding=float(preprocessing["bbox_padding"]),
            batch_size=int(args.osnet_batch_size),
            workers=workers,
        )

    osnet_val, osnet_val_time = _cache_embeddings(
        cache_dir / f"osnet_openvino_val_{osnet_digest}.npz",
        val_frame,
        lambda: osnet(val_frame),
    )
    del session

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    ablations: dict[str, Any] = {
        "convnext_no_tta": _retrieval(conv_val, val_frame),
        "osnet_vehicle_expert": _retrieval(osnet_val, val_frame),
    }
    best_weight, best_metrics = 1.0, ablations["convnext_no_tta"]
    for weight in weights:
        fused = weighted_concatenate(
            conv_val, osnet_val, primary_weight=weight
        )
        metrics = _retrieval(fused, val_frame)
        ablations[f"weighted_fusion_primary_{weight:.2f}"] = metrics
        if float(metrics["mAP"]) > float(best_metrics["mAP"]):
            best_weight, best_metrics = weight, metrics

    fused_val = weighted_concatenate(
        conv_val, osnet_val, primary_weight=best_weight
    )
    selected_variant = f"weighted_fusion_primary_{best_weight:.2f}"
    np.save(output_dir / "val_embeddings.npy", fused_val.astype(np.float32))

    report = {
        "idea": "frozen_vehicle_expert_weighted_cosine_fusion",
        "selection_rule": "best of five fixed convex cosine weights by validation mAP",
        "best_primary_weight": best_weight,
        "selected_variant": selected_variant,
        "selected_metrics": best_metrics,
        "ablations": ablations,
        "timing": {
            "convnext_val": conv_val_time,
            "osnet_val": osnet_val_time,
        },
        "artifacts": {
            "osnet_model": str(osnet_path.resolve()),
            "osnet_size_bytes": osnet_path.stat().st_size,
        },
    }
    save_json(report, output_dir / "fusion_report.json")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/best.pt"))
    parser.add_argument(
        "--osnet-model",
        type=Path,
        default=Path("weights/osnet_ain_x1_0_vehicle_reid.onnx"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/expert_fusion")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--osnet-batch-size", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    configure_logging(args.verbose)
    report = run_experiment(args)
    for name, metrics in report["ablations"].items():
        LOGGER.info(
            "%s: mAP=%.4f Rank-1=%.4f Rank-5=%.4f",
            name,
            metrics["mAP"],
            metrics["rank1"],
            metrics["rank5"],
        )


if __name__ == "__main__":
    main()


__all__ = [
    "l2_normalize",
    "weighted_concatenate",
]
