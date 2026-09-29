"""Final full-data fine-tune initialized from an honest validation checkpoint.

This stage intentionally performs no validation or model selection: every
labelled identity is used for fitting.  Retrieval/open-set metrics and refusal
calibration are copied from the pre-full-data checkpoint and explicitly marked
as such in the output provenance.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import (
    configured_path,
    load_config,
    normalize_input_size,
    normalize_resize_mode,
)
from .data import IMAGENET_MEAN, IMAGENET_STD, VehicleDataset, build_train_transform
from .engine import (
    build_inference_checkpoint,
    checkpoint_postprocess,
    create_model_from_config,
)
from .sampler import CameraAwarePKBatchSampler
from .train import (
    _make_criterion,
    _make_optimizer_scheduler,
    _optimizer_learning_rates,
    train_one_epoch,
)
from .utils import (
    LOGGER,
    atomic_torch_save,
    configure_logging,
    seed_everything,
    seed_worker,
    select_device,
    sha256_file,
)


def _validated_preprocessing(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent, training-compatible preprocessing snapshot.

    Full-data fitting is a continuation of the selected experiment, so its
    geometry and normalization must come from the selected inference artifact,
    not from a possibly edited YAML file.  The current transform implementation
    supports ImageNet normalization; rejecting another normalization is safer
    than silently training with different semantics and copying false metadata.
    """

    preprocessing = copy.deepcopy(dict(checkpoint["preprocessing"]))
    required = {"input_size", "bbox_padding", "resize_mode", "mean", "std"}
    missing = sorted(required.difference(preprocessing))
    if missing:
        raise ValueError(
            "Initialization checkpoint preprocessing is incomplete; missing: "
            + ", ".join(missing)
        )
    normalize_input_size(preprocessing["input_size"])
    normalize_resize_mode(preprocessing["resize_mode"])
    padding = float(preprocessing["bbox_padding"])
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError(
            "Initialization checkpoint bbox_padding must be finite and non-negative"
        )
    for name, expected in (("mean", IMAGENET_MEAN), ("std", IMAGENET_STD)):
        try:
            values = tuple(float(value) for value in preprocessing[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Initialization checkpoint preprocessing.{name} must contain "
                "three finite numbers"
            ) from exc
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Initialization checkpoint preprocessing.{name} must contain "
                "three finite numbers"
            )
        if any(
            abs(value - reference) > 1e-12
            for value, reference in zip(values, expected, strict=True)
        ):
            raise ValueError(
                f"Unsupported preprocessing.{name}: full-data training currently "
                "supports ImageNet normalization only"
            )
    return preprocessing


def _load_initialization_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Initialization checkpoint must contain a mapping")
    required = {"model_state", "preprocessing", "refusal", "metrics"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "Initialization checkpoint is not an inference checkpoint; missing: "
            + ", ".join(missing)
        )
    if not isinstance(checkpoint["model_state"], Mapping):
        raise ValueError("Initialization checkpoint model_state must be a mapping")
    for name in ("preprocessing", "refusal", "metrics"):
        if not isinstance(checkpoint[name], Mapping):
            raise ValueError(f"Initialization checkpoint {name} must be a mapping")
    return checkpoint


def _make_full_data_loader(
    config: dict[str, Any],
    frame: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[DataLoader, CameraAwarePKBatchSampler, VehicleDataset]:
    data_config = config["data"]
    transform = build_train_transform(
        preprocessing["input_size"],
        mean=tuple(float(value) for value in preprocessing["mean"]),
        std=tuple(float(value) for value in preprocessing["std"]),
        crop_scale=tuple(float(value) for value in data_config["train_crop_scale"]),
        crop_ratio=tuple(float(value) for value in data_config["train_crop_ratio"]),
        horizontal_flip_probability=float(
            data_config["horizontal_flip_probability"]
        ),
        random_erasing_probability=float(
            data_config["random_erasing_probability"]
        ),
        resize_mode=str(preprocessing["resize_mode"]),
    )
    dataset = VehicleDataset(
        frame,
        configured_path(config, "images_dir"),
        transform=transform,
        train=True,
        bbox_padding=float(preprocessing["bbox_padding"]),
        require_camera_id=True,
    )
    training = config["training"]
    sampler = CameraAwarePKBatchSampler(
        dataset.pids,
        dataset.camera_ids,
        identities_per_batch=int(training["identities_per_batch"]),
        instances_per_identity=int(training["instances_per_identity"]),
        seed=int(config["seed"]),
    )
    workers = int(data_config["num_workers"])
    loader_options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(
            data_config.get("pin_memory", True) and device.type == "cuda"
        ),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        **loader_options,
    )
    return loader, sampler, dataset


def _training_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the effective full-data choices inside the model artifact."""

    return {
        "seed": int(config["seed"]),
        "model": copy.deepcopy(dict(config["model"])),
        "data": copy.deepcopy(dict(config["data"])),
        "training": copy.deepcopy(dict(config["training"])),
    }


def run_full_data_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed_everything(int(config["seed"]), deterministic=True)
    device = select_device(str(args.device))

    requested_epochs = getattr(args, "epochs", None)
    epochs = int(
        requested_epochs
        if requested_epochs is not None
        else config["training"].get(
            "full_data_epochs", config["training"]["epochs"]
        )
    )
    if epochs <= 0:
        raise ValueError("Full-data epochs must be positive")
    max_train_batches = getattr(args, "max_train_batches", None)
    if max_train_batches is not None and int(max_train_batches) <= 0:
        raise ValueError("--max-train-batches must be positive")

    init_argument = getattr(args, "init_checkpoint", None)
    init_path = (
        Path(init_argument).expanduser().resolve()
        if init_argument is not None
        else (configured_path(config, "weights_dir") / "best.pt").resolve()
    )
    init_checkpoint = _load_initialization_checkpoint(init_path)
    init_sha256 = sha256_file(init_path)
    preprocessing = _validated_preprocessing(init_checkpoint)

    refusal = copy.deepcopy(dict(init_checkpoint["refusal"]))
    threshold = float(refusal.get("similarity_threshold", float("nan")))
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError(
            "Initialization checkpoint refusal threshold must be finite and "
            "within [-1, 1]"
        )
    calibration = refusal.get("calibration", {})
    if not isinstance(calibration, Mapping):
        raise ValueError("Initialization checkpoint calibration must be a mapping")

    destination = configured_path(config, "weights_dir") / "full_data.pt"
    if destination.resolve() == init_path:
        raise ValueError(
            "--init-checkpoint must not be the full-data output path "
            f"({destination.resolve()})"
        )

    frame = pd.read_csv(configured_path(config, "train_csv"))
    loader, sampler, dataset = _make_full_data_loader(
        config, frame, preprocessing, device=device
    )
    LOGGER.info(
        "Full-data fit on %s: %d rows, %d IDs, PK batch=%d, epochs=%d",
        device,
        len(dataset),
        len(dataset.pid_to_label),
        sampler.batch_size,
        epochs,
    )

    # Deliberately ignore checkpoint model metadata for construction: the
    # current config is the complete experiment definition.  strict=True then
    # proves that its architecture is compatible with the initialization.
    model = create_model_from_config(config, pretrained=False).to(device)
    try:
        model.load_state_dict(init_checkpoint["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The configured model is not strictly compatible with "
            f"--init-checkpoint {init_path}: {exc}"
        ) from exc

    channels_last = bool(config["model"].get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = _make_criterion(
        config, len(dataset.pid_to_label), device=device
    )
    optimizer, scheduler = _make_optimizer_scheduler(
        model, criterion, config, epochs
    )
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        metrics = train_one_epoch(
            model,
            criterion,
            loader,
            optimizer,
            scaler,
            device=device,
            amp=use_amp,
            channels_last=channels_last,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            max_batches=(
                int(max_train_batches) if max_train_batches is not None else None
            ),
        )
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **metrics,
            **_optimizer_learning_rates(optimizer, config),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        LOGGER.info(
            "Full-data epoch %d/%d loss=%.4f classification=%.4f "
            "metric=%.4f (%.1fs)",
            epoch + 1,
            epochs,
            metrics["loss"],
            metrics["classification_loss"],
            metrics["metric_loss"],
            row["seconds"],
        )

    source = {
        "stage": "full_data_finetune",
        "full_data_finetune": True,
        "full_data_rows": len(dataset),
        "full_data_vehicle_ids": len(dataset.pid_to_label),
        "full_data_cameras": len(set(dataset.camera_ids)),
        "full_data_epochs": epochs,
        "full_data_batches_per_epoch": len(sampler),
        "development_max_train_batches": (
            int(max_train_batches) if max_train_batches is not None else None
        ),
        "initialization_checkpoint": str(init_path),
        "initialization_checkpoint_sha256": init_sha256,
        "strict_model_state_initialization": True,
        "fresh_optimizer_scheduler": True,
        "fresh_training_classifier": True,
        "preprocessing_inherited_from_initialization": True,
        "effective_preprocessing": copy.deepcopy(preprocessing),
        "metrics_evaluated_after_full_data_finetune": False,
        "metrics_provenance": (
            "copied from the pre-full-data initialization checkpoint; "
            "no validation/model selection was performed after fitting all IDs"
        ),
        "parent_source": copy.deepcopy(dict(init_checkpoint.get("source", {}))),
        "config_path": str(Path(config["_config_path"])),
        "config_sha256": sha256_file(config["_config_path"]),
        "config_snapshot": _training_snapshot(config),
    }

    output_checkpoint = build_inference_checkpoint(
        model,
        input_size=preprocessing["input_size"],
        bbox_padding=float(preprocessing["bbox_padding"]),
        resize_mode=str(preprocessing["resize_mode"]),
        tta_horizontal_flip=bool(preprocessing.get("tta_horizontal_flip", False)),
        postprocess=checkpoint_postprocess(init_checkpoint),
        refusal_threshold=threshold,
        calibration=dict(calibration),
        metrics=copy.deepcopy(dict(init_checkpoint["metrics"])),
        source=source,
    )
    # Preserve these evaluated/inference semantics byte-for-byte at the Python
    # object level, including any future extension fields unknown to this CLI.
    output_checkpoint["preprocessing"] = preprocessing
    output_checkpoint["refusal"] = refusal
    output_checkpoint["metrics"] = copy.deepcopy(dict(init_checkpoint["metrics"]))
    if "search" in init_checkpoint:
        output_checkpoint["search"] = copy.deepcopy(init_checkpoint["search"])
    atomic_torch_save(output_checkpoint, destination)
    elapsed = time.perf_counter() - started
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": sha256_file(destination),
        "initialization_checkpoint": str(init_path),
        "rows": len(dataset),
        "vehicle_ids": len(dataset.pid_to_label),
        "epochs": epochs,
        "wall_seconds": elapsed,
        "history": history,
        "metrics_provenance": source["metrics_provenance"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="inference checkpoint providing strict model_state initialization; "
        "defaults to <weights_dir>/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="defaults to training.full_data_epochs or training.epochs",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="development smoke-test only; never use for the final artifact",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    configure_logging(args.verbose)
    summary = run_full_data_training(args)
    LOGGER.info(
        "Full-data checkpoint saved to %s (%d rows, %d IDs, %.1fs)",
        summary["checkpoint"],
        summary["rows"],
        summary["vehicle_ids"],
        summary["wall_seconds"],
    )


if __name__ == "__main__":
    main()


__all__ = ["run_full_data_training"]
