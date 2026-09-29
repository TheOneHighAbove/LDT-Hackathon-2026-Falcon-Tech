"""Fine-tune the vehicle ReID model with ArcFace and batch-hard triplet loss."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import (
    configured_path,
    load_config,
    normalize_cross_batch_memory_config,
    normalize_hard_identity_mining_config,
    normalize_multi_similarity_config,
)
from .data import VehicleDataset, build_eval_transform, build_train_transform
from .engine import create_model_from_config, extract_embeddings, save_inference_checkpoint
from .kfold import resolve_kfold_partition, split_provenance_fingerprint
from .losses import ReIDLoss
from .sampler import (
    CameraAwarePKBatchSampler,
    HardIdentityMiningPKBatchSampler,
    build_identity_neighbor_map,
)
from .split import create_and_save_split
from .utils import (
    LOGGER,
    atomic_torch_save,
    configure_logging,
    cpu_state_dict,
    save_json,
    seed_everything,
    seed_worker,
    select_device,
    sha256_file,
)
from .validation import evaluate_validation


def _ensure_split(config: dict[str, Any], recreate: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = configured_path(config, "splits_dir")
    train_path, val_path = split_dir / "train.csv", split_dir / "val.csv"
    if recreate or not (train_path.is_file() and val_path.is_file()):
        LOGGER.info("Creating identity-disjoint train/validation split")
        create_and_save_split(
            configured_path(config, "train_csv"),
            split_dir,
            val_fraction=float(config["validation"]["val_fraction"]),
            seed=int(config["seed"]),
            min_val_images=2,
            min_val_cameras=2,
        )
    train_frame, val_frame = pd.read_csv(train_path), pd.read_csv(val_path)
    overlap = set(train_frame.vehicle_id).intersection(val_frame.vehicle_id)
    if overlap:
        raise RuntimeError(f"Identity leakage in saved split: {list(overlap)[:5]}")
    return train_frame, val_frame


def _csv_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _resolve_training_split(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve the legacy holdout or an explicitly selected verified K-fold."""

    manifest_path = getattr(args, "kfold_manifest", None)
    fold_index = getattr(args, "fold_index", None)
    if (manifest_path is None) != (fold_index is None):
        raise ValueError("--kfold-manifest and --fold-index must be provided together")
    if manifest_path is not None:
        if bool(getattr(args, "recreate_split", False)):
            raise ValueError("--recreate-split cannot be combined with --kfold-manifest")
        return resolve_kfold_partition(manifest_path, fold_index)

    train_frame, val_frame = _ensure_split(
        config, bool(getattr(args, "recreate_split", False))
    )
    split_dir = configured_path(config, "splits_dir")
    provenance = {
        "protocol": "identity_disjoint_holdout",
        "seed": int(config["seed"]),
        "requested_val_fraction": float(config["validation"]["val_fraction"]),
        "train_csv": _csv_provenance(split_dir / "train.csv"),
        "val_csv": _csv_provenance(split_dir / "val.csv"),
    }
    return train_frame, val_frame, provenance


def _check_resume_split(
    resume_state: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    """Reject accidental continuation of a training state on another fold."""

    saved = resume_state.get("data_split")
    if saved is None:
        return  # Legacy training state, created before split provenance existed.
    if not isinstance(saved, Mapping):
        raise ValueError("resume checkpoint data_split provenance is malformed")
    if split_provenance_fingerprint(saved) != split_provenance_fingerprint(selected):
        raise ValueError(
            "resume checkpoint was created with a different data split; select "
            "the matching K-fold manifest/fold index or holdout files"
        )


def _make_loaders(
    config: dict[str, Any], train_frame: pd.DataFrame, val_frame: pd.DataFrame
) -> tuple[
    DataLoader,
    DataLoader,
    CameraAwarePKBatchSampler,
    VehicleDataset,
    DataLoader | None,
]:
    image_size = config["model"]["input_size"]
    data_config = config["data"]
    resize_mode = str(data_config.get("resize_mode", "direct"))
    train_transform = build_train_transform(
        image_size,
        crop_scale=tuple(float(x) for x in data_config["train_crop_scale"]),
        crop_ratio=tuple(float(x) for x in data_config["train_crop_ratio"]),
        horizontal_flip_probability=float(
            data_config["horizontal_flip_probability"]
        ),
        random_erasing_probability=float(
            data_config["random_erasing_probability"]
        ),
        resize_mode=resize_mode,
    )
    eval_transform = build_eval_transform(image_size, resize_mode=resize_mode)
    images_dir = configured_path(config, "images_dir")
    padding = float(data_config["bbox_padding"])
    train_dataset = VehicleDataset(
        train_frame,
        images_dir,
        transform=train_transform,
        train=True,
        bbox_padding=padding,
        require_camera_id=True,
    )
    val_dataset = VehicleDataset(
        val_frame,
        images_dir,
        transform=eval_transform,
        train=False,
        bbox_padding=padding,
        require_camera_id=True,
    )
    training = config["training"]
    hard_mining = normalize_hard_identity_mining_config(
        training.get("hard_identity_mining")
    )
    sampler_arguments = {
        "identities_per_batch": int(training["identities_per_batch"]),
        "instances_per_identity": int(training["instances_per_identity"]),
        "seed": int(config["seed"]),
    }
    if bool(hard_mining["enabled"]):
        sampler: CameraAwarePKBatchSampler = HardIdentityMiningPKBatchSampler(
            train_dataset.pids,
            train_dataset.camera_ids,
            hard_fraction=float(hard_mining["hard_fraction"]),
            **sampler_arguments,
        )
    else:
        sampler = CameraAwarePKBatchSampler(
            train_dataset.pids,
            train_dataset.camera_ids,
            **sampler_arguments,
        )
    workers = int(data_config["num_workers"])
    common = {
        "num_workers": workers,
        "pin_memory": bool(data_config["pin_memory"] and torch.cuda.is_available()),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        common["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["validation_batch_size"]),
        shuffle=False,
        generator=generator,
        **common,
    )
    mining_loader: DataLoader | None = None
    if bool(hard_mining["enabled"]):
        mining_dataset = VehicleDataset(
            train_frame,
            images_dir,
            transform=eval_transform,
            train=True,
            bbox_padding=padding,
            pid_to_label=train_dataset.pid_to_label,
            require_camera_id=True,
        )
        mining_options = {
            "num_workers": workers,
            "pin_memory": common["pin_memory"],
            # Avoid retaining a second worker pool alongside the augmented
            # training loader for the whole experiment.
            "persistent_workers": False,
            "worker_init_fn": seed_worker,
        }
        if workers > 0:
            mining_options["prefetch_factor"] = 2
        mining_loader = DataLoader(
            mining_dataset,
            batch_size=int(hard_mining["embedding_batch_size"]),
            shuffle=False,
            generator=torch.Generator().manual_seed(int(config["seed"]) + 1),
            **mining_options,
        )
    return train_loader, val_loader, sampler, train_dataset, mining_loader


def _make_optimizer_scheduler(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    config: dict[str, Any],
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    training = config["training"]
    shared_head_parameters = list(model.projection.parameters()) + list(
        model.bnneck.parameters()
    )
    if "lr_local_head" in training:
        # Keep the established projection/BN head on lr_head while allowing
        # the newly initialized spatial branch to learn at its own rate.
        parameter_groups = [
            {
                "params": model.backbone.parameters(),
                "lr": float(training["lr_backbone"]),
            },
            {
                "params": shared_head_parameters,
                "lr": float(training["lr_head"]),
            },
            {
                "params": model.pool.parameters(),
                "lr": float(training["lr_local_head"]),
            },
            {
                "params": criterion.parameters(),
                "lr": float(training.get("lr_classifier", training["lr_head"])),
            },
        ]
    else:
        # This branch intentionally retains the original three groups and
        # their exact order so old configs and resume checkpoints are
        # optimizer-state compatible.
        head_parameters = shared_head_parameters + list(model.pool.parameters())
        parameter_groups = [
            {
                "params": model.backbone.parameters(),
                "lr": float(training["lr_backbone"]),
            },
            {"params": head_parameters, "lr": float(training["lr_head"])},
            {
                "params": criterion.parameters(),
                "lr": float(training.get("lr_classifier", training["lr_head"])),
            },
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training["weight_decay"]),
    )
    warmup = int(training["warmup_epochs"])
    minimum = float(training["min_lr_ratio"])

    def multiplier(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return max(minimum, float(epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        progress = min(1.0, max(0.0, progress))
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _optimizer_learning_rates(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any]
) -> dict[str, float]:
    """Return stable history fields for the configured optimizer layout."""

    has_local_group = "lr_local_head" in config["training"]
    expected_groups = 4 if has_local_group else 3
    if len(optimizer.param_groups) != expected_groups:
        raise RuntimeError(
            "optimizer group layout does not match training.lr_local_head: "
            f"expected {expected_groups}, got {len(optimizer.param_groups)}"
        )
    rates = {
        "lr_backbone": float(optimizer.param_groups[0]["lr"]),
        "lr_head": float(optimizer.param_groups[1]["lr"]),
    }
    if has_local_group:
        rates["lr_local_head"] = float(optimizer.param_groups[2]["lr"])
    rates["lr_classifier"] = float(
        optimizer.param_groups[3 if has_local_group else 2]["lr"]
    )
    return rates


def _make_criterion(
    config: dict[str, Any], num_classes: int, *, device: torch.device
) -> ReIDLoss:
    """Construct the configured classification/metric objective."""

    training = config["training"]
    xbm = normalize_cross_batch_memory_config(
        training.get("cross_batch_memory")
    )
    multi_similarity = normalize_multi_similarity_config(
        training.get("multi_similarity")
    )
    return ReIDLoss(
        int(config["model"]["embedding_dim"]),
        int(num_classes),
        arcface_scale=float(training["arcface_scale"]),
        arcface_margin=float(training["arcface_margin"]),
        arcface_subcenters=int(training.get("arcface_subcenters", 1)),
        label_smoothing=float(training["label_smoothing"]),
        triplet_margin=float(training["triplet_margin"]),
        triplet_margin_mode=str(training.get("triplet_margin_mode", "fixed")),
        triplet_positive_mining=str(
            training.get("triplet_positive_mining", "all")
        ),
        metric_loss=str(training.get("metric_loss", "triplet")),
        multi_similarity_alpha=float(multi_similarity["alpha"]),
        multi_similarity_beta=float(multi_similarity["beta"]),
        multi_similarity_base=float(multi_similarity["base"]),
        multi_similarity_epsilon=float(multi_similarity["epsilon"]),
        classification_weight=float(training["classification_weight"]),
        triplet_weight=float(training["triplet_weight"]),
        cross_batch_memory_capacity=(
            int(xbm["capacity"]) if bool(xbm["enabled"]) else 0
        ),
    ).to(device)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: ReIDLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp: bool,
    channels_last: bool,
    gradient_clip_norm: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    criterion.train()
    # XBM is deliberately scoped to one epoch.  This bounds feature staleness
    # and makes a checkpoint resumed at an epoch boundary exactly reproducible.
    criterion.reset_cross_batch_memory()
    totals = {"loss": 0.0, "classification_loss": 0.0, "metric_loss": 0.0}
    processed = 0
    progress = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True, dtype=torch.long)
        camera_ids = batch["camera_id"].to(
            device, non_blocking=True, dtype=torch.long
        )
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp):
            output = model(images, return_dict=True)
            details = criterion(
                output,
                labels,
                camera_ids=camera_ids,
                return_details=True,
            )
            loss = details["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                max_norm=gradient_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.shape[0])
        processed += batch_size
        for key in totals:
            totals[key] += float(details[key].detach()) * batch_size
        progress.set_postfix(loss=f"{totals['loss'] / processed:.4f}")
    if processed == 0:
        raise RuntimeError("No training batches were processed")
    averaged = {key: value / processed for key, value in totals.items()}
    # Preserve old consumers while new logs use the objective-neutral name.
    averaged["triplet_loss"] = averaged["metric_loss"]
    return averaged


def _refresh_hard_identity_map(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: HardIdentityMiningPKBatchSampler,
    vehicle_ids: list[Any],
    config: Mapping[str, bool | int | float],
    *,
    epoch: int,
    device: torch.device,
    amp: bool,
    channels_last: bool,
) -> dict[str, Any]:
    """Refresh the sampler's offline neighbor map from current train features."""

    started = time.perf_counter()
    embeddings, _ = extract_embeddings(
        model,
        loader,
        device=device,
        amp=amp,
        tta_horizontal_flip=bool(config["tta_horizontal_flip"]),
        channels_last=channels_last,
    )
    neighbor_map, report = build_identity_neighbor_map(
        embeddings,
        vehicle_ids,
        neighbors_per_identity=int(config["neighbors_per_identity"]),
    )
    sampler.set_hard_neighbors(neighbor_map)
    if sampler.neighbor_fingerprint != report["fingerprint"]:
        raise RuntimeError("hard-identity neighbor-map fingerprint mismatch")
    return {
        **report,
        "refresh_epoch": int(epoch) + 1,
        "refresh_seconds": time.perf_counter() - started,
        "tta_horizontal_flip": bool(config["tta_horizontal_flip"]),
    }


def _training_state(
    *,
    epoch: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_map: float,
    history: list[dict[str, Any]],
    data_split: dict[str, Any] | None = None,
    sampler: CameraAwarePKBatchSampler | None = None,
    hard_identity_mining_config: Mapping[str, Any] | None = None,
    hard_identity_mining_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = {
        "epoch": int(epoch),
        "model_state": cpu_state_dict(model),
        "criterion_state": cpu_state_dict(criterion),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_map": float(best_map),
        "history": history,
        "data_split": dict(data_split or {}),
        "hard_identity_mining_config": dict(hard_identity_mining_config or {}),
        "hard_identity_mining_report": dict(hard_identity_mining_report or {}),
    }
    if sampler is not None:
        state["sampler_state"] = sampler.state_dict()
    return state


def _initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    allow_partial_init: bool = False,
) -> dict[str, Any]:
    """Load initialization weights and report the exact loading contract.

    ``global_local`` is the sole intentional architecture expansion supported
    implicitly: an average-pooling checkpoint contains every shared parameter
    and omits only the new neutral local branch.  Its model-level guarded loader
    validates that invariant before changing any weights.  All other cases keep
    the historical strict/explicitly-partial behavior.
    """

    if not isinstance(checkpoint, Mapping) or "model_state" not in checkpoint:
        raise ValueError("initialization checkpoint must contain 'model_state'")
    state = checkpoint["model_state"]
    if not isinstance(state, Mapping):
        raise ValueError("initialization checkpoint model_state must be a mapping")

    architecture = checkpoint.get("model")
    if isinstance(architecture, Mapping):
        source_pooling: str | None = str(
            architecture.get("pooling", "avg")
        ).lower()
    elif architecture is None:
        # Legacy training-state files did not store architecture metadata.  An
        # avg pool owns no ``pool.*`` tensors, while GeM/global-local do.
        source_pooling = (
            "avg"
            if not any(str(name).startswith("pool.") for name in state)
            else None
        )
    else:
        source_pooling = None

    target_pooling = str(getattr(model, "pooling", "")).lower()
    if target_pooling == "global_local" and source_pooling == "avg":
        guarded_loader = getattr(model, "warm_start_from_avg_state_dict", None)
        if not callable(guarded_loader):
            raise TypeError(
                "global_local model does not expose its guarded avg warm-start"
            )
        guarded_loader(state)
        return {
            "mode": "avg_to_global_local_strict",
            "source_pooling": "avg",
            "target_pooling": "global_local",
            "missing_keys": sorted(set(model.state_dict()).difference(state)),
            "unexpected_keys": [],
        }

    incompatible = model.load_state_dict(state, strict=not allow_partial_init)
    return {
        "mode": "partial" if allow_partial_init else "strict",
        "source_pooling": source_pooling,
        "target_pooling": target_pooling or None,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    seed_everything(int(config["seed"]), deterministic=True)
    device = select_device(args.device)
    LOGGER.info("Training on %s", device)
    train_frame, val_frame, split_provenance = _resolve_training_split(config, args)
    train_loader, val_loader, sampler, train_dataset, mining_loader = _make_loaders(
        config, train_frame, val_frame
    )
    hard_mining = normalize_hard_identity_mining_config(
        config["training"].get("hard_identity_mining")
    )
    LOGGER.info(
        "Split: %d train images/%d IDs, %d val images/%d IDs; PK batch=%d",
        len(train_frame),
        train_frame.vehicle_id.nunique(),
        len(val_frame),
        val_frame.vehicle_id.nunique(),
        sampler.batch_size,
    )
    if bool(hard_mining["enabled"]):
        LOGGER.info(
            "Offline hard-ID mining: fraction=%.3f, neighbors=%d, warmup=%d, "
            "refresh_interval=%d",
            float(hard_mining["hard_fraction"]),
            int(hard_mining["neighbors_per_identity"]),
            int(hard_mining["warmup_epochs"]),
            int(hard_mining["refresh_interval"]),
        )

    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if not isinstance(resume_state, Mapping):
            raise ValueError("--resume must contain a training-state mapping")
        _check_resume_split(resume_state, split_provenance)
    init_checkpoint = None
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=True
        )
        if not isinstance(init_checkpoint, dict) or "model_state" not in init_checkpoint:
            raise ValueError(
                "--init-checkpoint must contain a mapping with 'model_state'"
            )
    model = create_model_from_config(
        config, pretrained=resume_state is None and init_checkpoint is None
    ).to(device)
    initialization_report = None
    if init_checkpoint is not None:
        initialization_report = _initialize_model_from_checkpoint(
            model,
            init_checkpoint,
            allow_partial_init=bool(getattr(args, "allow_partial_init", False)),
        )
        if initialization_report["mode"] == "partial":
            LOGGER.info(
                "Partially initialized model from %s (missing=%s, unexpected=%s)",
                args.init_checkpoint,
                initialization_report["missing_keys"],
                initialization_report["unexpected_keys"],
            )
        elif initialization_report["mode"] == "avg_to_global_local_strict":
            LOGGER.info(
                "Safely warm-started global_local model from avg checkpoint %s "
                "(neutral local keys=%s)",
                args.init_checkpoint,
                initialization_report["missing_keys"],
            )
        else:
            LOGGER.info("Initialized model weights from %s", args.init_checkpoint)
    channels_last = bool(config["model"].get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    num_classes = len(train_dataset.pid_to_label)
    training = config["training"]
    criterion = _make_criterion(config, num_classes, device=device)
    epochs = int(training["epochs"])
    optimizer, scheduler = _make_optimizer_scheduler(model, criterion, config, epochs)
    use_amp = bool(training["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_map, history = 0, -1.0, []
    hard_mining_last_report: dict[str, Any] = {}
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        criterion.load_state_dict(resume_state["criterion_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        scaler.load_state_dict(resume_state["scaler_state"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_map = float(resume_state["best_map"])
        history = list(resume_state.get("history", []))
        saved_hard_config = resume_state.get("hard_identity_mining_config")
        if saved_hard_config and dict(saved_hard_config) != hard_mining:
            raise ValueError(
                "resume checkpoint hard-identity mining configuration does not "
                "match the current config"
            )
        sampler_state = resume_state.get("sampler_state")
        if sampler_state is not None:
            if not isinstance(sampler_state, Mapping):
                raise ValueError("resume checkpoint sampler_state is malformed")
            sampler.load_state_dict(sampler_state)
        saved_hard_report = resume_state.get("hard_identity_mining_report", {})
        if not isinstance(saved_hard_report, Mapping):
            raise ValueError(
                "resume checkpoint hard_identity_mining_report is malformed"
            )
        hard_mining_last_report = dict(saved_hard_report)
        LOGGER.info("Resuming from epoch %d", start_epoch + 1)

    weights_dir = configured_path(config, "weights_dir")
    outputs_dir = configured_path(config, "outputs_dir")
    weights_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last_training.pt"
    started = time.perf_counter()
    best_report: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        hard_mining_refreshed = False
        if bool(hard_mining["enabled"]) and epoch >= int(
            hard_mining["warmup_epochs"]
        ):
            if not isinstance(sampler, HardIdentityMiningPKBatchSampler):
                raise RuntimeError("hard-identity mining requires its dedicated sampler")
            if mining_loader is None:
                raise RuntimeError("hard-identity mining evaluation loader is missing")
            scheduled_refresh = (
                epoch - int(hard_mining["warmup_epochs"])
            ) % int(hard_mining["refresh_interval"]) == 0
            # A legacy resume may not contain sampler state.  Refreshing from
            # the resumed model is deterministic and safer than silently
            # reverting to random PK sampling.
            if scheduled_refresh or not sampler.has_hard_neighbors:
                hard_mining_last_report = _refresh_hard_identity_map(
                    model,
                    mining_loader,
                    sampler,
                    train_dataset.pids,
                    hard_mining,
                    epoch=epoch,
                    device=device,
                    amp=use_amp,
                    channels_last=channels_last,
                )
                hard_mining_refreshed = True
                LOGGER.info(
                    "Hard-ID map refreshed for epoch %d: %d IDs, top-1 cosine "
                    "mean=%.4f, fingerprint=%s",
                    epoch + 1,
                    int(hard_mining_last_report["identity_count"]),
                    float(hard_mining_last_report["mean_top1_similarity"]),
                    str(hard_mining_last_report["fingerprint"])[:12],
                )
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp=use_amp,
            channels_last=channels_last,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            max_batches=args.max_train_batches,
        )
        val_embeddings, _ = extract_embeddings(
            model,
            val_loader,
            device=device,
            amp=use_amp,
            tta_horizontal_flip=False,
            channels_last=channels_last,
        )
        report, _ = evaluate_validation(
            val_embeddings,
            val_frame,
            unknown_fraction=float(config["validation"]["open_set_unknown_fraction"]),
            seed=int(config["validation"]["open_set_seed"]),
            fallback_temperature=float(
                config["inference"]["calibration_temperature"]
            ),
        )
        hard_sampling_report = (
            sampler.sampling_report()
            if isinstance(sampler, HardIdentityMiningPKBatchSampler)
            else {
                "active": False,
                "batches": 0,
                "target_hard_group_size": 0,
                "achieved_hard_group_fraction": 0.0,
                "neighbor_fingerprint": None,
            }
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in report["retrieval"].items()},
            "val_f1": report["open_set"]["f1"],
            "val_tnr": report["open_set"]["tnr"],
            "val_pr_auc": report["open_set"]["pr_auc"],
            "refusal_threshold": report["open_set"]["threshold"],
            **_optimizer_learning_rates(optimizer, config),
            "hard_mining_active": bool(hard_sampling_report["active"]),
            "hard_mining_refreshed": hard_mining_refreshed,
            "hard_mining_hard_group_fraction": float(
                hard_sampling_report["achieved_hard_group_fraction"]
            ),
            "hard_mining_map_fingerprint": hard_sampling_report[
                "neighbor_fingerprint"
            ],
            "hard_mining_mean_top1_similarity": hard_mining_last_report.get(
                "mean_top1_similarity"
            ),
            "hard_mining_refresh_seconds": (
                float(hard_mining_last_report["refresh_seconds"])
                if hard_mining_refreshed
                else 0.0
            ),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(outputs_dir / "training_history.csv", index=False)
        save_json(
            {
                "best_mAP": max(best_map, row["val_mAP"]),
                "hard_identity_mining": hard_mining,
                "history": history,
            },
            outputs_dir / "training_log.json",
        )

        if float(row["val_mAP"]) > best_map:
            best_map = float(row["val_mAP"])
            best_report = report
            save_inference_checkpoint(
                best_path,
                model,
                input_size=config["model"]["input_size"],
                bbox_padding=float(config["data"]["bbox_padding"]),
                refusal_threshold=float(report["open_set"]["threshold"]),
                resize_mode=str(config["data"].get("resize_mode", "direct")),
                tta_horizontal_flip=False,
                calibration=report["confidence_calibration"],
                metrics=report,
                source={
                    "epoch": epoch + 1,
                    "seed": int(config["seed"]),
                    "pretraining": config["model"]["backbone"],
                    "config": str(config["_config_path"]),
                    "initialization_checkpoint": (
                        str(args.init_checkpoint)
                        if args.init_checkpoint is not None
                        else None
                    ),
                    "initialization_mode": (
                        initialization_report["mode"]
                        if initialization_report is not None
                        else None
                    ),
                    "pooling": str(config["model"].get("pooling", "avg")),
                    "triplet_margin_mode": str(
                        training.get("triplet_margin_mode", "fixed")
                    ),
                    "triplet_positive_mining": str(
                        training.get("triplet_positive_mining", "all")
                    ),
                    "metric_loss": str(training.get("metric_loss", "triplet")),
                    "multi_similarity": normalize_multi_similarity_config(
                        training.get("multi_similarity")
                    ),
                    "arcface_subcenters": int(
                        training.get("arcface_subcenters", 1)
                    ),
                    "cross_batch_memory": normalize_cross_batch_memory_config(
                        training.get("cross_batch_memory")
                    ),
                    "hard_identity_mining": {
                        "config": hard_mining,
                        "latest_map": hard_mining_last_report,
                        "epoch_sampling": hard_sampling_report,
                    },
                    "data_split": split_provenance,
                },
            )
            save_json(report, outputs_dir / "best_validation_metrics.json")
            LOGGER.info("Saved new best checkpoint: mAP=%.4f", best_map)
        if bool(training.get("save_last", True)):
            atomic_torch_save(
                _training_state(
                    epoch=epoch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_map=best_map,
                    history=history,
                    data_split=split_provenance,
                    sampler=sampler,
                    hard_identity_mining_config=hard_mining,
                    hard_identity_mining_report=hard_mining_last_report,
                ),
                last_path,
            )
        LOGGER.info(
            "Epoch %d/%d loss=%.4f mAP=%.4f R1=%.4f F1=%.4f TNR=%.4f (%.1fs)",
            epoch + 1,
            epochs,
            train_metrics["loss"],
            report["retrieval"]["mAP"],
            report["retrieval"]["rank1"],
            report["open_set"]["f1"],
            report["open_set"]["tnr"],
            row["seconds"],
        )
    summary = {
        "best_mAP": best_map,
        "best_checkpoint": str(best_path),
        "epochs_completed": len(history),
        "wall_seconds": time.perf_counter() - started,
        "best_report": best_report,
        "data_split": split_provenance,
        "hard_identity_mining": {
            "config": hard_mining,
            "latest_map": hard_mining_last_report,
        },
    }
    save_json(summary, outputs_dir / "training_summary.json")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="initialize model weights only; optimizer/loss/schedule start fresh",
    )
    parser.add_argument(
        "--allow-partial-init",
        action="store_true",
        help="allow missing/unexpected keys when using --init-checkpoint",
    )
    parser.add_argument("--recreate-split", action="store_true")
    parser.add_argument(
        "--kfold-manifest",
        type=Path,
        default=None,
        help="verified src.kfold manifest to use instead of the default holdout",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="zero-based fold from --kfold-manifest",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="development smoke-test only; do not use for final training",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    configure_logging(args.verbose)
    summary = run_training(args)
    LOGGER.info("Training complete: best mAP %.4f", summary["best_mAP"])


if __name__ == "__main__":
    main()
