"""Evaluate a checkpoint and finalize its open-set refusal calibration."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import configured_path, load_config, resolve_path
from .data import VehicleDataset, build_eval_transform
from .engine import (
    benchmark_embedding_model,
    checkpoint_postprocess,
    extract_embeddings,
    load_inference_checkpoint,
    save_inference_checkpoint,
)
from .kfold import resolve_kfold_partition, split_provenance_fingerprint
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
from .validation import evaluate_validation


def _csv_provenance(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _resolve_validation_split(
    config: dict, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the default holdout or one fully verified K-fold validation set."""

    manifest_path = getattr(args, "kfold_manifest", None)
    fold_index = getattr(args, "fold_index", None)
    if (manifest_path is None) != (fold_index is None):
        raise ValueError("--kfold-manifest and --fold-index must be provided together")
    if manifest_path is not None:
        _, val, provenance = resolve_kfold_partition(manifest_path, fold_index)
        return val, provenance

    split_dir = configured_path(config, "splits_dir")
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    if not val_path.is_file() or not train_path.is_file():
        create_and_save_split(
            configured_path(config, "train_csv"),
            split_dir,
            val_fraction=float(config["validation"]["val_fraction"]),
            seed=int(config["seed"]),
            min_val_images=2,
            min_val_cameras=2,
        )
    provenance: dict[str, object] = {
        "protocol": "identity_disjoint_holdout",
        "seed": int(config["seed"]),
        "requested_val_fraction": float(config["validation"]["val_fraction"]),
        "train_csv": _csv_provenance(train_path),
        "val_csv": _csv_provenance(val_path),
    }
    return pd.read_csv(val_path), provenance


def _check_checkpoint_split(
    checkpoint: Mapping[str, object], selected: Mapping[str, object]
) -> None:
    """Prevent evaluation on a fold that was part of the checkpoint's train set."""

    source = checkpoint.get("source")
    if not isinstance(source, Mapping):
        return
    trained = source.get("data_split")
    if not isinstance(trained, Mapping):
        return  # Legacy checkpoint: no auditable split provenance is available.
    if split_provenance_fingerprint(trained) != split_provenance_fingerprint(selected):
        raise ValueError(
            "validation data split does not match the checkpoint training split; "
            "select the same K-fold manifest/fold index or matching holdout CSVs"
        )


def run_validation(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    seed_everything(int(config["seed"]), deterministic=True)
    device = select_device(args.device)
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else configured_path(config, "weights_dir") / "best.pt"
    )
    model, checkpoint = load_inference_checkpoint(checkpoint_path, device=device)
    existing_postprocess = checkpoint_postprocess(checkpoint)
    if (
        not args.no_update_checkpoint
        and existing_postprocess is not None
        and bool(existing_postprocess["enabled"])
    ):
        raise ValueError(
            "This checkpoint uses gallery-conditioned DBA/QE calibration. "
            "Raw validation cannot overwrite its threshold; pass "
            "--no-update-checkpoint or run src.robust_open_set followed by "
            "src.finalize_checkpoint."
        )
    channels_last = bool(config["model"].get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    frame, split_provenance = _resolve_validation_split(config, args)
    _check_checkpoint_split(checkpoint, split_provenance)
    preprocessing = checkpoint["preprocessing"]
    resize_mode = str(preprocessing.get("resize_mode", "direct"))
    dataset = VehicleDataset(
        frame,
        configured_path(config, "images_dir"),
        transform=build_eval_transform(
            preprocessing["input_size"], resize_mode=resize_mode
        ),
        bbox_padding=float(preprocessing["bbox_padding"]),
        require_camera_id=True,
    )
    workers = int(config["data"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["validation_batch_size"]),
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
        tta_horizontal_flip=bool(config["inference"]["tta_horizontal_flip"]),
        channels_last=channels_last,
    )
    if metadata["image_id"].tolist() != frame.image_id.astype(str).tolist():
        raise RuntimeError("Validation DataLoader changed annotation order")
    report, details = evaluate_validation(
        embeddings,
        frame,
        unknown_fraction=float(config["validation"]["open_set_unknown_fraction"]),
        seed=int(config["validation"]["open_set_seed"]),
        fallback_temperature=float(config["inference"]["calibration_temperature"]),
    )
    sample = dataset[0]["image"].unsqueeze(0)
    if channels_last:
        sample = sample.contiguous(memory_format=torch.channels_last)
    report["performance"] = benchmark_embedding_model(
        model,
        sample,
        device=device,
        warmup=args.benchmark_warmup,
        iterations=args.benchmark_iterations,
        amp=device.type == "cuda",
        tta_horizontal_flip=bool(
            preprocessing.get(
                "tta_horizontal_flip", config["inference"]["tta_horizontal_flip"]
            )
        ),
    )
    report["performance"]["device"] = str(device)
    report["num_validation_embeddings"] = len(embeddings)
    report["data_split"] = split_provenance

    outputs_dir = (
        resolve_path(config, args.output_dir)
        if args.output_dir is not None
        else configured_path(config, "outputs_dir")
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)
    np.save(outputs_dir / "val_embeddings.npy", embeddings.astype(np.float32))
    np.savez_compressed(outputs_dir / "open_set_diagnostics.npz", **details)

    if not args.no_update_checkpoint:
        source = dict(checkpoint.get("source", {}))
        source["final_validation_tta"] = bool(
            config["inference"]["tta_horizontal_flip"]
        )
        source["validation_data_split"] = split_provenance
        save_inference_checkpoint(
            checkpoint_path,
            model,
            input_size=preprocessing["input_size"],
            bbox_padding=float(preprocessing["bbox_padding"]),
            refusal_threshold=float(report["open_set"]["threshold"]),
            resize_mode=resize_mode,
            tta_horizontal_flip=bool(
                config["inference"]["tta_horizontal_flip"]
            ),
            postprocess=existing_postprocess,
            calibration=report["confidence_calibration"],
            metrics=report,
            source=source,
        )
    report["checkpoint"] = {
        "path": str(Path(checkpoint_path).resolve()),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": Path(checkpoint_path).stat().st_size,
    }
    save_json(report, outputs_dir / "validation_metrics.json")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-update-checkpoint", action="store_true")
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
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    configure_logging(args.verbose)
    report = run_validation(args)
    LOGGER.info(
        "Validation: mAP=%.4f Rank-1=%.4f Rank-5=%.4f F1=%.4f TNR=%.4f threshold=%.4f",
        report["retrieval"]["mAP"],
        report["retrieval"]["rank1"],
        report["retrieval"]["rank5"],
        report["open_set"]["f1"],
        report["open_set"]["tnr"],
        report["open_set"]["threshold"],
    )


if __name__ == "__main__":
    main()
