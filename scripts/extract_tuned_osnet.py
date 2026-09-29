"""Extract aligned embeddings from a target-tuned OSNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.train_osnet_target import (
    CROP_CACHE,
    DEVICE,
    _cache_annotations,
    _eval_transform,
    _extract,
    _prepare_crop_cache,
)
from src.data import VehicleDataset
from src.osnet_trainable import load_vehicle_osnet_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations)
    _prepare_crop_cache(frame)
    dataset = VehicleDataset(
        _cache_annotations(frame),
        CROP_CACHE,
        transform=_eval_transform(),
        train=False,
        bbox_padding=0.0,
        require_camera_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    model = load_vehicle_osnet_onnx("weights/osnet_ain_x1_0_vehicle_reid.onnx").to(DEVICE)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    embeddings = _extract(model, loader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        embeddings=embeddings.astype(np.float16),
    )
    print(f"saved {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
