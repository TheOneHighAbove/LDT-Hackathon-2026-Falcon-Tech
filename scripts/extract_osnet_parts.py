"""Extract compact spatial OSNet descriptors for learned cross-image matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.train_osnet_target import (
    CROP_CACHE,
    DEVICE,
    _cache_annotations,
    _eval_transform,
    _prepare_crop_cache,
)
from src.data import VehicleDataset
from src.osnet_trainable import load_vehicle_osnet_onnx


@torch.inference_mode()
def _extract(model, loader, grid: int) -> np.ndarray:
    model.eval()
    batches = []
    for index, batch in enumerate(loader):
        feature_map = model.featuremaps(batch["image"].to(DEVICE, non_blocking=True))
        parts = F.adaptive_avg_pool2d(feature_map, (grid, grid)).flatten(2).transpose(1, 2)
        parts = F.normalize(parts.float(), dim=2)
        batches.append(parts.cpu().numpy().astype(np.float16))
        if index % 20 == 0:
            print(f"part batches {index + 1}/{len(loader)}", flush=True)
    return np.concatenate(batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations)
    _prepare_crop_cache(frame)
    dataset = VehicleDataset(
        _cache_annotations(frame), CROP_CACHE, transform=_eval_transform(), train=False,
        bbox_padding=0.0, require_camera_id=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )
    model = load_vehicle_osnet_onnx("weights/osnet_ain_x1_0_vehicle_reid.onnx").to(DEVICE)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    parts = _extract(model, loader, args.grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        parts=parts,
    )
    print(f"saved {parts.shape} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
