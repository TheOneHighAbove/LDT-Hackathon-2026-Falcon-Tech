"""Extract compact spatial tokens from the adapted DINOv2 vehicle model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scripts.train_dinov2_vehicle_metric import _model
from src.data import IMAGENET_MEAN, IMAGENET_STD, VehicleDataset


@torch.inference_mode()
def extract(frame: pd.DataFrame, checkpoint: str, batch_size: int,
            grid_size: int = 5) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((280, 280), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = VehicleDataset(frame, "dataset/images", transform=transform, bbox_padding=0.0)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    model = _model()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=not payload.get("compact", False))
    model.eval()
    result = []
    for number, batch in enumerate(loader, 1):
        image = batch["image"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = model.forward_features(image)["x_norm_patchtokens"]
        side = int(round(tokens.shape[1] ** 0.5))
        spatial = tokens.reshape(len(tokens), side, side, tokens.shape[2]).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(
            spatial.float(), (grid_size, grid_size)
        ).flatten(2).transpose(1, 2)
        result.append(F.normalize(pooled, dim=2).cpu().numpy().astype(np.float16))
        if number % 100 == 0 or number == len(loader):
            print(f"patch extraction {number}/{len(loader)}", flush=True)
    return np.concatenate(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="weights/dinov2_vehicle_metric_compact.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=5)
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations)
    tokens = extract(frame, args.checkpoint, args.batch_size, args.grid_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        tokens=tokens,
    )
    print(f"saved {args.output}: {tokens.shape}", flush=True)


if __name__ == "__main__":
    main()
