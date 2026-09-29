"""Extract adapted high-resolution DINO CLS embeddings for an annotation CSV."""

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
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--flip-tta", action="store_true")
    parser.add_argument(
        "--checkpoint", default="weights/dinov2_vehicle_metric_compact.pt"
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations)
    transform = transforms.Compose([
        transforms.Resize((280, 280), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = VehicleDataset(frame, "dataset/images", transform=transform, bbox_padding=0.0)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    model = _model()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=not payload.get("compact", False))
    model.eval()
    values = []
    for number, batch in enumerate(loader, 1):
        image = batch["image"].cuda(non_blocking=True)
        inputs = torch.cat((image, image.flip(3)), dim=0) if args.flip_tta else image
        with torch.autocast("cuda", dtype=torch.float16):
            cls = F.normalize(model.forward_features(inputs)["x_norm_clstoken"].float(), dim=1)
        if args.flip_tta:
            size = len(image)
            cls = F.normalize(cls[:size] + cls[size:], dim=1)
        values.append(cls.cpu().numpy().astype(np.float16))
        if number % 100 == 0 or number == len(loader):
            print(f"DINO CLS {number}/{len(loader)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        embeddings=np.concatenate(values),
    )
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
