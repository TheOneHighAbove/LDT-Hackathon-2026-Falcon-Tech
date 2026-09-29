"""Extract compact 7x7 target-adapted CLIP tokens for train identities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.train_clip_vehicle_metric import (
    CROP_CACHE,
    _cache_annotations,
    _prepare_crop_cache,
    _transform,
    _visual,
)
from src.data import VehicleDataset


DEVICE = torch.device("cuda")
CHECKPOINT = Path(os.environ.get(
    "CLIP_TOKEN_CHECKPOINT", "weights/clip_vitb16_vehicle_metric_stage2.pt"
))
TOKENS = Path(os.environ.get(
    "CLIP_TRAIN_TOKENS",
    "outputs/expert_fusion/cache/clip_vehicle_stage2_train_tokens_7x7.npy",
))
IMAGE_IDS = Path(os.environ.get(
    "CLIP_TRAIN_TOKEN_IDS",
    "outputs/expert_fusion/cache/clip_vehicle_stage2_train_token_ids.npy",
))
FLIP_TTA = os.environ.get("CLIP_TOKEN_FLIP_TTA", "1").lower() in {
    "1", "true", "yes"
}


@torch.inference_mode()
def main():
    if TOKENS.is_file() and IMAGE_IDS.is_file():
        print(f"tokens already exist: {TOKENS}", flush=True)
        return
    frame = pd.read_csv("splits/train.csv")
    _prepare_crop_cache(frame)
    data = VehicleDataset(
        _cache_annotations(frame), CROP_CACHE, transform=_transform(False),
        train=False, bbox_padding=0.0, require_camera_id=True,
    )
    loader = DataLoader(
        data, batch_size=32, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    model = _visual()
    model.load_state_dict(torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )["model_state"])
    model.eval()
    flip_index = torch.arange(196, device=DEVICE).reshape(14, 14).flip(1).flatten()
    output = []
    for index, batch in enumerate(loader, start=1):
        image = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            direct = model.forward_intermediates(
                image, indices=1, normalize_intermediates=True,
                output_fmt="NLC", output_extra_tokens=False,
            )["image_intermediates"][0]
            token = direct.float()
            if FLIP_TTA:
                flipped = model.forward_intermediates(
                    image.flip(3), indices=1, normalize_intermediates=True,
                    output_fmt="NLC", output_extra_tokens=False,
                )["image_intermediates"][0][:, flip_index]
                token = (token + flipped.float()) * 0.5
            token = F.avg_pool2d(
                token.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2), 2
            ).permute(0, 2, 3, 1).flatten(1, 2)
            token = F.normalize(token, dim=2)
        output.append(token.half().cpu().numpy())
        if index % 25 == 0:
            print(f"train CLIP token extraction {index}/{len(loader)}", flush=True)
    token = np.concatenate(output)
    np.save(TOKENS, token)
    np.save(IMAGE_IDS, frame.image_id.astype(str).to_numpy(dtype=np.str_))
    print({
        "shape": token.shape, "bytes": TOKENS.stat().st_size,
        "flip_tta": FLIP_TTA,
    }, flush=True)


if __name__ == "__main__":
    main()
