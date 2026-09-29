"""Extract the accepted LBS retrieval descriptor for train-ID hard mining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from scripts.probe_osnet_loss_branch_fusion import (
    CANDIDATE,
    _extract,
    _model,
    _weighted_concat,
)
from scripts.train_dino_patch_matcher import _load
from scripts.train_osnet_target import (
    CROP_CACHE,
    _cache_annotations,
    _eval_transform,
    _prepare_crop_cache,
)
from src.data import VehicleDataset


OUTPUT = Path(
    "outputs/expert_fusion/osnet_loss_branch_ensemble_train_embeddings.npy"
)


def main() -> None:
    if OUTPUT.is_file():
        print(f"embeddings already exist: {OUTPUT}", flush=True)
        return
    frame = pd.read_csv("splits/train.csv")
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
        batch_size=96,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    model = _model(CANDIDATE)
    identity, metric, _ = _extract(model, loader)
    specialized = _weighted_concat((identity, metric), (0.75, 0.25))
    conv = _load(
        "outputs/expert_fusion/cache/convnext_train_verifier.npz",
        frame,
        "embeddings",
    )
    embedding = _weighted_concat((conv, specialized), (0.20, 0.80))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, embedding.astype(np.float32), allow_pickle=False)
    print({"output": str(OUTPUT), "shape": embedding.shape}, flush=True)


if __name__ == "__main__":
    main()
