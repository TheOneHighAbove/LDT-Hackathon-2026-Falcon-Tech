"""Extract the locked loss-branch descriptor for a labelled split.

The output is an aligned NPZ suitable for train-only pair-verifier training.
It uses the split-trained checkpoint, never the full-data consolidation model,
so validation identities remain unseen by every downstream learned reranker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.probe_osnet_loss_branch_fusion import _extract, _model, _weighted_concat
from scripts.train_osnet_target import (
    CROP_CACHE,
    DEVICE,
    _cache_annotations,
    _eval_transform,
    _prepare_crop_cache,
)
from src.data import VehicleDataset
from src.engine import load_inference_checkpoint
from src.expert_fusion import _torch_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/osnet_target_loss_branches.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations)
    expected_ids = frame.image_id.astype(str).to_numpy(dtype=np.str_)
    _prepare_crop_cache(frame)

    conv_model, checkpoint = load_inference_checkpoint(
        Path("weights/best.pt"), device=DEVICE
    )
    conv_model = conv_model.to(memory_format=torch.channels_last)
    conv = _torch_embeddings(
        conv_model,
        frame,
        images_dir=Path("dataset/images"),
        preprocessing=checkpoint["preprocessing"],
        batch_size=64,
        workers=4,
        device=DEVICE,
        channels_last=True,
    )
    del conv_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

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
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )
    model = _model(args.checkpoint)
    identity, metric, _ = _extract(model, loader)
    specialized = _weighted_concat((identity, metric), (0.75, 0.25))
    embedding = _weighted_concat((conv, specialized), (0.20, 0.80))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=expected_ids,
        embeddings=embedding.astype(np.float16),
    )
    print(
        f"saved {args.output}: {embedding.shape}; "
        "conv=0.20, identity-branch=0.75 within OSNet",
        flush=True,
    )


if __name__ == "__main__":
    main()
