"""Tune branch energy and conservative fusion for the specialized OSNet.

Selection uses three fixed tune protocols; the last two protocols are opened
once for confirmation.  The resulting descriptor is still a plain weighted
concatenation, so it adds no search-time model and remains strict-streaming.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.probe_osnet_finetune_fusion import _evaluate
from scripts.probe_pair_verifier import CONFIRM_SEEDS, _normalize
from scripts.train_osnet_target import (
    CROP_CACHE,
    DEVICE,
    _cache_annotations,
    _eval_transform,
    _prepare_crop_cache,
)
from src.data import VehicleDataset
from src.osnet_trainable import load_vehicle_osnet_onnx


SOURCE = Path("weights/osnet_target_finetuned_hard_smoothap.pt")
CANDIDATE = Path("weights/osnet_target_loss_branches.pt")
REPORT = Path("outputs/expert_fusion/osnet_loss_branch_fusion.json")
OUTPUT = Path("outputs/expert_fusion/osnet_loss_branch_ensemble_val_embeddings.npy")


def _weighted_concat(arrays, weights):
    active = [
        np.sqrt(float(weight)) * _normalize(array)
        for array, weight in zip(arrays, weights, strict=True)
        if weight > 0
    ]
    if not active:
        raise ValueError("at least one fusion weight must be positive")
    return _normalize(np.concatenate(active, axis=1))


@torch.inference_mode()
def _extract(model, loader):
    model.eval()
    first, second, legacy = [], [], []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            branches = model.forward_branches(images)
        first.append(F.normalize(branches[0].float(), dim=1).cpu().numpy())
        second.append(F.normalize(branches[1].float(), dim=1).cpu().numpy())
        legacy.append(
            F.normalize(torch.cat(branches, dim=1).float(), dim=1).cpu().numpy()
        )
    return tuple(np.concatenate(values) for values in (first, second, legacy))


def _model(path):
    model = load_vehicle_osnet_onnx(
        "weights/osnet_ain_x1_0_vehicle_reid.onnx"
    ).to(DEVICE)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def main() -> None:
    if not CANDIDATE.is_file():
        raise FileNotFoundError(f"train the specialized candidate first: {CANDIDATE}")
    frame = pd.read_csv("splits/val.csv")
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
    source_model = _model(SOURCE)
    *_, source = _extract(source_model, loader)
    del source_model
    candidate_model = _model(CANDIDATE)
    identity_branch, metric_branch, _ = _extract(candidate_model, loader)
    del candidate_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    with np.load(
        "outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz",
        allow_pickle=False,
    ) as archive:
        conv = archive["embeddings"].astype(np.float32)

    tune_seeds, confirmation_seeds = CONFIRM_SEEDS[:3], CONFIRM_SEEDS[3:]
    grid = []
    candidates = {}
    for identity_share in (0.0, 0.10, 0.25, 0.50, 0.75):
        specialized = _weighted_concat(
            (identity_branch, metric_branch),
            (identity_share, 1.0 - identity_share),
        )
        for conv_weight in (0.15, 0.20, 0.25, 0.30):
            remaining = 1.0 - conv_weight
            for specialized_share in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
                weights = (
                    conv_weight,
                    remaining * (1.0 - specialized_share),
                    remaining * specialized_share,
                )
                embedding = _weighted_concat((conv, source, specialized), weights)
                metrics, _ = _evaluate(frame, embedding, tune_seeds)
                key = (identity_share, conv_weight, specialized_share)
                candidates[key] = embedding
                grid.append(
                    {
                        "identity_branch_share": identity_share,
                        "conv_weight": conv_weight,
                        "specialized_share_within_osnet": specialized_share,
                        "weights_conv_source_specialized": weights,
                        "tune": metrics,
                    }
                )
    selected = max(
        grid,
        key=lambda row: (
            row["tune"]["AP10"],
            row["tune"]["mAP"],
            row["tune"]["rank1"],
        ),
    )
    key = (
        selected["identity_branch_share"],
        selected["conv_weight"],
        selected["specialized_share_within_osnet"],
    )
    embedding = candidates[key]
    confirmation, rows = _evaluate(frame, embedding, confirmation_seeds)
    baseline_embedding = _weighted_concat((conv, source), (0.25, 0.75))
    baseline, baseline_rows = _evaluate(
        frame, baseline_embedding, confirmation_seeds
    )
    result = {
        "selection_rule": "maximize AP@10 on seeds 101,211,307",
        "selected": selected,
        "confirmation": confirmation,
        "baseline_confirmation": baseline,
        "delta": {
            name: float(confirmation[name]) - float(baseline[name])
            for name in confirmation
        },
        "confirmation_rows": rows,
        "baseline_rows": baseline_rows,
        "top_grid": sorted(
            grid,
            key=lambda row: (row["tune"]["AP10"], row["tune"]["mAP"]),
            reverse=True,
        )[:30],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.save(OUTPUT, embedding.astype(np.float32))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
