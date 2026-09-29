"""Frozen OpenAI CLIP ViT-B/16 as an independent vehicle-ReID branch."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_dino_patch_matcher import _fuse, _load
from scripts.train_osnet_target import CROP_CACHE, _cache_annotations, _prepare_crop_cache
from scripts.train_streaming_top25_verifier import _normalize
from src.data import VehicleDataset
from src.reranking import database_side_augmentation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = Path("outputs/expert_fusion/cache/clip_vitb16_openai_val.npz")
REPORT = Path("outputs/expert_fusion/clip_vitb16_openai_frozen.json")
LOCAL_CHECKPOINT = Path(
    "C:/Users/roman/.cache/huggingface/hub/"
    "models--timm--vit_base_patch16_clip_224.openai/snapshots/"
    "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_model.safetensors"
)


def _model_and_transform():
    if not LOCAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"OpenAI CLIP checkpoint not found: {LOCAL_CHECKPOINT}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(LOCAL_CHECKPOINT), device=DEVICE
    )
    model.eval()
    return model, transform


@torch.inference_mode()
def _extract(frame):
    if CACHE.is_file():
        return _normalize(_load(CACHE, frame, "embeddings"))
    _prepare_crop_cache(frame)
    model, transform = _model_and_transform()
    dataset = VehicleDataset(
        _cache_annotations(frame), CROP_CACHE, transform=transform,
        train=False, bbox_padding=0.0, require_camera_id=True,
    )
    loader = DataLoader(
        dataset, batch_size=48, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    output = []
    for index, batch in enumerate(loader, start=1):
        image = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda",
                            dtype=torch.float16):
            direct = F.normalize(model.encode_image(image).float(), dim=1)
            flipped = F.normalize(model.encode_image(image.flip(3)).float(), dim=1)
            embedding = F.normalize(direct + flipped, dim=1)
        output.append(embedding.cpu().numpy())
        if index % 10 == 0:
            print(f"CLIP extraction batch {index}/{len(loader)}", flush=True)
    embedding = np.concatenate(output)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        image_ids=frame.image_id.astype(str).to_numpy(dtype=np.str_),
        embeddings=embedding.astype(np.float16),
    )
    return embedding


def _evaluate(frame, protocols, base, clip, weight, dba_k):
    if weight == 0.0:
        embedding = base
    elif weight == 1.0:
        embedding = clip
    else:
        embedding = _normalize(np.concatenate((
            np.sqrt(1.0 - weight) * base,
            np.sqrt(weight) * clip,
        ), axis=1))
    rows = []
    for protocol in protocols:
        gallery = embedding[protocol["gi"]]
        if dba_k:
            gallery = database_side_augmentation(gallery, top_k=dba_k, alpha=2.0)
        score = embedding[protocol["qi"]] @ gallery.T
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    frame = pd.read_csv("splits/val.csv")
    clip = _extract(frame)
    osnet = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ))
    dino = _normalize(_load(
        "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz",
        frame, "cls",
    ))
    base = _fuse(osnet, dino)
    tune = _protocols(frame, TUNE_SEEDS, {"base": base})
    confirm = _protocols(frame, CONFIRM_SEEDS, {"base": base})
    grid = []
    for dba_k in (0, 3, 5, 7):
        for weight in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
                       0.40, 0.50, 0.70, 1.0):
            grid.append({
                "clip_weight": weight,
                "dba_k": dba_k,
                **_evaluate(frame, tune, base, clip, weight, dba_k),
            })
    selected = max(
        grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
    )
    configuration = {
        "clip_weight": selected["clip_weight"], "dba_k": selected["dba_k"]
    }
    confirmation = _evaluate(
        frame, confirm, base, clip,
        configuration["clip_weight"], configuration["dba_k"],
    )
    # Isolate the CLIP contribution from the independently selected DBA size.
    baseline = _evaluate(frame, confirm, base, clip, 0.0, configuration["dba_k"])
    clip_only = _evaluate(frame, confirm, base, clip, 1.0, configuration["dba_k"])
    report = {
        "design": "frozen OpenAI CLIP ViT-B/16, flip TTA, identity-disjoint validation",
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "clip_only_confirmation": clip_only,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "grid": grid,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
