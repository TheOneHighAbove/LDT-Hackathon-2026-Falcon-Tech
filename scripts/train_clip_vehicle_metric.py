"""Target-domain metric adaptation of OpenAI CLIP ViT-B/16 for vehicle ReID."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scripts.probe_clip_vehicle_retrieval import LOCAL_CHECKPOINT
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_dino_patch_matcher import _fuse, _load
from scripts.train_osnet_target import (
    CROP_CACHE,
    _cache_annotations,
    _group_topk_loss,
    _multi_positive_loss,
    _prepare_crop_cache,
    _smooth_ap_loss,
)
from scripts.train_streaming_top25_verifier import _normalize
from src.data import VehicleDataset
from src.losses import ArcMarginProduct, BatchHardTripletLoss
from src.reranking import database_side_augmentation
from src.sampler import HardIdentityMiningPKBatchSampler, build_identity_neighbor_map


DEVICE = torch.device("cuda")
EPOCHS = int(os.environ.get("CLIP_EPOCHS", "3"))
BATCHES_PER_EPOCH = int(os.environ.get("CLIP_BATCHES", "300"))
TRAINABLE_BLOCKS = int(os.environ.get("CLIP_BLOCKS", "3"))
BODY_LR = float(os.environ.get("CLIP_LR", "6e-6"))
INPUT_SIZE = int(os.environ.get("CLIP_SIZE", "224"))
INIT_CHECKPOINT = os.environ.get("CLIP_INIT", "").strip()
RUN_TAG = os.environ.get("CLIP_TAG", "").strip()
SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
CHECKPOINT = Path(f"weights/clip_vitb16_vehicle_metric{SUFFIX}.pt")
CACHE = Path(f"outputs/expert_fusion/cache/clip_vitb16_vehicle_metric_val{SUFFIX}.npz")
REPORT = Path(f"outputs/expert_fusion/clip_vitb16_vehicle_metric{SUFFIX}.json")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                (INPUT_SIZE, INPUT_SIZE), scale=(0.82, 1.0), ratio=(0.82, 1.22),
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.16, 0.16, 0.12, 0.025),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize(
            (INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def _visual():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(LOCAL_CHECKPOINT), device="cpu"
    )
    model = clip.visual
    del clip
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.transformer.resblocks[-TRAINABLE_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in model.ln_post.parameters():
        parameter.requires_grad_(True)
    model.proj.requires_grad_(True)
    model.set_grad_checkpointing(True)
    return model.to(DEVICE)


def _loaders(train, val, initial_embedding):
    _prepare_crop_cache(train, val)
    train_data = VehicleDataset(
        _cache_annotations(train), CROP_CACHE, transform=_transform(True), train=True,
        bbox_padding=0.0, require_camera_id=True,
    )
    val_data = VehicleDataset(
        _cache_annotations(val), CROP_CACHE, transform=_transform(False), train=False,
        bbox_padding=0.0, require_camera_id=True,
    )
    sampler = HardIdentityMiningPKBatchSampler(
        train_data.pids, train_data.camera_ids,
        identities_per_batch=8, instances_per_identity=2,
        batches_per_epoch=BATCHES_PER_EPOCH, seed=260923,
        hard_fraction=0.75,
    )
    neighbors, mining = build_identity_neighbor_map(
        initial_embedding, train.vehicle_id.to_numpy(), neighbors_per_identity=48
    )
    sampler.set_hard_neighbors(neighbors)
    train_loader = DataLoader(
        train_data, batch_sampler=sampler, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=48, shuffle=False, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    return train_data, sampler, train_loader, val_loader, mining


def _forward(model, image):
    return F.normalize(model(image).float(), dim=1)


@torch.inference_mode()
def _extract(model, loader):
    model.eval()
    output = []
    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            direct = _forward(model, image)
            flipped = _forward(model, image.flip(3))
            embedding = F.normalize(direct + flipped, dim=1)
        output.append(embedding.cpu().numpy())
    return np.concatenate(output)


def _evaluate(frame, clip_embedding, base, seeds):
    protocols = _protocols(frame, seeds, {"base": base})
    grid = []
    for dba_k in (3, 5):
        for weight in (0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20,
                       0.30, 0.40, 0.50, 0.70, 1.0):
            fused = base if weight == 0.0 else _normalize(np.concatenate((
                np.sqrt(1.0 - weight) * base,
                np.sqrt(weight) * clip_embedding,
            ), axis=1)) if weight < 1.0 else clip_embedding
            rows = []
            for protocol in protocols:
                gallery = database_side_augmentation(
                    fused[protocol["gi"]], top_k=dba_k, alpha=2.0
                )
                score = fused[protocol["qi"]] @ gallery.T
                rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
            grid.append({
                "clip_weight": weight, "dba_k": dba_k, **_aggregate(rows)
            })
    return max(
        grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])
    ), grid


def main():
    torch.manual_seed(260923)
    np.random.seed(260923)
    torch.backends.cudnn.benchmark = True
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    train_osnet = _normalize(_load(
        "outputs/expert_fusion/cache/osnet_smoothap_train.npz", train, "embeddings"
    ))
    val_osnet = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ))
    train_dino = _normalize(_load(
        "outputs/expert_fusion/cache/dinov2_vehicle_cls_train.npz", train, "embeddings"
    ))
    val_dino = _normalize(_load(
        "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz", val, "cls"
    ))
    train_base, val_base = _fuse(train_osnet, train_dino), _fuse(val_osnet, val_dino)
    train_data, sampler, train_loader, val_loader, mining = _loaders(
        train, val, train_base
    )
    model = _visual()
    initial_payload = None
    if INIT_CHECKPOINT:
        initial_payload = torch.load(
            INIT_CHECKPOINT, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial_payload["model_state"])
        print(f"continued CLIP visual from {INIT_CHECKPOINT}", flush=True)
    classifier = ArcMarginProduct(
        512, len(train_data.pid_to_label), scale=24.0, margin=0.20,
        num_subcenters=2,
    ).to(DEVICE)
    if initial_payload is not None and "classifier_state" in initial_payload:
        classifier.load_state_dict(initial_payload["classifier_state"])
    triplet = BatchHardTripletLoss(
        margin=0.15, margin_mode="soft", metric="cosine",
        positive_mining="cross_camera_only",
    ).to(DEVICE)
    body = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": BODY_LR},
        {"params": classifier.parameters(), "lr": 3e-4},
    ], weight_decay=2e-4)
    total_steps = max(EPOCHS * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda")

    initial = _extract(model, val_loader)
    initial_tune, _ = _evaluate(val, initial, val_base, TUNE_SEEDS)
    print(json.dumps({"initial_tune": initial_tune, "mining": mining}), flush=True)
    best, best_embedding = initial_tune, initial
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        classifier.train()
        losses = []
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            cameras = batch["camera_id"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                embedding = _forward(model, image)
                metric = triplet(embedding, labels, cameras)
                multi = _multi_positive_loss(embedding, labels, cameras)
                smooth = _smooth_ap_loss(embedding, labels, cameras, temperature=0.06)
                boundary = _group_topk_loss(
                    embedding, labels, cameras, shortlist_k=3,
                    margin=0.03, temperature=0.05,
                )
                classification = F.cross_entropy(
                    classifier(embedding, labels), labels, label_smoothing=0.05
                )
                loss = (
                    metric + 0.30 * multi + 0.55 * smooth
                    + 0.55 * boundary + 0.08 * classification
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(body, 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if batch_index % 100 == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_index}/{len(train_loader)} "
                    f"loss={np.mean(losses[-50:]):.4f}", flush=True,
                )
        embedding = _extract(model, val_loader)
        tune, _ = _evaluate(val, embedding, val_base, TUNE_SEEDS)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "seconds": time.perf_counter() - started,
            **tune,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if (tune["mAP@10"], tune["Rank-1"]) > (best["mAP@10"], best["Rank-1"]):
            best = tune
            best_embedding = embedding.copy()
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    confirmation, confirmation_grid = _evaluate(
        val, best_embedding, val_base, CONFIRM_SEEDS
    )
    selected_dba = best["dba_k"]
    baseline = next(
        row for row in confirmation_grid
        if row["clip_weight"] == 0.0 and row["dba_k"] == selected_dba
    )
    report = {
        "design": (
            f"last-{TRAINABLE_BLOCKS}-block OpenAI CLIP ViT-B/16 target metric "
            "adaptation; cross-camera hard PK; flip TTA"
        ),
        "trainable_blocks": TRAINABLE_BLOCKS,
        "body_lr": BODY_LR,
        "trainable_parameters": int(sum(p.numel() for p in body)),
        "initial_tune": initial_tune,
        "best_tune": best,
        "history": history,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "confirmation_grid": confirmation_grid,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "model_state": best_state,
        "classifier_state": classifier.state_dict(),
        "report": report,
    }, CHECKPOINT)
    np.savez_compressed(
        CACHE,
        image_ids=val.image_id.astype(str).to_numpy(dtype=np.str_),
        embeddings=best_embedding.astype(np.float16),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
