"""Target-domain metric adaptation of the last DINOv2-B blocks.

Training uses labelled train identities and cross-camera positives. Validation
is identity-disjoint and follows the strict one-query plus static-gallery
protocol.  No query-query operation is used by evaluation or intended
inference.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_osnet_target import (
    CROP_CACHE,
    _cache_annotations,
    _group_topk_loss,
    _multi_positive_loss,
    _prepare_crop_cache,
    _smooth_ap_loss,
)
from scripts.train_streaming_top25_verifier import _normalize
from src.data import IMAGENET_MEAN, IMAGENET_STD, VehicleDataset
from src.losses import ArcMarginProduct, BatchHardTripletLoss
from src.reranking import database_side_augmentation
from src.sampler import HardIdentityMiningPKBatchSampler, build_identity_neighbor_map


DEVICE = torch.device("cuda")
DINO_V2_SOURCE = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
EPOCHS = int(os.environ.get("DINO_EPOCHS", "2"))
BATCHES_PER_EPOCH = int(os.environ.get("DINO_BATCHES", "400"))
TRAINABLE_BLOCKS = int(os.environ.get("DINO_BLOCKS", "2"))
BODY_LR = float(os.environ.get("DINO_LR", "1e-5"))
INIT_CHECKPOINT = os.environ.get("DINO_INIT", "").strip()
IDENTITIES_PER_BATCH = int(os.environ.get("DINO_P", "4"))
INPUT_SIZE = int(os.environ.get("DINO_SIZE", "196"))
USE_ORIGINAL_IMAGES = os.environ.get("DINO_ORIGINAL", "0").lower() in {"1", "true", "yes"}
PART_MODE = os.environ.get("DINO_PARTS", "cls").strip().lower()
OUTPUT_CHECKPOINT = os.environ.get(
    "DINO_OUTPUT", "weights/dinov2_vehicle_metric.pt"
).strip()
OUTPUT_REPORT = os.environ.get(
    "DINO_REPORT", "outputs/expert_fusion/dinov2_vehicle_metric.json"
).strip()
OUTPUT_CACHE = os.environ.get(
    "DINO_CACHE", "outputs/expert_fusion/cache/dinov2_vehicle_metric_val.npz"
).strip()


def _transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                (INPUT_SIZE, INPUT_SIZE), scale=(0.82, 1.0), ratio=(0.82, 1.22),
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.18, 0.18, 0.14, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _loaders(train, val, initial_embedding):
    if USE_ORIGINAL_IMAGES:
        image_root = Path("dataset/images")
        train_annotations, val_annotations = train, val
        padding = 0.0
    else:
        _prepare_crop_cache(train, val)
        image_root = CROP_CACHE
        train_annotations, val_annotations = _cache_annotations(train), _cache_annotations(val)
        padding = 0.0
    train_data = VehicleDataset(
        train_annotations, image_root, transform=_transform(True), train=True,
        bbox_padding=padding, require_camera_id=True,
    )
    val_data = VehicleDataset(
        val_annotations, image_root, transform=_transform(False), train=False,
        bbox_padding=padding, require_camera_id=True,
    )
    sampler = HardIdentityMiningPKBatchSampler(
        train_data.pids,
        train_data.camera_ids,
        identities_per_batch=IDENTITIES_PER_BATCH,
        instances_per_identity=2,
        batches_per_epoch=BATCHES_PER_EPOCH,
        seed=92621,
        hard_fraction=0.75,
    )
    neighbors, report = build_identity_neighbor_map(
        initial_embedding, train.vehicle_id.to_numpy(), neighbors_per_identity=48
    )
    sampler.set_hard_neighbors(neighbors)
    train_loader = DataLoader(
        train_data, batch_sampler=sampler, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=16, shuffle=False, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    return train_data, sampler, train_loader, val_loader, report


def _model():
    model = torch.hub.load(
        DINO_V2_SOURCE, "dinov2_vitb14_reg", pretrained=True,
        trust_repo=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks[-TRAINABLE_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in model.norm.parameters():
        parameter.requires_grad_(True)
    return model.to(DEVICE)


def _forward(model, images):
    output = model.forward_features(images)
    cls = F.normalize(output["x_norm_clstoken"].float(), dim=1)
    if PART_MODE == "cls":
        return cls
    if PART_MODE != "cls_h2":
        raise ValueError(f"unsupported DINO_PARTS={PART_MODE!r}")
    tokens = output["x_norm_patchtokens"]
    side = int(round(tokens.shape[1] ** 0.5))
    spatial = tokens.reshape(len(tokens), side, side, tokens.shape[2])
    bands = [F.normalize(chunk.mean(dim=(1, 2)).float(), dim=1)
             for chunk in torch.tensor_split(spatial, 2, dim=1)]
    return F.normalize(torch.cat((cls, *bands), dim=1), dim=1)


@torch.inference_mode()
def _extract(model, loader, cls_only=False):
    model.eval()
    values = []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            images = batch["image"].to(DEVICE, non_blocking=True)
            if cls_only:
                output = model.forward_features(images)
                embedding = F.normalize(output["x_norm_clstoken"].float(), dim=1)
            else:
                embedding = _forward(model, images)
            values.append(embedding.cpu().numpy())
    return np.concatenate(values)


def _evaluate(frame, embedding, osnet, seeds):
    protocols = _protocols(frame, seeds, {"base": osnet})
    grid = []
    for weight in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
        fused = _normalize(np.concatenate((
            np.sqrt(1.0 - weight) * osnet,
            np.sqrt(weight) * embedding,
        ), axis=1)) if weight else osnet
        rows = []
        for protocol in protocols:
            query = fused[protocol["qi"]]
            gallery = database_side_augmentation(fused[protocol["gi"]], top_k=5, alpha=2.0)
            rows.append(_official_from_scores(query @ gallery.T, protocol["q"], protocol["g"]))
        grid.append({"dino_weight": weight, **_aggregate(rows)})
    return max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"])), grid


def main():
    torch.manual_seed(92621)
    np.random.seed(92621)
    torch.backends.cudnn.benchmark = True
    train, val = pd.read_csv("splits/train.csv"), pd.read_csv("splits/val.csv")
    with np.load("outputs/expert_fusion/cache/osnet_smoothap_train.npz", allow_pickle=False) as archive:
        train_osnet = _normalize(archive["embeddings"])
    val_osnet = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy", allow_pickle=False
    ))
    train_data, sampler, train_loader, val_loader, mining = _loaders(
        train, val, train_osnet
    )
    model = _model()
    if INIT_CHECKPOINT:
        payload = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        print(f"continued from {INIT_CHECKPOINT}", flush=True)
    embedding_dim = 768 if PART_MODE == "cls" else 768 * 3
    classifier = ArcMarginProduct(
        embedding_dim, len(train_data.pid_to_label), scale=24.0, margin=0.20, num_subcenters=2
    ).to(DEVICE)
    if INIT_CHECKPOINT:
        payload = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=False)
        if "classifier_state" in payload:
            try:
                classifier.load_state_dict(payload["classifier_state"])
            except RuntimeError:
                print("classifier shape changed; initialized a new ArcFace head", flush=True)
    triplet = BatchHardTripletLoss(
        margin=0.15, margin_mode="soft", metric="cosine",
        positive_mining="cross_camera_only",
    ).to(DEVICE)
    body = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": BODY_LR},
        {"params": classifier.parameters(), "lr": 3.0e-4},
    ], weight_decay=2.0e-4)
    total_steps = max(EPOCHS * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda")

    # When horizontal parts are an auxiliary training signal, select the
    # checkpoint on the deployable global CLS descriptor.  This prevents the
    # auxiliary head from optimizing a validation representation we do not use.
    cls_selection = PART_MODE != "cls"
    initial = _extract(model, val_loader, cls_only=cls_selection)
    initial_tune, _ = _evaluate(val, initial, val_osnet, TUNE_SEEDS)
    print(json.dumps({"initial_tune": initial_tune, "mining": mining}), flush=True)
    best = initial_tune
    best_state = copy.deepcopy(model.state_dict())
    best_embedding = initial
    history = []
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        classifier.train()
        losses = []
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, 1):
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            cameras = batch["camera_id"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                embedding = _forward(model, images)
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
                loss = metric + 0.30 * multi + 0.50 * smooth + 0.50 * boundary + 0.08 * classification
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(body, 3.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if batch_index % 100 == 0:
                print(f"epoch={epoch + 1} batch={batch_index}/{len(train_loader)} loss={np.mean(losses[-50:]):.4f}", flush=True)
        val_embedding = _extract(model, val_loader, cls_only=cls_selection)
        tune, _ = _evaluate(val, val_embedding, val_osnet, TUNE_SEEDS)
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
            best_state = copy.deepcopy(model.state_dict())
            best_embedding = val_embedding.copy()

    model.load_state_dict(best_state)
    confirmation, confirm_grid = _evaluate(val, best_embedding, val_osnet, CONFIRM_SEEDS)
    baseline = next(row for row in confirm_grid if row["dino_weight"] == 0.0)
    result = {
        "design": f"last-{TRAINABLE_BLOCKS}-block DINOv2-B metric adaptation; strict streaming evaluation",
        "trainable_blocks": TRAINABLE_BLOCKS,
        "body_lr": BODY_LR,
        "identities_per_batch": IDENTITIES_PER_BATCH,
        "input_size": INPUT_SIZE,
        "original_bbox_crops": USE_ORIGINAL_IMAGES,
        "part_mode": PART_MODE,
        "trainable_parameters": int(sum(p.numel() for p in body)),
        "initial_tune": initial_tune,
        "best_tune": best,
        "history": history,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {key: confirmation[key] - baseline[key] for key in ("mAP@10", "Rank-1", "Rank-5")},
        "confirmation_grid": confirm_grid,
    }
    Path(OUTPUT_REPORT).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({
        "model_state": best_state,
        "classifier_state": classifier.state_dict(),
        "result": result,
    }, OUTPUT_CHECKPOINT)
    np.savez_compressed(
        OUTPUT_CACHE,
        image_ids=val.image_id.astype(str).to_numpy(dtype=np.str_),
        embeddings=best_embedding.astype(np.float16),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
