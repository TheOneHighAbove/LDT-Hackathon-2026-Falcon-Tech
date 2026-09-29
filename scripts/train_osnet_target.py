"""Target-domain fine-tuning of the strongest (OSNet) retrieval branch."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from scripts.probe_pair_verifier import CONFIRM_SEEDS, _metrics_from_order, _normalize, _protocol
from scripts.probe_train_fitted_verifier import _colors, _fused, _train_parts
from src.data import IMAGENET_MEAN, IMAGENET_STD, VehicleDataset, crop_vehicle
from src.losses import ArcMarginProduct, BatchHardTripletLoss
from src.osnet_trainable import load_vehicle_osnet_onnx
from src.sampler import (
    CameraAwarePKBatchSampler,
    HardIdentityMiningPKBatchSampler,
    build_identity_neighbor_map,
)


DEVICE = torch.device("cuda")
EPOCHS = int(os.environ.get("OSNET_EPOCHS", "6"))
CROP_CACHE = Path("outputs/osnet_crop_cache_v1")
RUN_SEED = int(os.environ.get("OSNET_RUN_SEED", "8201"))
RUN_TAG = os.environ.get("OSNET_RUN_TAG", "").strip()
SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
INIT_CHECKPOINT = os.environ.get("OSNET_INIT_CHECKPOINT", "").strip()
HARD_MINING = os.environ.get("OSNET_HARD_MINING", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
SMOOTH_AP = os.environ.get("OSNET_SMOOTH_AP", "0").strip().lower() in {
    "1", "true", "yes", "on",
}


def _train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (208, 208), scale=(0.88, 1.0), ratio=(0.85, 1.18),
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.12, hue=0.03),
            transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR, fill=(124, 116, 104)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.12), value="random"),
        ]
    )


def _eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((208, 208), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _prepare_crop_cache(*frames):
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("image_id")
    manifest = CROP_CACHE / "manifest.json"
    if manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("count") == len(combined):
            return
    CROP_CACHE.mkdir(parents=True, exist_ok=True)

    def prepare(row):
        destination = CROP_CACHE / f"{row.image_id}.jpg"
        if destination.is_file():
            return
        with Image.open(Path("dataset/images") / f"{row.image_id}.jpg") as source:
            crop = crop_vehicle(source, (row.x, row.y, row.w, row.h), padding=0.05)
            crop = crop.resize((208, 208), Image.Resampling.BICUBIC)
            crop.save(destination, format="JPEG", quality=95, subsampling=0)

    print(f"preparing {len(combined)} reusable OSNet crops", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(prepare, combined.itertuples(index=False)))
    manifest.write_text(json.dumps({"count": len(combined)}) + "\n", encoding="utf-8")


def _cache_annotations(frame):
    cached = frame.copy()
    cached.loc[:, ["x", "y"]] = 0
    cached.loc[:, ["w", "h"]] = 208
    return cached


def _loaders(train_frame, val_frame):
    _prepare_crop_cache(train_frame, val_frame)
    train_dataset = VehicleDataset(
        _cache_annotations(train_frame), CROP_CACHE, transform=_train_transform(), train=True,
        bbox_padding=0.0, require_camera_id=True,
    )
    val_dataset = VehicleDataset(
        _cache_annotations(val_frame), CROP_CACHE, transform=_eval_transform(), train=False,
        bbox_padding=0.0, require_camera_id=True,
    )
    sampler_class = HardIdentityMiningPKBatchSampler if HARD_MINING else CameraAwarePKBatchSampler
    sampler = sampler_class(
        train_dataset.pids, train_dataset.camera_ids,
        identities_per_batch=16, instances_per_identity=4, seed=RUN_SEED,
        **({"hard_fraction": 0.75} if HARD_MINING else {}),
    )
    common = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, **common)
    train_eval_loader = None
    if HARD_MINING:
        train_eval_dataset = VehicleDataset(
            _cache_annotations(train_frame), CROP_CACHE, transform=_eval_transform(), train=False,
            bbox_padding=0.0, require_camera_id=True,
        )
        train_eval_loader = DataLoader(
            train_eval_dataset, batch_size=96, shuffle=False, **common
        )
    return train_dataset, sampler, train_loader, val_loader, train_eval_loader


def _multi_positive_loss(embeddings, labels, cameras, temperature=0.09):
    vectors = F.normalize(embeddings.float(), dim=1)
    logits = vectors @ vectors.T / temperature
    self_mask = torch.eye(len(vectors), device=vectors.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    cross_camera = positive & cameras[:, None].ne(cameras[None, :])
    # PK sampling guarantees cross-camera positives; fallback keeps the loss
    # defined if a future dataset has only one camera for an identity.
    positive = torch.where(cross_camera.any(dim=1, keepdim=True), cross_camera, positive)
    denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    valid = positive.any(dim=1)
    per_anchor = -(log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return per_anchor[valid].mean()


def _smooth_ap_loss(embeddings, labels, cameras, temperature=0.05):
    """Differentiable AP over valid cross-camera positives in a PK batch."""

    vectors = F.normalize(embeddings.float(), dim=1)
    similarity = vectors @ vectors.T
    same_identity = labels[:, None].eq(labels[None, :])
    same_camera = cameras[:, None].eq(cameras[None, :])
    self_mask = torch.eye(len(vectors), device=vectors.device, dtype=torch.bool)
    positives = same_identity & ~same_camera
    # The retrieval protocol removes same-ID/same-camera images entirely.
    candidates = ~self_mask & ~(same_identity & same_camera)
    values = []
    for anchor in range(len(vectors)):
        positive_scores = similarity[anchor, positives[anchor]]
        if positive_scores.numel() == 0:
            continue
        candidate_scores = similarity[anchor, candidates[anchor]]
        # For every positive p, approximate how many candidates outrank p.
        comparisons = torch.sigmoid(
            (candidate_scores[None, :] - positive_scores[:, None]) / temperature
        )
        total_rank = 1.0 + comparisons.sum(dim=1) - 0.5
        positive_comparisons = torch.sigmoid(
            (positive_scores[None, :] - positive_scores[:, None]) / temperature
        )
        positive_rank = 1.0 + positive_comparisons.sum(dim=1) - 0.5
        values.append((positive_rank / total_rank).mean())
    if not values:
        return embeddings.sum() * 0.0
    return 1.0 - torch.stack(values).mean()


def _group_topk_loss(
    embeddings, labels, cameras, shortlist_k=3, margin=0.03, temperature=0.05
):
    """Push every cross-camera positive above the top-k negative boundary.

    Same-identity/same-camera samples are junk in the official protocol and
    therefore take part in neither side of the comparison.  The k-th hardest
    negative is a less noisy boundary than the single hardest look-alike.
    """

    vectors = F.normalize(embeddings.float(), dim=1)
    similarity = vectors @ vectors.T
    same_identity = labels[:, None].eq(labels[None, :])
    same_camera = cameras[:, None].eq(cameras[None, :])
    positives = same_identity & ~same_camera
    negatives = ~same_identity
    values = []
    for anchor in range(len(vectors)):
        positive_scores = similarity[anchor, positives[anchor]]
        negative_scores = similarity[anchor, negatives[anchor]]
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            continue
        count = min(int(shortlist_k), int(negative_scores.numel()))
        boundary = torch.topk(negative_scores, k=count).values[-1]
        values.append(
            F.softplus(
                (boundary + margin - positive_scores) / temperature
            ).mean()
            * temperature
        )
    if not values:
        return embeddings.sum() * 0.0
    return torch.stack(values).mean()


def _extract(model, loader):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                embeddings = model(images)
            chunks.append(F.normalize(embeddings.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def _evaluate(osnet, val_frame, conv, colors, parts, seeds):
    fused = _fused(conv, osnet)
    views = np.full((len(val_frame), 8), 1.0 / 8.0, dtype=np.float32)
    rows = []
    for seed in seeds:
        protocol = _protocol(seed, val_frame, fused, conv, osnet, colors, parts, views)
        rows.append({"seed": seed, **_metrics_from_order(protocol, None, 0.0)})
    return {
        key: float(np.mean([row[key] for row in rows])) for key in ("mAP", "rank1", "rank5")
    }, rows


def main():
    torch.manual_seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.backends.cudnn.benchmark = True
    train_frame = pd.read_csv("splits/train.csv")
    val_frame = pd.read_csv("splits/val.csv")
    train_dataset, sampler, train_loader, val_loader, train_eval_loader = _loaders(
        train_frame, val_frame
    )

    model = load_vehicle_osnet_onnx("weights/osnet_ain_x1_0_vehicle_reid.onnx").to(DEVICE)
    if INIT_CHECKPOINT:
        checkpoint = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        print(f"continued from {INIT_CHECKPOINT}", flush=True)
    classifier = ArcMarginProduct(
        512, len(train_dataset.pid_to_label), scale=24.0, margin=0.25, num_subcenters=2
    ).to(DEVICE)
    triplet = BatchHardTripletLoss(
        margin=0.20, margin_mode="soft", metric="cosine", positive_mining="cross_camera_only"
    ).to(DEVICE)
    head_parameters = list(model.fc.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    body_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_parameters, "lr": 1.5e-5},
            {"params": head_parameters, "lr": 7.5e-5},
            {"params": classifier.parameters(), "lr": 4e-4},
        ],
        weight_decay=2e-4,
    )
    schedule_epochs = int(os.environ.get("OSNET_SCHEDULE_EPOCHS", str(EPOCHS)))
    total_steps = schedule_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * step / max(total_steps, 1))),
    )
    scaler = torch.amp.GradScaler("cuda")

    conv = _normalize(np.load(
        "outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz", allow_pickle=False
    )["embeddings"])
    original_osnet = _normalize(np.load(
        "outputs/expert_fusion/cache/osnet_openvino_val_4aaad3e5db648618.npz", allow_pickle=False
    )["embeddings"])
    colors = _colors(val_frame, "val")
    parts = _train_parts(val_frame) if False else np.load(
        "outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", allow_pickle=False
    )["parts"].astype(np.float32)
    tune_seeds, confirm_seeds = CONFIRM_SEEDS[:3], CONFIRM_SEEDS[3:]
    baseline, _ = _evaluate(original_osnet, val_frame, conv, colors, parts, tune_seeds)
    print(f"baseline tune={baseline}", flush=True)
    if INIT_CHECKPOINT:
        initial_embeddings = _extract(model, val_loader)
        initial_tune, _ = _evaluate(
            initial_embeddings, val_frame, conv, colors, parts, tune_seeds
        )
        initial_confirmation, _ = _evaluate(
            initial_embeddings, val_frame, conv, colors, parts, confirm_seeds
        )
    else:
        initial_embeddings = original_osnet
        initial_tune = baseline
        initial_confirmation, _ = _evaluate(
            original_osnet, val_frame, conv, colors, parts, confirm_seeds
        )
    print(f"initial tune={initial_tune}", flush=True)

    best_map = initial_tune["mAP"]
    best_state = None
    history = []
    mining_history = []
    for epoch in range(EPOCHS):
        if HARD_MINING:
            assert isinstance(sampler, HardIdentityMiningPKBatchSampler)
            assert train_eval_loader is not None
            train_embeddings = _extract(model, train_eval_loader)
            neighbor_map, mining_report = build_identity_neighbor_map(
                train_embeddings,
                train_frame.vehicle_id.to_numpy(),
                neighbors_per_identity=32,
            )
            sampler.set_hard_neighbors(neighbor_map)
            mining_history.append({"epoch": epoch + 1, **mining_report})
            print(
                f"hard-ID mining epoch={epoch + 1} "
                f"mean_top1={mining_report['mean_top1_similarity']:.4f}",
                flush=True,
            )
        sampler.set_epoch(epoch)
        model.train()
        classifier.train()
        losses = []
        started = time.perf_counter()
        for batch in train_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            cameras = batch["camera_id"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                embeddings = model(images)
                metric_loss = triplet(embeddings, labels, cameras)
                multi_loss = _multi_positive_loss(embeddings, labels, cameras)
                class_loss = F.cross_entropy(classifier(embeddings, labels), labels, label_smoothing=0.05)
                ap_loss = (
                    _smooth_ap_loss(embeddings, labels, cameras)
                    if SMOOTH_AP
                    else embeddings.sum() * 0.0
                )
                loss = (
                    metric_loss
                    + 0.35 * multi_loss
                    + 0.05 * class_loss
                    + (0.75 * ap_loss if SMOOTH_AP else 0.0)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP deliberately skips an optimizer step after a non-finite
            # gradient.  Keep the LR schedule aligned with real updates.
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            losses.append(float(loss.detach()))
        embeddings = _extract(model, val_loader)
        tune, _ = _evaluate(embeddings, val_frame, conv, colors, parts, tune_seeds)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "seconds": time.perf_counter() - started,
            **{f"tune_{key}": value for key, value in tune.items()},
        }
        if HARD_MINING:
            row["hard_sampling"] = sampler.sampling_report()
        history.append(row)
        print(json.dumps(row), flush=True)
        if tune["mAP"] > best_map:
            best_map = tune["mAP"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        result = {
            "seed": RUN_SEED,
            "retained": False,
            "reason": "no fine-tuned epoch beat its initialization on tune seeds",
            "baseline_tune": baseline,
            "initial_tune": initial_tune,
            "history": history,
            "hard_mining": mining_history,
        }
    else:
        model.load_state_dict(best_state)
        embeddings = _extract(model, val_loader)
        tune, tune_rows = _evaluate(embeddings, val_frame, conv, colors, parts, tune_seeds)
        confirmation, confirm_rows = _evaluate(
            embeddings, val_frame, conv, colors, parts, confirm_seeds
        )
        baseline_confirmation, _ = _evaluate(
            original_osnet, val_frame, conv, colors, parts, confirm_seeds
        )
        result = {
            "seed": RUN_SEED,
            "retained": confirmation["mAP"] > initial_confirmation["mAP"],
            "selected_epoch": int(max(history, key=lambda row: row["tune_mAP"])["epoch"]),
            "init_checkpoint": INIT_CHECKPOINT or None,
            "hard_identity_mining": HARD_MINING,
            "smooth_ap": SMOOTH_AP,
            "baseline_tune": baseline,
            "initial_tune": initial_tune,
            "tune": tune,
            "baseline_confirmation": baseline_confirmation,
            "initial_confirmation": initial_confirmation,
            "confirmation": confirmation,
            "confirmation_delta": {
                key: confirmation[key] - baseline_confirmation[key] for key in confirmation
            },
            "confirmation_delta_vs_initial": {
                key: confirmation[key] - initial_confirmation[key] for key in confirmation
            },
            "tune_rows": tune_rows,
            "confirmation_rows": confirm_rows,
            "history": history,
            "hard_mining": mining_history,
        }
        torch.save(
            {"model_state": best_state, "result": result, "source_onnx": "weights/osnet_ain_x1_0_vehicle_reid.onnx"},
            f"weights/osnet_target_finetuned{SUFFIX}.pt",
        )
        np.savez_compressed(
            f"outputs/expert_fusion/cache/osnet_target_finetuned_val{SUFFIX}.npz",
            image_ids=val_frame.image_id.astype(str).to_numpy(dtype=np.str_),
            embeddings=embeddings.astype(np.float16),
        )
    path = Path(f"outputs/expert_fusion/osnet_target_finetune{SUFFIX}.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
