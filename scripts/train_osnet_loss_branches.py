"""Loss-branch-specialized target fine-tuning for the compact vehicle OSNet.

The original public OSNet contains two independent 256-D heads.  The standard
target fine-tune sends every loss through their 512-D concatenation, which
encourages the heads to learn correlated representations.  This experiment
implements the metadata-free part of MBR/Loss Branch Specialization:

* branch 0 receives only sub-center ArcFace identity supervision;
* branch 1 receives only cross-camera metric, SupCon and Smooth-AP losses;
* inference concatenates equally normalized branches into one 512-D vector.

Validation identities are never used for fitting.  Camera IDs are used only to
sample/define valid train positives and to reproduce the official junk mask.
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

from scripts.probe_pair_verifier import CONFIRM_SEEDS, _normalize
from scripts.probe_train_fitted_verifier import _colors
from scripts.train_osnet_target import (
    DEVICE,
    _evaluate,
    _loaders,
    _multi_positive_loss,
    _smooth_ap_loss,
)
from src.losses import ArcMarginProduct, BatchHardTripletLoss
from src.osnet_trainable import load_vehicle_osnet_onnx
from src.sampler import HardIdentityMiningPKBatchSampler, build_identity_neighbor_map


RUN_SEED = int(os.environ.get("OSNET_LBS_SEED", "12647"))
EPOCHS = int(os.environ.get("OSNET_LBS_EPOCHS", "6"))
SOURCE = Path(
    os.environ.get(
        "OSNET_LBS_INIT",
        "weights/osnet_target_finetuned_hard_smoothap.pt",
    )
)
OUTPUT = Path("weights/osnet_target_loss_branches.pt")
CACHE = Path("outputs/expert_fusion/cache/osnet_target_loss_branches_val.npz")
REPORT = Path("outputs/expert_fusion/osnet_target_loss_branches.json")


def _balanced_embedding(branches: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Concatenate equal-energy branches into a normalized descriptor."""

    if len(branches) != 2:
        raise ValueError("loss-branch OSNet requires exactly two descriptors")
    normalized = [F.normalize(value.float(), dim=1) for value in branches]
    return F.normalize(torch.cat(normalized, dim=1), dim=1)


@torch.inference_mode()
def _extract_balanced(model, loader) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
            dtype=torch.float16,
        ):
            branches = model.forward_branches(images)
        chunks.append(_balanced_embedding(branches).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _evaluation_context(val_frame: pd.DataFrame):
    with np.load(
        "outputs/expert_fusion/cache/convnext_val_e25580fb7c33dd97.npz",
        allow_pickle=False,
    ) as archive:
        conv = _normalize(archive["embeddings"])
    with np.load(
        "outputs/expert_fusion/cache/convnext_val_parts_3x3.npz",
        allow_pickle=False,
    ) as archive:
        parts = archive["parts"].astype(np.float32)
    colors = _colors(val_frame, "val")
    return conv, colors, parts


def _load_source(model) -> dict:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"initialization checkpoint does not exist: {SOURCE}")
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if "model_state" not in payload:
        raise ValueError(f"checkpoint has no model_state: {SOURCE}")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def main() -> None:
    torch.manual_seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.backends.cudnn.benchmark = True

    train_frame = pd.read_csv("splits/train.csv")
    val_frame = pd.read_csv("splits/val.csv")
    train_dataset, sampler, train_loader, val_loader, train_eval_loader = _loaders(
        train_frame, val_frame
    )
    if not isinstance(sampler, HardIdentityMiningPKBatchSampler):
        raise RuntimeError(
            "loss-branch training requires OSNET_HARD_MINING=1 so difficult "
            "same-model identities occur in each PK batch"
        )
    if train_eval_loader is None:
        raise RuntimeError("hard-mining evaluation loader was not created")

    model = load_vehicle_osnet_onnx(
        "weights/osnet_ain_x1_0_vehicle_reid.onnx"
    ).to(DEVICE)
    source_payload = _load_source(model)
    classifier = ArcMarginProduct(
        256,
        len(train_dataset.pid_to_label),
        scale=24.0,
        margin=0.25,
        num_subcenters=2,
    ).to(DEVICE)
    triplet = BatchHardTripletLoss(
        margin=0.20,
        margin_mode="soft",
        metric="cosine",
        positive_mining="cross_camera_only",
    ).to(DEVICE)

    head_parameters = list(model.fc.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    body_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_parameters, "lr": 8e-6},
            {"params": model.fc[0].parameters(), "lr": 8e-5},
            {"params": model.fc[1].parameters(), "lr": 8e-5},
            {"params": classifier.parameters(), "lr": 4e-4},
        ],
        weight_decay=2e-4,
    )
    total_steps = max(1, EPOCHS * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.05
        + 0.95 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    conv, colors, parts = _evaluation_context(val_frame)
    tune_seeds, confirmation_seeds = CONFIRM_SEEDS[:3], CONFIRM_SEEDS[3:]
    initial_embeddings = _extract_balanced(model, val_loader)
    initial_tune, _ = _evaluate(
        initial_embeddings, val_frame, conv, colors, parts, tune_seeds
    )
    initial_confirmation, _ = _evaluate(
        initial_embeddings, val_frame, conv, colors, parts, confirmation_seeds
    )
    print(
        json.dumps(
            {
                "initial_tune": initial_tune,
                "initial_confirmation": initial_confirmation,
            }
        ),
        flush=True,
    )

    best_tune = float(initial_tune["mAP"])
    best_state = None
    best_epoch = 0
    history = []
    mining_history = []
    for epoch in range(EPOCHS):
        mining_embeddings = _extract_balanced(model, train_eval_loader)
        neighbors, mining_report = build_identity_neighbor_map(
            mining_embeddings,
            train_frame.vehicle_id.to_numpy(),
            neighbors_per_identity=32,
        )
        sampler.set_hard_neighbors(neighbors)
        sampler.set_epoch(epoch)
        mining_history.append({"epoch": epoch + 1, **mining_report})

        model.train()
        classifier.train()
        totals = {
            "loss": [],
            "classification": [],
            "triplet": [],
            "supcon": [],
            "smooth_ap": [],
        }
        started = time.perf_counter()
        for batch in train_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            cameras = batch["camera_id"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type == "cuda",
                dtype=torch.float16,
            ):
                identity_branch, metric_branch = model.forward_branches(images)
                classification_loss = F.cross_entropy(
                    classifier(identity_branch, labels),
                    labels,
                    label_smoothing=0.05,
                )
                triplet_loss = triplet(metric_branch, labels, cameras)
                supcon_loss = _multi_positive_loss(
                    metric_branch, labels, cameras, temperature=0.08
                )
                smooth_ap_loss = _smooth_ap_loss(
                    metric_branch, labels, cameras, temperature=0.05
                )
                loss = (
                    0.35 * classification_loss
                    + triplet_loss
                    + 0.35 * supcon_loss
                    + 0.75 * smooth_ap_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()), 5.0
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            for name, value in (
                ("loss", loss),
                ("classification", classification_loss),
                ("triplet", triplet_loss),
                ("supcon", supcon_loss),
                ("smooth_ap", smooth_ap_loss),
            ):
                totals[name].append(float(value.detach()))

        embeddings = _extract_balanced(model, val_loader)
        tune, _ = _evaluate(embeddings, val_frame, conv, colors, parts, tune_seeds)
        row = {
            "epoch": epoch + 1,
            "seconds": time.perf_counter() - started,
            **{name: float(np.mean(values)) for name, values in totals.items()},
            **{f"tune_{key}": value for key, value in tune.items()},
            "hard_sampling": sampler.sampling_report(),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if float(tune["mAP"]) > best_tune:
            best_tune = float(tune["mAP"])
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    retained = False
    if best_state is not None:
        model.load_state_dict(best_state)
    final_embeddings = _extract_balanced(model, val_loader)
    tune, tune_rows = _evaluate(
        final_embeddings, val_frame, conv, colors, parts, tune_seeds
    )
    confirmation, confirmation_rows = _evaluate(
        final_embeddings, val_frame, conv, colors, parts, confirmation_seeds
    )
    if best_state is not None:
        retained = float(confirmation["mAP"]) > float(initial_confirmation["mAP"])

    result = {
        "design": "OSNet two-head loss branch specialization (CE / cross-camera metric)",
        "compliance": {
            "validation_identity_training": False,
            "camera_at_inference": False,
            "other_queries_at_inference": False,
            "test_pseudo_labels": False,
        },
        "seed": RUN_SEED,
        "epochs": EPOCHS,
        "source_checkpoint": str(SOURCE),
        "source_result": source_payload.get("result"),
        "selected_epoch": best_epoch,
        "retained": retained,
        "initial_tune": initial_tune,
        "initial_confirmation": initial_confirmation,
        "tune": tune,
        "confirmation": confirmation,
        "confirmation_delta_vs_initial": {
            key: float(confirmation[key]) - float(initial_confirmation[key])
            for key in confirmation
        },
        "tune_rows": tune_rows,
        "confirmation_rows": confirmation_rows,
        "history": history,
        "hard_mining": mining_history,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if best_state is not None:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "result": result,
                "balanced_branches": True,
                "source_onnx": "weights/osnet_ain_x1_0_vehicle_reid.onnx",
            },
            OUTPUT,
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE,
            image_ids=val_frame.image_id.astype(str).to_numpy(dtype=np.str_),
            embeddings=final_embeddings.astype(np.float16),
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
