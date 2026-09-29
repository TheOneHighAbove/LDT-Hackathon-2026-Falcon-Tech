"""Measure whether the top-50 token cross-matcher adds to the current final stack."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.probe_dino_verifier_streaming import (
    _patch_gallery_affinity,
    _predict_patch_plain,
    _prepare,
    _scores,
)
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from scripts.train_dino_patch_matcher import _load as _load_aligned
from scripts.train_dino_token_cross_top50 import (
    DEVICE,
    DinoTokenCrossMatcher,
    _base_protocol,
    _fuse,
    _normalize,
    _predict as _predict_token,
)
from scripts.train_streaming_top25_verifier import _local_for_protocol


def _propagate_patch_knn(score, data, config):
    affinity = data["gallery_sources"]["patch"]
    order = np.argsort(-affinity, axis=1, kind="stable")
    neighborhoods = []
    for index, row in enumerate(order[:, :config["neighbor_k"]]):
        valid = row[affinity[index, row] >= config["threshold"]]
        neighborhoods.append(np.concatenate(([index], valid)))
    original = score.copy()
    for index, neighborhood in enumerate(neighborhoods):
        support = original[:, neighborhood].max(axis=1)
        score[:, index] = (
            (1.0 - config["propagation"]) * original[:, index]
            + config["propagation"] * support
        )
    return score


def _evaluate(protocols, data, local, patch, token_data, token, config, token_weight):
    rows = []
    for protocol, item, local_value, patch_value, token_item, token_value in zip(
        protocols, data, local, patch, token_data, token, strict=True
    ):
        score = _scores(
            item,
            local_value,
            patch_value,
            config["verifier_weight"],
            config["part_source"],
            config["part_weight"],
            config["patch_weight"],
        )
        candidate = token_item["candidates"]
        base = np.take_along_axis(token_item["score"], candidate, axis=1)
        residual = token_value - base
        current = np.take_along_axis(score, candidate, axis=1)
        np.put_along_axis(score, candidate, current + token_weight * residual, axis=1)
        if config["group_source"] == "patch_knn":
            score = _propagate_patch_knn(score, item, config)
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    frame = pd.read_csv("splits/val.csv")
    osnet = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy",
        allow_pickle=False,
    ))
    with np.load(
        "outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz",
        allow_pickle=False,
    ) as archive:
        dino = _normalize(archive["cls"])
        parts = {name: _normalize(archive[name]) for name in ("mean", "h2", "h4")}
    fused = _fuse(osnet, dino)
    tokens = _load_aligned(
        "outputs/expert_fusion/cache/dinov2_vehicle_patches_val.npz", frame, "tokens"
    )
    conv_parts = _load(
        "outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", frame, "parts"
    )
    osnet_parts = _load(
        "outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz", frame, "parts"
    )

    verifier_checkpoint = torch.load(
        "weights/dino_top25_verifier.pt", map_location=DEVICE, weights_only=False
    )
    verifier = CrossImageTransformer(dim=96).to(DEVICE)
    verifier.load_state_dict(verifier_checkpoint["model_state"])
    verifier.eval()
    patch_model = joblib.load("weights/dino_patch_matcher.joblib")
    token_checkpoint = torch.load(
        "weights/dino_token_cross_top50.pt", map_location=DEVICE, weights_only=False
    )
    token_model = DinoTokenCrossMatcher().to(DEVICE)
    token_model.load_state_dict(token_checkpoint["model_state"])
    token_model.eval()

    tune = _protocols(frame, TUNE_SEEDS, {"base": fused})
    confirm = _protocols(frame, CONFIRM_SEEDS, {"base": fused})
    tune_data = [_prepare(p, fused, osnet, parts) for p in tune]
    confirm_data = [_prepare(p, fused, osnet, parts) for p in confirm]
    tune_token_data = [_base_protocol(frame, fused, seed) for seed in TUNE_SEEDS]
    confirm_token_data = [_base_protocol(frame, fused, seed) for seed in CONFIRM_SEEDS]

    tune_local = [
        _local_for_protocol(verifier, p, d, conv_parts, osnet_parts, osnet)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_local = [
        _local_for_protocol(verifier, p, d, conv_parts, osnet_parts, osnet)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    tune_patch = [
        _predict_patch_plain(patch_model, p, d, tokens, fused, osnet, dino)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_patch = [
        _predict_patch_plain(patch_model, p, d, tokens, fused, osnet, dino)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    tune_token = [
        _predict_token(token_model, value, tokens, fused) for value in tune_token_data
    ]
    confirm_token = [
        _predict_token(token_model, value, tokens, fused) for value in confirm_token_data
    ]
    for protocol, data in zip(tune + confirm, tune_data + confirm_data, strict=True):
        data["gallery_sources"]["patch"] = _patch_gallery_affinity(
            patch_model, protocol, tokens, fused, osnet, dino
        )

    previous = json.loads(Path(
        "outputs/expert_fusion/dino_verifier_streaming.json"
    ).read_text(encoding="utf-8"))
    keys = (
        "verifier_weight", "part_source", "part_weight", "patch_weight",
        "group_source", "neighbor_k", "threshold", "propagation",
    )
    config = {key: previous["selected_tune"][key] for key in keys}
    grid = [
        {"token_weight": weight, **_evaluate(
            tune, tune_data, tune_local, tune_patch,
            tune_token_data, tune_token, config, weight,
        )}
        for weight in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30)
    ]
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    confirmation = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch,
        confirm_token_data, confirm_token, config, selected["token_weight"],
    )
    baseline = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch,
        confirm_token_data, confirm_token, config, 0.0,
    )
    report = {
        "design": "top-50 DINO token cross-matcher added to the fixed strict final stack",
        "fixed_previous_config": config,
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "grid": grid,
    }
    Path("outputs/retrieval_v2/token_cross_final.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
