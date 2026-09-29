"""Combine adapted DINOv2 retrieval with the strict streaming pair verifier."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.probe_gallery_groups_official import _groups
from scripts.probe_multi_query_aggregation import CONFIRM_SEEDS, TUNE_SEEDS
from scripts.probe_streaming_official import _aggregate, _official_from_scores, _protocols
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from scripts.train_dino_patch_matcher import (
    _load as _load_aligned, _patch_features,
)
from scripts.train_streaming_top25_verifier import TOP, _local_for_protocol, _normalize
from src.reranking import database_side_augmentation


def _prepare(protocol, fused, osnet, parts):
    query = fused[protocol["qi"]]
    fused_gallery = database_side_augmentation(fused[protocol["gi"]], top_k=5, alpha=2.0)
    base = query @ fused_gallery.T
    candidates = np.argsort(-base, axis=1, kind="stable")[:, :TOP]
    raw_fused = query @ fused[protocol["gi"]].T
    part_scores = {}
    for name, embedding in parts.items():
        gallery = database_side_augmentation(embedding[protocol["gi"]], top_k=5, alpha=2.0)
        part_scores[name] = embedding[protocol["qi"]] @ gallery.T
    osnet_gallery = database_side_augmentation(osnet[protocol["gi"]], top_k=5, alpha=2.0)
    gallery_sources = {
        "fused": fused_gallery @ fused_gallery.T,
        "osnet": osnet_gallery @ osnet_gallery.T,
    }
    return {
        "base": base,
        "candidates": candidates,
        "qi": protocol["qi"],
        "gi": protocol["gi"],
        "raw_fused": raw_fused,
        "part_scores": part_scores,
        "gallery_sources": gallery_sources,
        "groups": {},
    }


def _patch_gallery_affinity(model, protocol, tokens, fused, osnet, dino, neighbors=16,
                            grid_size=None):
    indices = protocol["gi"]
    similarity = fused[indices] @ fused[indices].T
    np.fill_diagonal(similarity, -np.inf)
    candidate = np.argpartition(-similarity, neighbors - 1, axis=1)[:, :neighbors]
    query = np.repeat(indices, neighbors)
    gallery = indices[candidate.reshape(-1)]
    features = _patch_features(
        tokens, fused, osnet, dino, query, gallery, grid_size=grid_size
    )
    probability = model.predict_proba(features)[:, 1].reshape(len(indices), neighbors)
    affinity = np.zeros_like(similarity, dtype=np.float32)
    np.put_along_axis(affinity, candidate, probability, axis=1)
    # Requiring both directions removes visually plausible but asymmetric links.
    affinity = np.minimum(affinity, affinity.T)
    np.fill_diagonal(affinity, -np.inf)
    return affinity


def _predict_patch_plain(model, protocol, prepared, tokens, fused, osnet, dino,
                         grid_size=None):
    candidate = prepared["candidates"]
    query = np.repeat(prepared["qi"], TOP)
    gallery = prepared["gi"][candidate.reshape(-1)]
    features = _patch_features(
        tokens, fused, osnet, dino, query, gallery, grid_size=grid_size
    )
    return model.predict_proba(features)[:, 1].reshape(len(candidate), TOP)


def _scores(data, local, patch, verifier_weight, part_source="none", part_weight=0.0,
            patch_weight=0.0):
    score = data["base"].copy()
    candidates = data["candidates"]
    base = np.take_along_axis(score, candidates, axis=1)
    raw = np.take_along_axis(data["raw_fused"], candidates, axis=1)
    reranked = base + verifier_weight * (local - raw)
    if part_source != "none" and part_weight:
        part = np.take_along_axis(data["part_scores"][part_source], candidates, axis=1)
        reranked += part_weight * (part - raw)
    if patch_weight:
        mean = reranked.mean(1, keepdims=True)
        std = reranked.std(1, keepdims=True) + 1e-6
        base_z = (reranked - mean) / std
        patch_z = (patch - patch.mean(1, keepdims=True)) / (patch.std(1, keepdims=True) + 1e-6)
        reranked = mean + std * ((1.0 - patch_weight) * base_z + patch_weight * patch_z)
    np.put_along_axis(score, candidates, reranked, axis=1)
    return score


def _evaluate(protocols, data, local, patch, config):
    rows = []
    for protocol, item, prediction, patch_prediction in zip(
        protocols, data, local, patch, strict=True
    ):
        score = _scores(
            item, prediction, patch_prediction, config["verifier_weight"],
            config.get("part_source", "none"), config.get("part_weight", 0.0),
            config.get("patch_weight", 0.0),
        )
        if config["group_source"] == "patch_knn":
            affinity = item["gallery_sources"]["patch"]
            cache_key = ("patch_knn", config["neighbor_k"], config["threshold"])
            neighborhoods = item["groups"].get(cache_key)
            if neighborhoods is None:
                order = np.argsort(-affinity, axis=1, kind="stable")
                neighborhoods = []
                for index, row in enumerate(order[:, :config["neighbor_k"]]):
                    valid = row[affinity[index, row] >= config["threshold"]]
                    neighborhoods.append(np.concatenate(([index], valid)))
                item["groups"][cache_key] = neighborhoods
            original = score.copy()
            for index, neighborhood in enumerate(neighborhoods):
                support = original[:, neighborhood].max(axis=1)
                score[:, index] = (
                    (1.0 - config["propagation"]) * original[:, index]
                    + config["propagation"] * support
                )
        elif config["group_source"] != "none":
            key = (config["group_source"], config["neighbor_k"], config["threshold"])
            groups = item["groups"].get(key)
            if groups is None:
                groups = _groups(
                    item["gallery_sources"][config["group_source"]],
                    config["neighbor_k"], config["threshold"], True,
                )
                item["groups"][key] = groups
            for group in groups:
                family = score[:, group].max(axis=1, keepdims=True)
                score[:, group] = (
                    (1.0 - config["propagation"]) * score[:, group]
                    + config["propagation"] * family
                )
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
    return _aggregate(rows)


def main():
    frame = pd.read_csv("splits/val.csv")
    osnet = _normalize(np.load(
        "outputs/expert_fusion/osnet_smoothap_ensemble_val_embeddings.npy", allow_pickle=False
    ))
    with np.load("outputs/expert_fusion/cache/dinov2_vehicle_parts_val.npz", allow_pickle=False) as archive:
        dino = _normalize(archive["cls"])
        parts = {name: _normalize(archive[name]) for name in ("mean", "h2", "h4")}
    fused = _normalize(np.concatenate((np.sqrt(0.75) * osnet, np.sqrt(0.25) * dino), axis=1))
    conv_parts = _load("outputs/expert_fusion/cache/convnext_val_parts_3x3.npz", frame, "parts")
    osnet_parts = _load("outputs/expert_fusion/cache/osnet_smoothap_val_parts4.npz", frame, "parts")
    checkpoint = torch.load("weights/dino_top25_verifier.pt", map_location="cuda", weights_only=False)
    model = CrossImageTransformer(dim=96).cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tune = _protocols(frame, TUNE_SEEDS, {"base": fused})
    confirm = _protocols(frame, CONFIRM_SEEDS, {"base": fused})
    tune_data = [_prepare(p, fused, osnet, parts) for p in tune]
    confirm_data = [_prepare(p, fused, osnet, parts) for p in confirm]
    tune_local = [
        _local_for_protocol(model, p, d, conv_parts, osnet_parts, osnet)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_local = [
        _local_for_protocol(model, p, d, conv_parts, osnet_parts, osnet)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    patch_model = joblib.load("weights/dino_patch_matcher.joblib")
    patch_tokens = _load_aligned(
        "outputs/expert_fusion/cache/dinov2_vehicle_patches_val.npz", frame, "tokens"
    )
    tune_patch = [
        _predict_patch_plain(patch_model, p, d, patch_tokens, fused, osnet, dino)
        for p, d in zip(tune, tune_data, strict=True)
    ]
    confirm_patch = [
        _predict_patch_plain(patch_model, p, d, patch_tokens, fused, osnet, dino)
        for p, d in zip(confirm, confirm_data, strict=True)
    ]
    for protocol, data in zip(tune + confirm, tune_data + confirm_data, strict=True):
        data["gallery_sources"]["patch"] = _patch_gallery_affinity(
            patch_model, protocol, patch_tokens, fused, osnet, dino
        )

    # First choose query-to-candidate evidence, then tune the independent
    # gallery-only propagation.  This avoids a huge, validation-fragile grid.
    base_grid = []
    part_configs = [("none", 0.0)] + list(itertools.product(
        ("mean", "h2"), (0.025, 0.05, 0.10)
    ))
    for verifier_weight in (0.0, 0.025, 0.05, 0.075, 0.10):
        for part_source, part_weight in part_configs:
          for patch_weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            config = {
                "verifier_weight": verifier_weight,
                "part_source": part_source,
                "part_weight": part_weight,
                "patch_weight": patch_weight,
                "group_source": "none", "neighbor_k": 1,
                "threshold": 2.0, "propagation": 0.0,
            }
            base_grid.append({**config, **_evaluate(
                tune, tune_data, tune_local, tune_patch, config
            )})
    selected_base = max(base_grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))

    grid = []
    group_configs = [("none", 1, 2.0, 0.0)] + list(itertools.product(
        ("fused", "osnet"), (1, 2), (0.45, 0.50, 0.55, 0.60), (0.05, 0.10, 0.20, 0.30)
    ))
    group_configs += list(itertools.product(
        ("patch",), (2, 3, 4, 5), (0.05, 0.10, 0.15, 0.20, 0.30),
        (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
    ))
    group_configs += list(itertools.product(
        ("patch_knn",), (1, 2, 3, 5), (0.05, 0.10, 0.15, 0.20, 0.30),
        (0.10, 0.20, 0.30, 0.40, 0.50)
    ))
    for source, neighbor, threshold, propagation in group_configs:
        config = {
            **{key: selected_base[key] for key in (
                "verifier_weight", "part_source", "part_weight", "patch_weight"
            )},
            "group_source": source, "neighbor_k": neighbor,
            "threshold": threshold, "propagation": propagation,
        }
        grid.append({**config, **_evaluate(tune, tune_data, tune_local, tune_patch, config)})
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    keys = ("verifier_weight", "part_source", "part_weight", "patch_weight",
            "group_source", "neighbor_k", "threshold", "propagation")
    config = {key: selected[key] for key in keys}
    confirmation = _evaluate(confirm, confirm_data, confirm_local, confirm_patch, config)
    baseline_config = {
        **config, "verifier_weight": 0.0, "part_source": "none", "part_weight": 0.0,
        "patch_weight": 0.0, "group_source": "none", "propagation": 0.0,
    }
    baseline = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch, baseline_config
    )
    report = {
        "design": "adapted DINOv2+OSNet, retrained current-query verifier, local parts and static gallery propagation",
        "selected_base_tune": selected_base,
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {key: confirmation[key] - baseline[key] for key in ("mAP@10", "Rank-1", "Rank-5")},
        "top_tune": sorted(grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True)[:20],
        "top_base_tune": sorted(base_grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True)[:20],
    }
    Path("outputs/expert_fusion/dino_verifier_streaming.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
