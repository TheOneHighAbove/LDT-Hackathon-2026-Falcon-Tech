"""Use the learned token matcher as a symmetric static-gallery linker.

Every edge is scored in both directions and retained only when the two gallery
images occur in each other's shortlist.  This is a strict static-gallery
operation: neither the current query nor another query participates in edge
construction.
"""

from __future__ import annotations

import itertools
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
from scripts.probe_token_cross_final import _propagate_patch_knn
from scripts.train_cross_image_transformer import CrossImageTransformer, _load
from scripts.train_dino_patch_matcher import _load as _load_aligned
from scripts.train_dino_token_cross_top50 import (
    DEVICE,
    TOP,
    DinoTokenCrossMatcher,
    _base_protocol,
    _fuse,
    _normalize,
    _predict as _predict_token,
)
from scripts.train_streaming_top25_verifier import _local_for_protocol
from src.reranking import database_side_augmentation


def _gallery_protocol(indices, fused):
    raw = fused[indices]
    gallery = database_side_augmentation(raw, top_k=5, alpha=2.0)
    score = raw @ gallery.T
    np.fill_diagonal(score, -np.inf)
    candidate = np.argsort(-score, axis=1, kind="stable")[:, :TOP]
    return {
        "qi": indices,
        "gi": indices,
        "score": score.astype(np.float32),
        "candidates": candidate,
    }


def _symmetric_affinity(model, data, tokens, fused):
    prediction = _predict_token(model, data, tokens, fused)
    n = len(data["qi"])
    directed = np.full((n, n), -np.inf, dtype=np.float32)
    np.put_along_axis(directed, data["candidates"], prediction, axis=1)
    mutual = np.isfinite(directed) & np.isfinite(directed.T)
    affinity = np.where(mutual, 0.5 * (directed + directed.T), -np.inf)
    np.fill_diagonal(affinity, -np.inf)
    base_order = np.argsort(-data["score"], axis=1, kind="stable")
    base_rank = np.empty_like(base_order, dtype=np.int16)
    np.put_along_axis(
        base_rank,
        base_order,
        np.broadcast_to(np.arange(n, dtype=np.int16), base_order.shape),
        axis=1,
    )
    return affinity, base_rank


def _token_neighborhoods(affinity, base_rank, config):
    order = np.argsort(-affinity, axis=1, kind="stable")
    neighborhoods = []
    for index, row in enumerate(order[:, :config["neighbor_k"]]):
        keep = []
        for other in row:
            if not np.isfinite(affinity[index, other]):
                continue
            if affinity[index, other] < config["threshold"]:
                continue
            if max(base_rank[index, other], base_rank[other, index]) >= config["reciprocal_rank"]:
                continue
            keep.append(int(other))
        neighborhoods.append(np.asarray([index, *keep], dtype=np.int64))
    return neighborhoods


def _propagate(score, neighborhoods, amount):
    original = score.copy()
    width = max(len(value) for value in neighborhoods)
    index = np.arange(len(neighborhoods), dtype=np.int64)[:, None]
    padded = np.broadcast_to(index, (len(neighborhoods), width)).copy()
    for row, value in enumerate(neighborhoods):
        padded[row, :len(value)] = value
    support = original[:, padded].max(axis=2)
    return (1.0 - amount) * original + amount * support


def _group_stats(protocol, neighborhoods):
    labels = protocol["g"].vehicle_id.to_numpy()
    correct = predicted = 0
    nodes = set()
    for index, neighborhood in enumerate(neighborhoods):
        for other in neighborhood[1:]:
            if index < other:
                predicted += 1
                correct += int(labels[index] == labels[other])
                nodes.update((index, int(other)))
    return correct / max(predicted, 1), predicted, len(nodes) / len(labels)


def _evaluate(protocols, data, local, patch, token_data, token_prediction,
              gallery_link, previous, config):
    rows, stats = [], []
    for protocol, item, local_value, patch_value, token_item, token_value, link in zip(
        protocols, data, local, patch, token_data, token_prediction, gallery_link,
        strict=True,
    ):
        score = _scores(
            item, local_value, patch_value,
            previous["verifier_weight"], previous["part_source"],
            previous["part_weight"], previous["patch_weight"],
        )
        candidate = token_item["candidates"]
        base = np.take_along_axis(token_item["score"], candidate, axis=1)
        current = np.take_along_axis(score, candidate, axis=1)
        np.put_along_axis(
            score,
            candidate,
            current + config["query_token_weight"] * (token_value - base),
            axis=1,
        )
        if config["keep_patch_linker"]:
            score = _propagate_patch_knn(score, item, previous)
        neighborhoods = _token_neighborhoods(link[0], link[1], config)
        score = _propagate(score, neighborhoods, config["propagation"])
        rows.append(_official_from_scores(score, protocol["q"], protocol["g"]))
        stats.append(_group_stats(protocol, neighborhoods))
    result = _aggregate(rows)
    result.update({
        "edge_precision": float(np.mean([value[0] for value in stats])),
        "edge_count": float(np.mean([value[1] for value in stats])),
        "linked_fraction": float(np.mean([value[2] for value in stats])),
    })
    return result


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

    verifier_state = torch.load(
        "weights/dino_top25_verifier.pt", map_location=DEVICE, weights_only=False
    )
    verifier = CrossImageTransformer(dim=96).to(DEVICE)
    verifier.load_state_dict(verifier_state["model_state"])
    verifier.eval()
    patch_model = joblib.load("weights/dino_patch_matcher.joblib")
    token_state = torch.load(
        "weights/dino_token_cross_top50.pt", map_location=DEVICE, weights_only=False
    )
    token_model = DinoTokenCrossMatcher().to(DEVICE)
    token_model.load_state_dict(token_state["model_state"])
    token_model.eval()

    tune = _protocols(frame, TUNE_SEEDS, {"base": fused})
    confirm = _protocols(frame, CONFIRM_SEEDS, {"base": fused})
    tune_data = [_prepare(value, fused, osnet, parts) for value in tune]
    confirm_data = [_prepare(value, fused, osnet, parts) for value in confirm]
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
    tune_token = [_predict_token(token_model, d, tokens, fused) for d in tune_token_data]
    confirm_token = [_predict_token(token_model, d, tokens, fused) for d in confirm_token_data]

    gallery_data = [
        _gallery_protocol(protocol["gi"], fused) for protocol in tune + confirm
    ]
    gallery_link = [
        _symmetric_affinity(token_model, data, tokens, fused) for data in gallery_data
    ]
    tune_link, confirm_link = gallery_link[:len(tune)], gallery_link[len(tune):]
    for protocol, data in zip(tune + confirm, tune_data + confirm_data, strict=True):
        data["gallery_sources"]["patch"] = _patch_gallery_affinity(
            patch_model, protocol, tokens, fused, osnet, dino
        )

    previous_report = json.loads(Path(
        "outputs/expert_fusion/dino_verifier_streaming.json"
    ).read_text(encoding="utf-8"))
    previous_keys = (
        "verifier_weight", "part_source", "part_weight", "patch_weight",
        "group_source", "neighbor_k", "threshold", "propagation",
    )
    previous = {
        key: previous_report["selected_tune"][key] for key in previous_keys
    }

    grid = []
    options = itertools.product(
        (True, False),
        (1, 2),
        (1, 2, 4),
        (-np.inf, 0.45, 0.55, 0.65),
        (0.05, 0.10, 0.20, 0.30),
    )
    for keep_patch, neighbor_k, reciprocal_rank, threshold, propagation in options:
        config = {
            "query_token_weight": 0.25,
            "keep_patch_linker": keep_patch,
            "neighbor_k": neighbor_k,
            "reciprocal_rank": reciprocal_rank,
            "threshold": threshold,
            "propagation": propagation,
        }
        grid.append({**config, **_evaluate(
            tune, tune_data, tune_local, tune_patch, tune_token_data, tune_token,
            tune_link, previous, config,
        )})
    selected = max(grid, key=lambda row: (row["mAP@10"], row["Rank-1"], row["Rank-5"]))
    config_keys = (
        "query_token_weight", "keep_patch_linker", "neighbor_k",
        "reciprocal_rank", "threshold", "propagation",
    )
    config = {key: selected[key] for key in config_keys}
    confirmation = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch,
        confirm_token_data, confirm_token, confirm_link, previous, config,
    )
    baseline_config = {**config, "propagation": 0.0, "keep_patch_linker": True}
    baseline = _evaluate(
        confirm, confirm_data, confirm_local, confirm_patch,
        confirm_token_data, confirm_token, confirm_link, previous, baseline_config,
    )
    report = {
        "design": "bidirectional mutual top-50 token matcher as a static-gallery linker",
        "selected_tune": selected,
        "baseline_confirmation": baseline,
        "confirmation": confirmation,
        "delta": {
            key: confirmation[key] - baseline[key]
            for key in ("mAP@10", "Rank-1", "Rank-5")
        },
        "top_tune": sorted(
            grid, key=lambda row: (row["mAP@10"], row["Rank-1"]), reverse=True
        )[:30],
    }
    Path("outputs/retrieval_v2/token_gallery_linker.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
